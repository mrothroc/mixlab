package train

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/mrothroc/mixlab/gpu"
)

type optimizerReport struct {
	Schema        string                  `json:"schema"`
	RateBasis     string                  `json:"rate_basis"`
	Groups        []optimizerReportGroup  `json:"groups"`
	Tensors       []optimizerReportTensor `json:"tensors"`
	FrozenTensors int                     `json:"frozen_tensors"`
	BufferTensors int                     `json:"buffer_tensors"`
}

type optimizerReportGroup struct {
	Index                 *int     `json:"index,omitempty"`
	Name                  string   `json:"name"`
	Optimizer             string   `json:"optimizer,omitempty"`
	ConfiguredLR          *float32 `json:"configured_lr,omitempty"`
	ConfiguredWeightDecay *float32 `json:"configured_weight_decay,omitempty"`
	Tensors               int      `json:"tensors"`
	Parameters            int64    `json:"parameters"`
	DecayEligibleTensors  int      `json:"decay_eligible_tensors"`
}

type optimizerReportTensor struct {
	Index                 int      `json:"index"`
	Name                  string   `json:"name"`
	Shape                 []int    `json:"shape"`
	Parameters            int64    `json:"parameters"`
	GroupIndex            *int     `json:"group_index,omitempty"`
	Group                 string   `json:"group,omitempty"`
	Optimizer             string   `json:"optimizer,omitempty"`
	ConfiguredLR          *float32 `json:"configured_lr,omitempty"`
	ConfiguredWeightDecay *float32 `json:"configured_weight_decay,omitempty"`
	EffectiveWeightDecay  float32  `json:"effective_weight_decay"`
	DecayEligible         bool     `json:"decay_eligible"`
	Frozen                bool     `json:"frozen"`
	Buffer                bool     `json:"buffer"`
}

// RunOptimizerReport resolves config policy only: no data, checkpoint, or GPU is loaded.
func RunOptimizerReport(configPath string, out io.Writer) error {
	if configPath == "" {
		return fmt.Errorf("-config is required for optimizer-report mode")
	}
	cfg, err := LoadArchConfig(configPath)
	if err != nil {
		return err
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		return err
	}
	spec, err := buildTrainerOptimizerSpec(cfg, shapes)
	if err != nil {
		return err
	}
	report, err := resolvedOptimizerReport(spec, shapes)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(out)
	enc.SetIndent("", "  ")
	return enc.Encode(report)
}

// Consume the final assignment rather than reproducing name/rank routing rules.
func resolvedOptimizerReport(spec gpu.TrainerOptimizerSpec, shapes []WeightShape) (optimizerReport, error) {
	r := optimizerReport{Schema: "mixlab_optimizer_report_v1", RateBasis: "configured_before_schedule",
		Tensors: make([]optimizerReportTensor, 0, len(shapes))}
	if len(spec.Weights) != len(shapes) {
		return r, fmt.Errorf("optimizer report: %d assignments for %d tensors", len(spec.Weights), len(shapes))
	}
	for _, name := range []string{"embed", "head", "scalar", "matrix"} {
		r.Groups = append(r.Groups, optimizerReportGroup{Name: name})
	}
	groupRows := make([]int, len(spec.Groups))
	for i, g := range spec.Groups {
		name := g.ReportName
		if name == "" {
			name = fmt.Sprintf("group_%d", i)
		}
		row := len(r.Groups)
		for j, existing := range r.Groups {
			if existing.Name == name && existing.Index == nil {
				row = j
				break
			}
		}
		kind, err := reportOptimizerName(g)
		if err != nil {
			return r, err
		}
		entry := optimizerReportGroup{Index: &i, Name: name, Optimizer: kind,
			ConfiguredLR: &g.LR, ConfiguredWeightDecay: &g.WeightDecay}
		if row == len(r.Groups) {
			r.Groups = append(r.Groups, entry)
		} else {
			r.Groups[row] = entry
		}
		groupRows[i] = row
	}
	for i, s := range shapes {
		w := spec.Weights[i]
		n := int64(1)
		for _, dim := range s.Shape {
			n *= int64(dim)
		}
		t := optimizerReportTensor{Index: i, Name: s.Name, Shape: s.Shape, Parameters: n,
			Frozen: w.Frozen, Buffer: s.IsBuffer}
		if w.Frozen {
			if s.IsBuffer {
				r.BufferTensors++
			} else {
				r.FrozenTensors++
			}
		} else {
			if w.GroupIndex < 0 || w.GroupIndex >= len(groupRows) {
				return r, fmt.Errorf("optimizer report: tensor %d has invalid group %d", i, w.GroupIndex)
			}
			g := &r.Groups[groupRows[w.GroupIndex]]
			g.Tensors++
			g.Parameters += n
			t.GroupIndex, t.Group, t.Optimizer = g.Index, g.Name, g.Optimizer
			t.ConfiguredLR, t.ConfiguredWeightDecay = g.ConfiguredLR, g.ConfiguredWeightDecay
			t.DecayEligible = w.Decay
			if w.Decay {
				g.DecayEligibleTensors++
				t.EffectiveWeightDecay = *g.ConfiguredWeightDecay
			}
		}
		r.Tensors = append(r.Tensors, t)
	}
	return r, nil
}

func reportOptimizerName(g gpu.OptimizerGroup) (string, error) {
	switch g.Kind {
	case gpu.OptimizerAdamW:
		return "adamw", nil
	case gpu.OptimizerLAMB:
		return "lamb", nil
	case gpu.OptimizerSGD:
		return "sgd", nil
	case gpu.OptimizerMuon:
		switch g.MuonNormalization {
		case gpu.MuonNormalizationNorMuon:
			return "normuon", nil
		case gpu.MuonNormalizationRowL2:
			return "muon_eq_r", nil
		}
		if g.RowNormalize {
			return "muon_eq_r", nil
		}
		return "muon", nil
	default:
		return "", fmt.Errorf("optimizer report: unknown optimizer kind %d", g.Kind)
	}
}

func (r optimizerReport) summary() string {
	parts := make([]string, 0, len(r.Groups))
	for _, g := range r.Groups {
		if g.Index == nil {
			parts = append(parts, g.Name+"=unused (0 tensors)")
			continue
		}
		parts = append(parts, fmt.Sprintf("%s=%s lr=%g wd=%g (%d tensors, %d params, %d decay-eligible)",
			g.Name, g.Optimizer, *g.ConfiguredLR, *g.ConfiguredWeightDecay, g.Tensors, g.Parameters, g.DecayEligibleTensors))
	}
	return fmt.Sprintf("optimizer (configured rates, before schedule): %s | frozen=%d buffers=%d",
		strings.Join(parts, " | "), r.FrozenTensors, r.BufferTensors)
}
