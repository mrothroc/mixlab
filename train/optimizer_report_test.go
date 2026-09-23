package train

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/gpu"
)

func optimizerReportForTest(t *testing.T, cfg *ArchConfig, shapes []WeightShape) (optimizerReport, gpu.TrainerOptimizerSpec) {
	t.Helper()
	if shapes == nil {
		var err error
		shapes, err = computeWeightShapes(cfg)
		if err != nil {
			t.Fatal(err)
		}
	}
	spec, err := buildTrainerOptimizerSpec(cfg, shapes)
	if err != nil {
		t.Fatal(err)
	}
	r, err := resolvedOptimizerReport(spec, shapes)
	if err != nil {
		t.Fatal(err)
	}
	return r, spec
}

func TestOptimizerReportViTCoverage(t *testing.T) {
	cfg, err := LoadArchConfig("../examples/vit_pytorch_init.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg.Training.Optimizer = "muon"
	cfg.Training.MatrixLR = 0.01
	r, _ := optimizerReportForTest(t, cfg, nil)
	wantCounts := map[string]int{"embed": 0, "matrix": 39, "scalar": 45, "head": 2}
	wantParams := map[string]int64{"embed": 0, "matrix": 9495552, "scalar": 23040, "head": 5130}
	for _, g := range r.Groups {
		if g.Tensors != wantCounts[g.Name] || g.Parameters != wantParams[g.Name] {
			t.Fatalf("wrong coverage: %+v", g)
		}
		if g.Name == "matrix" && (g.Optimizer != "muon" || *g.ConfiguredLR != 0.01) {
			t.Fatalf("wrong matrix settings: %+v", g)
		}
	}
	found := 0
	for _, w := range r.Tensors {
		switch w.Name {
		case "position_embeddings", "cls_token", "input_adapter_proj":
			found++
			if w.Group != "matrix" || w.Optimizer != "muon" {
				t.Fatalf("legacy routing changed: %+v", w)
			}
		}
	}
	if found != 3 || !strings.Contains(r.summary(), "embed=unused (0 tensors)") {
		t.Fatalf("missing routing diagnostics: %s", r.summary())
	}
}

func TestOptimizerReportMatchesResolvedSpec(t *testing.T) {
	for _, optimizer := range []string{"muon", "muon_eq_r", "normuon", "adamw", "lamb"} {
		t.Run(optimizer, func(t *testing.T) {
			cfg, err := ParseArchConfig([]byte(fmt.Sprintf(`{"model_dim":16,"vocab_size":32,"seq_len":4,
				"blocks":[{"type":"plain","heads":2}],"training":{"batch_tokens":4,"optimizer":%q,
				"lr":0.001,"matrix_lr":0,"scalar_lr":0.002,"weight_decay":0.1,"weight_decay_policy":"all"}}`, optimizer)), "report")
			if err != nil {
				t.Fatal(err)
			}
			shapes := []WeightShape{
				{Name: "embed", Shape: []int{32, 16}},
				{Name: "wq", Shape: []int{16, 16}},
				{Name: "norm_scale", Shape: []int{16}, IsNormScale: true, ForceNoDecay: true},
				{Name: "head_bias", Shape: []int{32}, ForceDecay: true},
				{Name: "A_log", Shape: []int{2}, OptimizerRole: "ssm_state", OptimizerLR: 0.0002, ForceNoDecay: true},
				{Name: "s4d_C_real", Shape: []int{16, 8}, OptimizerRole: "s4d_main"},
				{Name: "sobolev", Shape: []int{16}, OptimizerRole: "s4d_sobolev", OptimizerLR: 0.0003, OptimizerWeightDecay: 0.2, ForceDecay: true},
				{Name: "buffer", Shape: []int{16}, IsBuffer: true},
				{Name: "frozen", Shape: []int{16, 16}, Frozen: true},
			}
			r, spec := optimizerReportForTest(t, cfg, shapes)
			if r.BufferTensors != 1 || r.FrozenTensors != 1 {
				t.Fatal("inactive tensors miscounted")
			}
			for i, row := range r.Tensors {
				w := spec.Weights[i]
				if row.Index != i || row.Name != shapes[i].Name || !reflect.DeepEqual(row.Shape, shapes[i].Shape) {
					t.Fatalf("identity mismatch: %+v", row)
				}
				if w.Frozen {
					if row.GroupIndex != nil || row.ConfiguredLR != nil || row.EffectiveWeightDecay != 0 {
						t.Fatalf("inactive tensor assigned optimizer: %+v", row)
					}
					continue
				}
				g := spec.Groups[w.GroupIndex]
				if row.GroupIndex == nil || *row.GroupIndex != w.GroupIndex || *row.ConfiguredLR != g.LR ||
					*row.ConfiguredWeightDecay != g.WeightDecay || row.DecayEligible != w.Decay {
					t.Fatalf("spec mismatch: %+v vs %+v", row, g)
				}
				wantDecay := float32(0)
				if w.Decay {
					wantDecay = g.WeightDecay
				}
				if row.EffectiveWeightDecay != wantDecay {
					t.Fatalf("effective decay mismatch: %+v", row)
				}
			}
			if r.Tensors[1].Optimizer != optimizer || *r.Tensors[1].ConfiguredLR != 0 {
				t.Fatal("lost optimizer variant or explicit zero")
			}
			if !strings.HasPrefix(r.Tensors[4].Group, "extra:ssm_state_") || r.Tensors[4].EffectiveWeightDecay != 0 {
				t.Fatal("lost specialized state group")
			}
			// Diagnostic metadata must never alter resumable optimizer hashes.
			before, err := optimizerSpecHash(spec)
			if err != nil {
				t.Fatal(err)
			}
			for i := range spec.Groups {
				spec.Groups[i].ReportName = ""
			}
			after, err := optimizerSpecHash(spec)
			if err != nil || before != after {
				t.Fatal("diagnostics changed optimizer hash")
			}
		})
	}
}

func TestOptimizerReportSharing(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{"model_dim":16,"vocab_size":32,"seq_len":4,
		"blocks":[{"type":"plain","heads":2,"weight_group":"shared"},
		{"type":"plain","heads":2,"weight_group":"shared"}],"training":{"batch_tokens":4}}`), "shared_report")
	if err != nil {
		t.Fatal(err)
	}
	r, _ := optimizerReportForTest(t, cfg, nil)
	cfg.Blocks = cfg.Blocks[:1]
	single, _ := optimizerReportForTest(t, cfg, nil)
	if !reflect.DeepEqual(r, single) {
		t.Fatal("shared tensors counted more than once")
	}
}

func TestOptimizerReportFinalOverridesAndErrors(t *testing.T) {
	shapes := []WeightShape{{Name: "embed", Shape: []int{2, 4}}, {Name: "wq", Shape: []int{4, 4}}}
	spec := gpu.TrainerOptimizerSpec{
		Groups:  []gpu.OptimizerGroup{{Kind: gpu.OptimizerSGD, LR: 0.02, WeightDecay: 0.3}},
		Weights: []gpu.WeightOptimizer{{GroupIndex: 0, Decay: false}, {Frozen: true}},
	}
	r, err := resolvedOptimizerReport(spec, shapes)
	if err != nil {
		t.Fatal(err)
	}
	if r.Tensors[0].Group != "group_0" || r.Tensors[0].Optimizer != "sgd" || !r.Tensors[1].Frozen {
		t.Fatal("report reclassified final overrides")
	}
	if _, err := resolvedOptimizerReport(spec, shapes[:1]); err == nil {
		t.Fatal("accepted missing assignments")
	}
	spec.Weights[0].GroupIndex = 99
	if _, err := resolvedOptimizerReport(spec, shapes); err == nil {
		t.Fatal("accepted invalid group")
	}
}

func TestRunOptimizerReport(t *testing.T) {
	var first, second bytes.Buffer
	for _, out := range []*bytes.Buffer{&first, &second} {
		if err := RunOptimizerReport("../examples/vit_pytorch_init.json", out); err != nil {
			t.Fatal(err)
		}
	}
	if !bytes.Equal(first.Bytes(), second.Bytes()) {
		t.Fatal("nondeterministic report")
	}
	var r optimizerReport
	if err := json.Unmarshal(first.Bytes(), &r); err != nil {
		t.Fatal(err)
	}
	if r.Schema != "mixlab_optimizer_report_v1" || len(r.Tensors) != 86 || r.RateBasis != "configured_before_schedule" {
		t.Fatal("unexpected JSON schema")
	}
	if err := RunOptimizerReport("", &first); err == nil || !strings.Contains(err.Error(), "-config") {
		t.Fatal("missing config error")
	}
	if err := RunOptimizerReport("missing-config.json", &first); err == nil {
		t.Fatal("missing file accepted")
	}
	if err := RunOptimizerReport("../examples/vit_pytorch_init.json", failingReportWriter{}); err == nil {
		t.Fatal("write error swallowed")
	}
}

type failingReportWriter struct{}

func (failingReportWriter) Write([]byte) (int, error) { return 0, errors.New("write failed") }
