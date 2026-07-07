package arch

import (
	"fmt"
	"strings"
)

const (
	LayerAggregationNone = "none"
	LayerAggregationDWA  = "dwa"

	dwaAlphaInitMode = "dwa_alpha"
)

func normalizeLayerAggregation(raw string) string {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "none", "off", "disabled", "false":
		return LayerAggregationNone
	case "dwa", "dense_weighted_aggregation", "dense-weighted-aggregation":
		return LayerAggregationDWA
	default:
		return strings.ToLower(strings.TrimSpace(raw))
	}
}

func (cfg *ArchConfig) EffectiveLayerAggregation() string {
	if cfg == nil {
		return LayerAggregationNone
	}
	return normalizeLayerAggregation(cfg.LayerAggregation)
}

func validateLayerAggregation(cfg *ArchConfig, source string) error {
	mode := normalizeLayerAggregation(cfg.LayerAggregation)
	if strings.TrimSpace(cfg.LayerAggregation) != "" {
		cfg.LayerAggregation = mode
	}
	switch mode {
	case LayerAggregationNone:
		return nil
	case LayerAggregationDWA:
	default:
		return fmt.Errorf("config %q has invalid layer_aggregation=%q (must be \"none\" or \"dwa\")", source, cfg.LayerAggregation)
	}
	if cfg.UNet {
		return fmt.Errorf("config %q layer_aggregation=\"dwa\" is not supported with unet", source)
	}
	if cfg.ParallelResidual {
		return fmt.Errorf("config %q layer_aggregation=\"dwa\" is not supported with parallel_residual", source)
	}
	if len(cfg.Recurrence) > 0 || len(cfg.RecurrencePhases) > 0 || cfg.executionOrderSet {
		return fmt.Errorf("config %q layer_aggregation=\"dwa\" is not supported with recurrence or custom execution order in v1", source)
	}
	for i, block := range cfg.Blocks {
		if block.ParallelGroup > 0 {
			return fmt.Errorf("config %q layer_aggregation=\"dwa\" is not supported with blocks[%d].parallel_group", source, i)
		}
		if block.ParallelResidual != nil && *block.ParallelResidual {
			return fmt.Errorf("config %q layer_aggregation=\"dwa\" is not supported with blocks[%d].parallel_residual", source, i)
		}
		if block.KVSource > 0 {
			return fmt.Errorf("config %q layer_aggregation=\"dwa\" is not supported with blocks[%d].kv_source in v1", source, i)
		}
		if block.SkipAttention {
			return fmt.Errorf("config %q layer_aggregation=\"dwa\" is not supported with blocks[%d].skip_attention in v1", source, i)
		}
		switch blockTypeKey(block) {
		case "plain", "swiglu", "geglu", "mlp", "moe":
		default:
			return fmt.Errorf("config %q layer_aggregation=\"dwa\" does not support blocks[%d].type=%q in v1", source, i, block.Type)
		}
	}
	return nil
}

func layerAggregationWeightShapes(blocks []BlockSpec, mode string) ([]WeightMeta, error) {
	if normalizeLayerAggregation(mode) != LayerAggregationDWA {
		return nil, nil
	}
	n, err := dwaSublayerCount(blocks)
	if err != nil {
		return nil, err
	}
	metas := make([]WeightMeta, 0, n)
	for i := 0; i < n; i++ {
		metas = append(metas, WeightMeta{
			Name:     fmt.Sprintf("dwa_alpha_%d", i),
			Shape:    []int{i + 2},
			InitMode: dwaAlphaInitMode,
		})
	}
	return metas, nil
}

func dwaSublayerCount(blocks []BlockSpec) (int, error) {
	plan, err := newParallelResidualPlan(blocks, false)
	if err != nil {
		return 0, err
	}
	total := 0
	for i := 0; i < len(blocks); {
		if plan.startsAt(i) {
			groupLen := plan.groupLenAt(i)
			total += groupLen
			i += groupLen
			continue
		}
		block := blocks[i]
		switch blockTypeKey(block) {
		case "plain":
			total += 2
		case "swiglu", "geglu", "mlp", "moe":
			total++
		default:
			return 0, fmt.Errorf("layer_aggregation=\"dwa\" does not support blocks[%d].type=%q", i, block.Type)
		}
		i++
	}
	return total, nil
}

type layerAggregationBuildState struct {
	weightStart int
	step        int
	history     []string
	captureOnly bool
}

func newLayerAggregationBuildState(prog *Program, mode string, weightStart int, stream string) *layerAggregationBuildState {
	if normalizeLayerAggregation(mode) != LayerAggregationDWA {
		return nil
	}
	static := "dwa_static_embeddings"
	prog.ScalarMul(stream, 1.0, static)
	return &layerAggregationBuildState{
		weightStart: weightStart,
		history:     []string{static},
	}
}

func newLayerAggregationCaptureState(prog *Program, stream string) *layerAggregationBuildState {
	static := "dwa_static_embeddings"
	prog.ScalarMul(stream, 1.0, static)
	return &layerAggregationBuildState{
		history:     []string{static},
		captureOnly: true,
	}
}

func (d *layerAggregationBuildState) apply(prog *Program, stream string) {
	if d == nil {
		return
	}
	state := fmt.Sprintf("dwa_state_%d", d.step)
	prog.ScalarMul(stream, 1.0, state)
	d.history = append(d.history, state)
	if d.captureOnly {
		d.step++
		return
	}

	alpha := weightName(d.weightStart + d.step)
	sum := ""
	for i, hist := range d.history {
		alphaPart := fmt.Sprintf("dwa_%d_alpha_%d", d.step, i)
		term := fmt.Sprintf("dwa_%d_term_%d", d.step, i)
		// 1-D weights are materialized as [1, n] row vectors, so the per-element
		// alpha scalar lives on axis 1; slicing it to [1, 1] broadcasts against
		// the [B*T, D] hidden state.
		prog.Slice(alpha, i, i+1, 1, 1, alphaPart)
		prog.Mul(hist, alphaPart, term)
		if i == 0 {
			sum = term
			continue
		}
		out := fmt.Sprintf("dwa_%d_sum_%d", d.step, i)
		if i == len(d.history)-1 {
			out = stream
		}
		prog.Add(sum, term, out)
		sum = out
	}
	d.step++
}

func emitLayerAggregationFromHistory(prog *Program, history []string, alphaStart int, rowStart, rows int, outPrefix string) (string, error) {
	if len(history) < 2 {
		return "", fmt.Errorf("layer aggregation history is empty")
	}
	current := ""
	for i, hist := range history {
		hSlice := fmt.Sprintf("%s_hist_%d", outPrefix, i)
		alphaPart := fmt.Sprintf("%s_alpha_%d", outPrefix, i)
		term := fmt.Sprintf("%s_term_%d", outPrefix, i)
		prog.Slice(hist, rowStart, rowStart+rows, 1, 0, hSlice)
		prog.Slice(weightName(alphaStart), i, i+1, 1, 1, alphaPart)
		prog.Mul(hSlice, alphaPart, term)
		if i == 0 {
			current = term
			continue
		}
		next := fmt.Sprintf("%s_sum_%d", outPrefix, i)
		prog.Add(current, term, next)
		current = next
	}
	return current, nil
}
