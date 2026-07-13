package arch

import (
	"fmt"
	"strings"
)

// TTTMLPStateLayout describes one block's persistent inference state and the
// checkpoint weights used to initialize it.
type TTTMLPStateLayout struct {
	BlockIndex           int
	Heads                int
	HeadDim              int
	HiddenDim            int
	ChunkSize            int
	StateSize            int
	InitialWeightIndices [4]int
	StateInput           string
	GradientInput        string
	ConvInput            string
	StateOutput          string
	GradientOutput       string
	ConvOutput           string
}

// BuildTTTMLPStatefulInferenceIRProgram builds a batch-one, inference-only
// graph whose TTT recurrent state is supplied and returned explicitly. The
// graph accepts one chunk fragment and deliberately rejects other token mixers
// until they have their own cache contracts.
func BuildTTTMLPStatefulInferenceIRProgram(cfg *ArchConfig, tokenCount int, offsets []int) (*Program, []TTTMLPStateLayout, error) {
	if cfg == nil {
		return nil, nil, fmt.Errorf("nil config")
	}
	if tokenCount <= 0 {
		return nil, nil, fmt.Errorf("tokenCount=%d must be > 0", tokenCount)
	}
	if err := validateTTTMLPStatefulInferenceConfig(cfg); err != nil {
		return nil, nil, err
	}
	tttCount := 0
	for _, block := range cfg.Blocks {
		if blockTypeKey(block) == "ttt_mlp" {
			tttCount++
		}
	}
	if len(offsets) != tttCount {
		return nil, nil, fmt.Errorf("TTT state offset count=%d, want %d", len(offsets), tttCount)
	}

	metas, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		return nil, nil, err
	}
	prog := NewProgram(len(metas))
	prog.DeclareInput("tokens", TensorInt32, []int{1, tokenCount})
	prog.Embed(weightName(0), "tokens", "x_embed")
	prog.Reshape("x_embed", []int{tokenCount, cfg.ModelDim}, "x")

	norm := cfg.EffectiveNormSpec()
	wi := fixedWeightCountWithHeadAndNorm(cfg.ReservesUntiedHeadWeight(), norm)
	layouts := make([]TTTMLPStateLayout, 0, tttCount)
	tttOrdinal := 0
	for blockIndex, block := range cfg.Blocks {
		if blockTypeKey(block) == "ttt_mlp" {
			offset := offsets[tttOrdinal]
			layout, nextWI, err := emitTTTMLPStatefulInferenceBlock(prog, block, wi, cfg.ModelDim, tokenCount, blockIndex, tttOrdinal, offset, cfg.BlockScales)
			if err != nil {
				return nil, nil, err
			}
			layouts = append(layouts, layout)
			wi = nextWI
			tttOrdinal++
			continue
		}
		wi, err = EmitBlock(prog, block, "x", wi, cfg.ModelDim, tokenCount, 1, cfg.VocabSize, blockIndex, EmitOptions{
			MLPMult:             cfg.EffectiveMLPMult(),
			BlockScales:         cfg.BlockScales,
			Norm:                norm,
			NormPlacement:       cfg.EffectiveNormPlacement(),
			FFNInternalNorm:     cfg.FFNInternalNorm,
			PositionalEmbedding: cfg.EffectivePositionalEmbedding(),
			BlockIndex:          blockIndex,
		})
		if err != nil {
			return nil, nil, fmt.Errorf("emit blocks[%d]: %w", blockIndex, err)
		}
	}
	if wi != len(metas) {
		return nil, nil, fmt.Errorf("TTT stateful weight layout consumed %d of %d weights", wi, len(metas))
	}

	if _, err := emitNamedNormIR(prog, "x", finalNormWeightIndexWithHeadAndNorm(cfg.ReservesUntiedHeadWeight(), norm), "x_final_norm", norm); err != nil {
		return nil, nil, err
	}
	if cfg.TieEmbeddings {
		prog.Transpose(weightName(0), []int{1, 0}, "tied_head")
		prog.MatMul("x_final_norm", "tied_head", "logits_raw")
	} else {
		prog.MatMul("x_final_norm", weightName(1), "logits_raw")
	}
	logits := "logits_raw"
	if cfg.LogitSoftcap > 0 {
		prog.ScalarMul(logits, 1/cfg.LogitSoftcap, "logits_scaled")
		prog.Tanh("logits_scaled", "logits_tanh")
		prog.ScalarMul("logits_tanh", cfg.LogitSoftcap, "logits")
		logits = "logits"
	} else {
		prog.ScalarMul(logits, 1, "logits")
		logits = "logits"
	}
	_ = logits
	prog.DeclareOutput("logits", TensorFloat32, []int{tokenCount, cfg.VocabSize})
	return prog, layouts, nil
}

func validateTTTMLPStatefulInferenceConfig(cfg *ArchConfig) error {
	if cfg.Training.EffectiveObjective() != ObjectiveCausal {
		return fmt.Errorf("TTT stateful inference requires training.objective=causal")
	}
	if cfg.EffectivePositionalEmbedding() != PositionalEmbeddingRope {
		return fmt.Errorf("TTT stateful inference requires positional_embedding=rope")
	}
	if cfg.UNet || cfg.ParallelResidual || cfg.ResidMix || len(cfg.Recurrence) > 0 || len(cfg.RecurrencePhases) > 0 {
		return fmt.Errorf("TTT stateful inference does not support U-Net, parallel, resid_mix, or recurrence execution")
	}
	if cfg.CharVocabSize > 0 || cfg.BigramVocabSize > 0 || cfg.TrigramVocabSize > 0 || cfg.SmearEmbeddings {
		return fmt.Errorf("TTT stateful inference does not support embedding feature channels in v1")
	}
	if cfg.Backout != nil || cfg.MTP != nil || cfg.EffectiveMLMHead() != MLMHeadLinear || normalizeLayerAggregation(cfg.LayerAggregation) != LayerAggregationNone {
		return fmt.Errorf("TTT stateful inference does not support auxiliary heads, backout, or layer aggregation in v1")
	}
	found := false
	for i, block := range cfg.Blocks {
		kind := blockTypeKey(block)
		switch kind {
		case "ttt_mlp":
			found = true
		case "swiglu", "geglu", "mlp":
		default:
			return fmt.Errorf("TTT stateful inference blocks[%d] type=%q is not pointwise and has no state cache", i, block.Type)
		}
		if strings.TrimSpace(block.WeightGroup) != "" || block.ParallelGroup > 0 || block.ParallelResidual != nil {
			return fmt.Errorf("TTT stateful inference blocks[%d] cannot share or group weights", i)
		}
	}
	if !found {
		return fmt.Errorf("TTT stateful inference requires at least one ttt_mlp block")
	}
	return nil
}

func emitTTTMLPStatefulInferenceBlock(prog *Program, spec BlockSpec, wi, modelDim, tokenCount, blockIndex, ordinal, offset int, blockScales bool) (TTTMLPStateLayout, int, error) {
	headDim := modelDim / spec.Heads
	hidden, err := effectiveTTTMLPInnerHiddenDim(spec, modelDim)
	if err != nil {
		return TTTMLPStateLayout{}, wi, err
	}
	chunk := effectiveTTTMLPChunkSize(spec)
	if offset < 0 || offset >= chunk || offset+tokenCount > chunk {
		return TTTMLPStateLayout{}, wi, fmt.Errorf("blocks[%d] state offset=%d plus tokenCount=%d crosses chunk_size=%d", blockIndex, offset, tokenCount, chunk)
	}
	stateSize := spec.Heads * (2*headDim*hidden + hidden + headDim)
	prefix := fmt.Sprintf("ttt_state_%d", ordinal)
	layout := TTTMLPStateLayout{
		BlockIndex:           blockIndex,
		Heads:                spec.Heads,
		HeadDim:              headDim,
		HiddenDim:            hidden,
		ChunkSize:            chunk,
		StateSize:            stateSize,
		InitialWeightIndices: [4]int{wi + 10, wi + 11, wi + 12, wi + 13},
		StateInput:           prefix + "_mlp",
		GradientInput:        prefix + "_grad",
		ConvInput:            prefix + "_conv",
		StateOutput:          prefix + "_mlp_next",
		GradientOutput:       prefix + "_grad_next",
		ConvOutput:           prefix + "_conv_next",
	}
	prog.DeclareInput(layout.StateInput, TensorFloat32, []int{1, stateSize})
	prog.DeclareInput(layout.GradientInput, TensorFloat32, []int{1, stateSize})
	prog.DeclareInput(layout.ConvInput, TensorFloat32, []int{1, 2, 3, modelDim})

	blockPrefix := fmt.Sprintf("ttt_stateful_block_%d", blockIndex)
	prog.RMSNorm("x", weightName(wi), blockPrefix+"_x_norm", 1e-6)
	prog.MatMul(blockPrefix+"_x_norm", weightName(wi+1), blockPrefix+"_qk")
	prog.MatMul(blockPrefix+"_x_norm", weightName(wi+6), blockPrefix+"_v")
	prog.MatMul(blockPrefix+"_x_norm", weightName(wi+7), blockPrefix+"_lr_raw")
	prog.Add(blockPrefix+"_lr_raw", weightName(wi+8), blockPrefix+"_lr")
	prog.Full([]int{1}, 1, blockPrefix+"_lr_scale")
	prog.TTTMLPStatefulScan(
		blockPrefix+"_qk", blockPrefix+"_v", blockPrefix+"_lr",
		weightName(wi+2), weightName(wi+3), weightName(wi+4), weightName(wi+5),
		blockPrefix+"_lr_scale", weightName(wi+9), weightName(wi+14), weightName(wi+15),
		layout.StateInput, layout.GradientInput, layout.ConvInput,
		blockPrefix+"_scan", layout.StateOutput, layout.GradientOutput, layout.ConvOutput,
		1, tokenCount, spec.Heads, headDim, hidden, chunk, offset, float32(effectiveTTTMLPInnerLRBase(spec)))
	prog.Reshape(blockPrefix+"_scan", []int{tokenCount, modelDim}, blockPrefix+"_merged")
	prog.LayerNorm(blockPrefix+"_merged", weightName(wi+16), weightName(wi+17), blockPrefix+"_post_norm", 1e-6)
	prog.MatMul(blockPrefix+"_x_norm", weightName(wi+18), blockPrefix+"_gate_raw")
	prog.GELU(blockPrefix+"_gate_raw", blockPrefix+"_gate")
	prog.Mul(blockPrefix+"_post_norm", blockPrefix+"_gate", blockPrefix+"_gated")
	prog.MatMul(blockPrefix+"_gated", weightName(wi+19), blockPrefix+"_delta")
	nextWI := wi + 20
	delta := blockPrefix + "_delta"
	if blockScales {
		prog.Mul(delta, weightName(nextWI), blockPrefix+"_scaled")
		delta = blockPrefix + "_scaled"
		nextWI++
	}
	prog.Add("x", delta, "x")
	prog.DeclareOutput(layout.StateOutput, TensorFloat32, []int{1, stateSize})
	prog.DeclareOutput(layout.GradientOutput, TensorFloat32, []int{1, stateSize})
	prog.DeclareOutput(layout.ConvOutput, TensorFloat32, []int{1, 2, 3, modelDim})
	return layout, nextWI, nil
}
