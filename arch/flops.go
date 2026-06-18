package arch

import (
	"fmt"
	"strings"
)

// FLOPsEstimate summarizes analytical floating-point operation counts for an
// architecture and training batch.
type FLOPsEstimate struct {
	ForwardFLOPs       int64 // FLOPs per forward pass
	TrainingFLOPs      int64 // FLOPs per training step (3x forward)
	FLOPsPerToken      int64 // TrainingFLOPs / batch_tokens
	ParamCount         int64 // Unique parameter count after sharing
	ExpandedParamCount int64 // Parameter count with recurrent/shared blocks expanded
}

// EstimateFLOPs returns an analytical FLOPs estimate for one configured batch.
func EstimateFLOPs(cfg *ArchConfig) FLOPsEstimate {
	if cfg == nil || cfg.ModelDim <= 0 || cfg.VocabSize <= 0 || cfg.SeqLen <= 0 || cfg.Training.BatchTokens <= 0 {
		return FLOPsEstimate{}
	}
	paramCount, expandedParamCount, err := ParameterCountsFromConfig(cfg)
	if err != nil {
		return FLOPsEstimate{}
	}
	if len(cfg.RecurrencePhases) > 0 {
		best := FLOPsEstimate{}
		for i, phase := range cfg.RecurrencePhases {
			est := estimateFLOPsForOrder(cfg, phase.Order, paramCount, expandedParamCount)
			if i == 0 || est.ForwardFLOPs > best.ForwardFLOPs {
				best = est
			}
		}
		return best
	}

	return estimateFLOPsForOrder(cfg, nil, paramCount, expandedParamCount)
}

// MaxCostRecurrencePhaseIndex returns the recurrence phase with the highest
// estimated forward FLOPs. It returns -1 when recurrence_phases is absent.
func MaxCostRecurrencePhaseIndex(cfg *ArchConfig) int {
	if cfg == nil || len(cfg.RecurrencePhases) == 0 {
		return -1
	}
	bestIdx := 0
	bestForward := int64(-1)
	paramCount, expandedParamCount, err := ParameterCountsFromConfig(cfg)
	if err != nil {
		return bestIdx
	}
	for i, phase := range cfg.RecurrencePhases {
		est := estimateFLOPsForOrder(cfg, phase.Order, paramCount, expandedParamCount)
		if est.ForwardFLOPs > bestForward {
			bestForward = est.ForwardFLOPs
			bestIdx = i
		}
	}
	return bestIdx
}

func estimateFLOPsForOrder(cfg *ArchConfig, order []int, paramCount, expandedParamCount int64) FLOPsEstimate {
	B := cfg.Training.BatchTokens / cfg.SeqLen
	if B <= 0 {
		return FLOPsEstimate{}
	}
	T := cfg.SeqLen
	D := cfg.ModelDim
	V := cfg.VocabSize
	ffn := ffnDim(D, cfg.MLPMult)

	forward := int64(0)
	if order == nil {
		for _, block := range cfg.Blocks {
			forward += estimateBlockFLOPs(block, B, T, D, V, ffn, cfg.MLPMult, cfg.BlockScales, cfg.ResidMix)
		}
	} else {
		for _, idx := range order {
			if idx < 0 || idx >= len(cfg.Blocks) {
				return FLOPsEstimate{}
			}
			forward += estimateBlockFLOPs(cfg.Blocks[idx], B, T, D, V, ffn, cfg.MLPMult, cfg.BlockScales, cfg.ResidMix)
		}
	}

	// Embedding lookup is indexing only; LM head projection produces logits.
	if cfg.CharVocabSize > 0 {
		charDim := cfg.EffectiveCharDim()
		charSlots := cfg.EffectiveCharMaxPerToken()
		forward += i64(B) * i64(T) * i64(charSlots) * i64(charDim)
		if charDim != D {
			forward += 2 * i64(B) * i64(T) * i64(charDim) * i64(D)
		}
		forward += 2 * i64(B) * i64(T) * i64(D) // learned scale + residual add
	}
	forward += 2 * i64(B) * i64(T) * i64(D) * i64(V)

	training := 3 * forward
	perToken := int64(0)
	if cfg.Training.BatchTokens > 0 {
		perToken = training / int64(cfg.Training.BatchTokens)
	}
	return FLOPsEstimate{
		ForwardFLOPs:       forward,
		TrainingFLOPs:      training,
		FLOPsPerToken:      perToken,
		ParamCount:         paramCount,
		ExpandedParamCount: expandedParamCount,
	}
}

func ParameterCountsFromConfig(cfg *ArchConfig) (int64, int64, error) {
	if cfg == nil {
		return 0, 0, fmt.Errorf("nil config")
	}

	uniqueRefs, err := normalizeWeightRefs(cfg.Blocks, cfg.Recurrence)
	if err != nil {
		return 0, 0, err
	}
	uniqueShapes, err := collectWeightShapesWithRefsHeadLayoutFeatures(
		cfg.ModelDim,
		cfg.VocabSize,
		cfg.SeqLen,
		cfg.EffectiveMLPMult(),
		cfg.ReservesUntiedHeadWeight(),
		cfg.BlockScales,
		cfg.ResidMix,
		cfg.UNet,
		cfg.ParallelResidual,
		cfg.CharVocabSize,
		cfg.EffectiveCharDim(),
		cfg.BigramVocabSize,
		cfg.EffectiveBigramDim(),
		cfg.TrigramVocabSize,
		cfg.EffectiveTrigramDim(),
		cfg.Blocks,
		uniqueRefs,
	)
	if err != nil {
		return 0, 0, err
	}

	expandedShapes, err := collectWeightShapesWithRefsHeadLayoutFeatures(
		cfg.ModelDim,
		cfg.VocabSize,
		cfg.SeqLen,
		cfg.EffectiveMLPMult(),
		cfg.ReservesUntiedHeadWeight(),
		cfg.BlockScales,
		cfg.ResidMix,
		cfg.UNet,
		cfg.ParallelResidual,
		cfg.CharVocabSize,
		cfg.EffectiveCharDim(),
		cfg.BigramVocabSize,
		cfg.EffectiveBigramDim(),
		cfg.TrigramVocabSize,
		cfg.EffectiveTrigramDim(),
		cfg.Blocks,
		identityWeightRefs(cfg.Blocks),
	)
	if err != nil {
		return 0, 0, err
	}

	return countWeightMetaElements(uniqueShapes), countWeightMetaElements(expandedShapes), nil
}

// ActiveParameterCountFromConfig returns the per-token active parameter count
// for sparse MoE models. Dense models return hasMoE=false.
func ActiveParameterCountFromConfig(cfg *ArchConfig) (active int64, hasMoE bool, err error) {
	if cfg == nil {
		return 0, false, fmt.Errorf("nil config")
	}
	uniqueParams, _, err := ParameterCountsFromConfig(cfg)
	if err != nil {
		return 0, false, err
	}
	refs, err := normalizeWeightRefs(cfg.Blocks, cfg.Recurrence)
	if err != nil {
		return 0, false, err
	}
	plan, err := newParallelResidualPlan(cfg.Blocks, cfg.ParallelResidual)
	if err != nil {
		return 0, false, err
	}
	if plan.any {
		if err := validateParallelResidualRefs(plan, refs); err != nil {
			return 0, false, err
		}
	}

	var moeTotal int64
	var moeActive int64
	for i, block := range cfg.Blocks {
		if refs[i] != i || blockTypeKey(block) != "moe" {
			continue
		}
		hasMoE = true
		metas, err := parallelBlockWeightShapes(block, plan.secondAt(i), cfg.ModelDim, cfg.SeqLen, cfg.Training.BatchTokens/cfg.SeqLen, cfg.VocabSize, cfg.EffectiveMLPMult(), cfg.BlockScales, cfg.ResidMix)
		if err != nil {
			return 0, false, err
		}
		moeTotal += countWeightMetaElements(metas)
		var nonExpert int64
		var firstExpert int64
		for _, meta := range metas {
			n := countWeightMetaElements([]WeightMeta{meta})
			if strings.HasPrefix(meta.Name, "expert_0_") {
				firstExpert += n
			} else if !strings.HasPrefix(meta.Name, "expert_") {
				nonExpert += n
			}
		}
		moeActive += nonExpert + int64(effectiveMoETopK(block))*firstExpert
	}
	if !hasMoE {
		return 0, false, nil
	}
	return uniqueParams - moeTotal + moeActive, true, nil
}

func countWeightMetaElements(metas []WeightMeta) int64 {
	total := int64(0)
	for _, meta := range metas {
		elements := int64(1)
		for _, dim := range meta.Shape {
			elements *= int64(dim)
		}
		total += elements
	}
	return total
}

func estimateBlockFLOPs(block BlockSpec, B, T, D, V, ffn int, mlpMult float64, blockScales, residMix bool) int64 {
	switch strings.ToLower(strings.TrimSpace(block.Type)) {
	case "plain":
		return estimatePlainBlockFLOPs(block, B, T, D, ffn)
	case "swiglu", "geglu":
		return estimateSwiGLUBlockFLOPs(B, T, D, ffn)
	case "moe":
		return estimateMoEBlockFLOPs(block, B, T, D, mlpMult)
	case "mamba":
		inner := block.InnerDim
		if inner <= 0 {
			inner = D
		}
		return 2 * i64(B) * i64(T) * i64(D) * i64(inner) * 4
	case "hgrn2":
		return estimateHGRN2BlockFLOPs(block, B, T, D)
	case "mlstm":
		return estimateMLSTMBlockFLOPs(block, B, T, D)
	default:
		return estimateWeightShapeFLOPs(block, B, T, D, V, mlpMult, blockScales, residMix)
	}
}

func estimatePlainBlockFLOPs(block BlockSpec, B, T, D, ffn int) int64 {
	total := int64(0)
	heads := block.Heads
	if heads <= 0 {
		heads = 4
	}
	if heads <= 0 || D%heads != 0 {
		return total
	}
	headDim := D / heads
	kvHeads := block.KVHeads
	if kvHeads <= 0 {
		kvHeads = heads
	}

	if !block.SkipAttention {
		attnWindow := T
		if block.WindowSize > 0 && block.WindowSize < T {
			attnWindow = block.WindowSize
		}
		switch {
		case block.KVSource > 0:
			total += 2 * i64(B) * i64(T) * i64(D) * i64(D) // Q + O only
		case kvHeads < heads:
			total += 2 * i64(B) * i64(T) * i64(D) * (i64(D) + 2*i64(D)*i64(kvHeads)/i64(heads))
		default:
			total += 3 * 2 * i64(B) * i64(T) * i64(D) * i64(D)
		}
		if block.QKNorm {
			qkNormTensors := int64(1)
			if block.KVSource <= 0 {
				qkNormTensors = 2
			}
			total += 5 * qkNormTensors * i64(B) * i64(heads) * i64(T) * i64(headDim)
		}
		total += 2 * i64(B) * i64(heads) * i64(T) * i64(attnWindow) * i64(headDim)
		total += 2 * i64(B) * i64(heads) * i64(T) * i64(attnWindow) * i64(headDim)
		if relativeAttentionEnabled(block) {
			relWindow := effectiveRelativeAttentionWindow(block)
			relRows := 2*relWindow - 1
			total += 2 * 2 * i64(relRows) * i64(D) * i64(D) // relative key/query projections
			total += 2 * 2 * i64(B) * i64(heads) * i64(T) * i64(relRows) * i64(headDim)
		}
		total += 2 * i64(B) * i64(T) * i64(D) * i64(D)
	}

	total += 2 * 2 * i64(B) * i64(T) * i64(D) * i64(ffn)
	if plainFFNActivationUsesGate(block.FFNActivation) {
		total += 2 * i64(B) * i64(T) * i64(D) * i64(ffn) // gate projection
		total += i64(B) * i64(T) * i64(ffn)              // gated elementwise multiply
	}
	return total
}

func estimateSwiGLUBlockFLOPs(B, T, D, ffn int) int64 {
	gateUp := 2 * 2 * i64(B) * i64(T) * i64(D) * i64(ffn)
	down := 2 * i64(B) * i64(T) * i64(ffn) * i64(D)
	return gateUp + down
}

func estimateWeightShapeFLOPs(block BlockSpec, B, T, D, V int, mlpMult float64, blockScales, residMix bool) int64 {
	metas, err := blockWeightShapes(block, D, T, B, V, mlpMult, blockScales, residMix)
	if err != nil {
		return 0
	}
	total := int64(0)
	for _, meta := range metas {
		if len(meta.Shape) != 2 {
			continue
		}
		elements := int64(1)
		for _, dim := range meta.Shape {
			elements *= int64(dim)
		}
		total += 2 * i64(B) * i64(T) * elements
	}
	return total
}

func i64(v int) int64 {
	return int64(v)
}
