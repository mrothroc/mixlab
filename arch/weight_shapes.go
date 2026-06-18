package arch

import (
	"fmt"
	"math"
	"strings"
)

// DefaultFFNMultiplier is the default FFN hidden dimension as a multiple of
// model dim. Used by plain attention, GLU, and MLP blocks when no override is set.
const DefaultFFNMultiplier = 2.67

// WeightMeta describes a single weight tensor's shape and initialization hint.
type WeightMeta struct {
	Name          string
	Shape         []int
	IsNormScale   bool // true for normalization scales (init 1.0); false for other 1-D params (init 0.0)
	InitOne       bool // true for non-norm weights that should initialize to 1.0
	InitValue     float32
	InitZero      bool
	InitMode      string
	InitLogArange bool
	InitDtBias    bool
	DtMin         float64
	DtMax         float64
	GPTBERTScale  float32
}

// ffnDim computes the FFN hidden dimension, clamped to at least D.
func ffnDim(D int, multiplier float64) int {
	if multiplier <= 0 {
		multiplier = DefaultFFNMultiplier
	}
	f := int(math.Round(float64(D) * multiplier))
	if f < D {
		return D
	}
	return f
}

func defaultMamba3CanonicalRank(inner int) int {
	r := inner / 16
	if r < 1 {
		return 1
	}
	return r
}

// blockWeightShapes returns the WeightMeta for a single block given model
// dimensions. This is the single source of truth for weight names, shapes, and
// initialization hints. The emitXxxIR functions consume weights in the same
// order returned here.
func blockWeightShapes(spec BlockSpec, D, T, B, V int, mlpMult float64, blockScales, residMix bool) ([]WeightMeta, error) {
	reg, err := lookupBlock(spec)
	if err != nil {
		return nil, err
	}
	if reg.weightShapesWithOptions != nil {
		return reg.weightShapesWithOptions(spec, D, T, B, V, EmitOptions{
			MLPMult:     mlpMult,
			BlockScales: blockScales,
			ResidMix:    residMix,
		})
	}
	if reg.WeightShapes == nil {
		return nil, fmt.Errorf("block type %q has no weight shaper", spec.Type)
	}
	return reg.WeightShapes(spec, D, T, B, V)
}

func blockWeightShapesWithEmitOptions(spec BlockSpec, D, T, B, V int, opts EmitOptions) ([]WeightMeta, error) {
	reg, err := lookupBlock(spec)
	if err != nil {
		return nil, err
	}
	if reg.weightShapesWithOptions != nil {
		return reg.weightShapesWithOptions(spec, D, T, B, V, opts)
	}
	if reg.WeightShapes == nil {
		return nil, fmt.Errorf("block type %q has no weight shaper", spec.Type)
	}
	return reg.WeightShapes(spec, D, T, B, V)
}

func builtinBlockWeightShapes(spec BlockSpec, D, T, B, V int, mlpMult float64, blockScales, residMix bool) ([]WeightMeta, error) {
	return builtinBlockWeightShapesWithOptions(spec, D, T, B, V, EmitOptions{
		MLPMult:       mlpMult,
		BlockScales:   blockScales,
		ResidMix:      residMix,
		Norm:          defaultNormSpec(),
		NormPlacement: NormPlacementPre,
	})
}

func builtinBlockWeightShapesWithOptions(spec BlockSpec, D, T, B, V int, opts EmitOptions) ([]WeightMeta, error) {
	mlpMult := opts.MLPMult
	blockScales := opts.BlockScales
	residMix := opts.ResidMix
	norm := normSpecOrDefault(opts.Norm)
	placement := normPlacementOrDefault(opts.NormPlacement)
	ffnInternalNorm := opts.FFNInternalNorm
	switch blockTypeName(spec.Type) {
	case "plain":
		heads := spec.Heads
		if heads <= 0 {
			heads = 4
		}
		kvHeads, err := normalizePlainKVHeads(heads, spec.KVHeads)
		if err != nil {
			return nil, err
		}
		if D%heads != 0 {
			return nil, fmt.Errorf("invalid attention dimensions D=%d H=%d", D, heads)
		}
		kvProjDim := kvHeads * (D / heads)
		ffn := ffnDim(D, mlpMult)
		var metas []WeightMeta
		if placement == NormPlacementPre || placement == NormPlacementSandwich {
			metas = append(metas, normWeights("norm", D, norm)...)
		}
		metas = append(metas, WeightMeta{Name: "wq", Shape: []int{D, D}})
		if spec.AttnBias {
			metas = append(metas, WeightMeta{Name: "wq_bias", Shape: []int{D}, InitZero: true})
		}
		if spec.KVSource <= 0 {
			valueProjDim := kvProjDim
			if spec.AttnValueGate {
				valueProjDim += D
			}
			metas = append(metas,
				WeightMeta{Name: "wk", Shape: []int{D, kvProjDim}},
			)
			if spec.AttnBias {
				metas = append(metas, WeightMeta{Name: "wk_bias", Shape: []int{kvProjDim}, InitZero: true})
			}
			metas = append(metas, WeightMeta{Name: "wv", Shape: []int{D, valueProjDim}})
			if spec.AttnBias {
				metas = append(metas, WeightMeta{Name: "wv_bias", Shape: []int{valueProjDim}, InitZero: true})
			}
		}
		if spec.QKNorm {
			headDim := D / heads
			metas = append(metas, WeightMeta{Name: "q_norm_scale", Shape: []int{headDim}, IsNormScale: true, InitOne: true})
			if spec.KVSource <= 0 {
				metas = append(metas, WeightMeta{Name: "k_norm_scale", Shape: []int{headDim}, IsNormScale: true, InitOne: true})
			}
		}
		if relativeAttentionUsesPerBlockProjections(spec) {
			relWindow := effectiveRelativeAttentionWindow(spec)
			metas = append(metas,
				WeightMeta{Name: "relative_embeddings", Shape: []int{2*relWindow - 1, D}},
				WeightMeta{Name: "w_pos_key", Shape: []int{D, D}},
				WeightMeta{Name: "w_pos_query", Shape: []int{D, D}},
			)
		}
		if spec.QKGain > 0 {
			metas = append(metas, WeightMeta{Name: "qk_gain", Shape: []int{heads}, InitValue: float32(spec.QKGain)})
		}
		if spec.SparseAttnGate {
			gateDim := plainSparseAttnGateWidth(D)
			metas = append(metas, WeightMeta{Name: "attn_gate_w", Shape: []int{heads, gateDim}, InitZero: true})
		}
		metas = append(metas, WeightMeta{Name: "wo", Shape: []int{D, D}})
		if spec.AttnBias {
			metas = append(metas, WeightMeta{Name: "wo_bias", Shape: []int{D}, InitZero: true})
		}
		if placement == NormPlacementPost || placement == NormPlacementSandwich {
			metas = append(metas, normWeights("post_attn_norm", D, norm)...)
		}
		if blockScales {
			metas = append(metas, WeightMeta{Name: "attn_scale", Shape: []int{D}, InitOne: true})
		}
		if placement == NormPlacementSandwich {
			metas = append(metas, normWeights("ffn_norm", D, norm)...)
		}
		if plainFFNActivationUsesGate(spec.FFNActivation) {
			metas = append(metas, WeightMeta{Name: "ff_gate", Shape: []int{D, ffn}})
		}
		metas = append(metas, WeightMeta{Name: "ff1", Shape: []int{D, ffn}})
		if ffnInternalNorm {
			metas = append(metas, normWeights("ffn_internal_norm", ffn, norm)...)
		}
		metas = append(metas, WeightMeta{Name: "ff2", Shape: []int{ffn, D}})
		if placement == NormPlacementPost || placement == NormPlacementSandwich {
			metas = append(metas, normWeights("post_ffn_norm", D, norm)...)
		}
		if blockScales {
			metas = append(metas, WeightMeta{Name: "mlp_scale", Shape: []int{D}, InitOne: true})
		}
		if residMix {
			metas = append([]WeightMeta{{Name: "resid_mix", Shape: []int{2, D}}}, metas...)
		}
		return metas, nil

	case "swiglu", "geglu":
		ffn := ffnDim(D, mlpMult)
		var metas []WeightMeta
		if placement == NormPlacementPre || placement == NormPlacementSandwich {
			metas = append(metas, normWeights("ffn_norm", D, norm)...)
		}
		metas = append(metas,
			WeightMeta{Name: "w_gate", Shape: []int{D, ffn}},
			WeightMeta{Name: "w_up", Shape: []int{D, ffn}},
		)
		if ffnInternalNorm {
			metas = append(metas, normWeights("ffn_internal_norm", ffn, norm)...)
		}
		metas = append(metas, WeightMeta{Name: "w_down", Shape: []int{ffn, D}})
		if placement == NormPlacementPost || placement == NormPlacementSandwich {
			metas = append(metas, normWeights("post_ffn_norm", D, norm)...)
		}
		if blockScales {
			metas = append(metas, WeightMeta{Name: "mlp_scale", Shape: []int{D}, InitOne: true})
		}
		return metas, nil

	case "mlp":
		ffn := ffnDim(D, mlpMult)
		var metas []WeightMeta
		if placement == NormPlacementPre || placement == NormPlacementSandwich {
			metas = append(metas, normWeights("ffn_norm", D, norm)...)
		}
		metas = append(metas, WeightMeta{Name: "w_up", Shape: []int{D, ffn}})
		if ffnInternalNorm {
			metas = append(metas, normWeights("ffn_internal_norm", ffn, norm)...)
		}
		metas = append(metas, WeightMeta{Name: "w_down", Shape: []int{ffn, D}})
		if placement == NormPlacementPost || placement == NormPlacementSandwich {
			metas = append(metas, normWeights("post_ffn_norm", D, norm)...)
		}
		return metas, nil

	case "perceiver", "bottleneck":
		L := spec.NumLatents
		if L <= 0 {
			if spec.Type == "bottleneck" {
				L = 4
			} else {
				L = 32
			}
		}
		return []WeightMeta{
			{Name: "latent_init", Shape: []int{L, D}},
			{Name: "wq_cross", Shape: []int{D, D}},
			{Name: "wk_cross", Shape: []int{D, D}},
			{Name: "wv_cross", Shape: []int{D, D}},
			{Name: "wo_cross", Shape: []int{D, D}},
			{Name: "wq_self", Shape: []int{D, D}},
			{Name: "wk_self", Shape: []int{D, D}},
			{Name: "wv_self", Shape: []int{D, D}},
			{Name: "wo_self", Shape: []int{D, D}},
			{Name: "wq_broad", Shape: []int{D, D}},
			{Name: "wk_broad", Shape: []int{D, D}},
			{Name: "wv_broad", Shape: []int{D, D}},
			{Name: "wo_broad", Shape: []int{D, D}},
			{Name: "ff1", Shape: []int{D, D * 2}},
			{Name: "ff2", Shape: []int{D * 2, D}},
		}, nil

	case "retnet":
		heads := spec.Heads
		if heads <= 0 {
			heads = 4
		}
		retFFN := D * 2
		return []WeightMeta{
			{Name: "norm_scale", Shape: []int{D}, IsNormScale: true, InitOne: true},
			{Name: "wq", Shape: []int{D, D}},
			{Name: "wk", Shape: []int{D, D}},
			{Name: "wv", Shape: []int{D, D}},
			{Name: "decay", Shape: []int{heads}},
			{Name: "wo", Shape: []int{D, D}},
			{Name: "ff1", Shape: []int{D, retFFN}},
			{Name: "ff2", Shape: []int{retFFN, D}},
		}, nil

	case "rwkv":
		return []WeightMeta{
			{Name: "mu", Shape: []int{D}},
			{Name: "wr", Shape: []int{D, D}},
			{Name: "wk", Shape: []int{D, D}},
			{Name: "wv", Shape: []int{D, D}},
			{Name: "w_decay", Shape: []int{D}},
			{Name: "wo", Shape: []int{D, D}},
			{Name: "mu2", Shape: []int{D}},
			{Name: "wr2", Shape: []int{D, D}},
			{Name: "wk2", Shape: []int{D, D}},
			{Name: "wv2", Shape: []int{D, D}},
		}, nil

	case "mamba":
		inner := spec.InnerDim
		if inner <= 0 {
			inner = D
		}
		return []WeightMeta{
			{Name: "in_proj", Shape: []int{D, 2 * inner}},
			{Name: "conv_w", Shape: []int{inner, inner}},
			{Name: "out_proj", Shape: []int{inner, D}},
			{Name: "scan_decay", Shape: []int{inner}},
		}, nil

	case "gated_linear_ssm":
		inner := spec.InnerDim
		if inner <= 0 {
			inner = D
		}
		return []WeightMeta{
			{Name: "norm_scale", Shape: []int{D}, IsNormScale: true, InitOne: true},
			{Name: "w_gate", Shape: []int{D, inner}},
			{Name: "w_ssm", Shape: []int{D, inner}},
			{Name: "w_dt", Shape: []int{D, inner}},
			{Name: "wo", Shape: []int{inner, D}},
			{Name: "scan_decay", Shape: []int{inner}},
		}, nil

	case "mamba3-canonical":
		inner := spec.InnerDim
		if inner <= 0 {
			inner = D
		}
		stateSize := spec.StateSize
		if stateSize <= 0 {
			stateSize = 16
		}
		if stateSize%2 != 0 {
			return nil, fmt.Errorf("mamba3-canonical state_size must be even for complex 2x2 state pairs, got %d", stateSize)
		}
		nGroups := spec.NGroups
		if nGroups <= 0 {
			nGroups = 4
		}
		if inner%nGroups != 0 {
			return nil, fmt.Errorf("mamba3-canonical inner_dim=%d must be divisible by n_groups=%d", inner, nGroups)
		}
		dtRank := spec.DTRank
		if dtRank <= 0 {
			dtRank = defaultMamba3CanonicalRank(inner)
		}
		convKernel := spec.ConvKernel
		if convKernel <= 0 {
			convKernel = 4
		}
		dtMin := spec.DTMin
		if dtMin <= 0 {
			dtMin = 0.001
		}
		dtMax := spec.DTMax
		if dtMax <= 0 {
			dtMax = 0.1
		}
		if dtMin <= 0 || dtMax <= dtMin {
			return nil, fmt.Errorf("mamba3-canonical requires 0 < dt_min < dt_max, got dt_min=%g dt_max=%g", dtMin, dtMax)
		}
		groupState := nGroups * stateSize
		metas := []WeightMeta{
			{Name: "pre_norm_scale", Shape: []int{D}, IsNormScale: true, InitOne: true},
			{Name: "w_x", Shape: []int{D, inner}},
		}
		useConv := spec.UseConv == nil || *spec.UseConv
		if useConv {
			metas = append(metas, WeightMeta{Name: "conv_w", Shape: []int{inner, convKernel}})
		}
		metas = append(metas,
			WeightMeta{Name: "w_dt_low", Shape: []int{inner, dtRank}},
			WeightMeta{Name: "w_dt_high", Shape: []int{dtRank, inner}},
			WeightMeta{Name: "w_lambda_low", Shape: []int{inner, dtRank}},
			WeightMeta{Name: "w_lambda_high", Shape: []int{dtRank, inner}},
			WeightMeta{Name: "w_theta_low", Shape: []int{inner, dtRank}},
			WeightMeta{Name: "w_theta_high", Shape: []int{dtRank, inner * (stateSize / 2)}},
			WeightMeta{Name: "w_B", Shape: []int{inner, groupState}},
			WeightMeta{Name: "w_C", Shape: []int{inner, groupState}},
			WeightMeta{Name: "B_norm_scale", Shape: []int{stateSize}, IsNormScale: true, InitOne: true},
			WeightMeta{Name: "C_norm_scale", Shape: []int{stateSize}, IsNormScale: true, InitOne: true},
			WeightMeta{Name: "B_bias", Shape: []int{groupState}, InitOne: true},
			WeightMeta{Name: "C_bias", Shape: []int{groupState}, InitOne: true},
			WeightMeta{Name: "A_log", Shape: []int{inner, stateSize}, InitLogArange: true},
			WeightMeta{Name: "dt_bias", Shape: []int{inner}, InitDtBias: true, DtMin: dtMin, DtMax: dtMax},
			WeightMeta{Name: "post_norm_scale", Shape: []int{inner}, IsNormScale: true, InitOne: true},
			WeightMeta{Name: "w_gate", Shape: []int{D, inner}},
			WeightMeta{Name: "w_out", Shape: []int{inner, D}},
		)
		return metas, nil

	case "gated_deltanet":
		return gatedDeltaNetWeightShapes(spec, D, T, B, V)

	case "cross_attention":
		heads := spec.Heads
		if heads <= 0 {
			heads = 4
		}
		_ = heads
		ffn := ffnDim(D, mlpMult)
		return []WeightMeta{
			{Name: "norm_scale", Shape: []int{D}, IsNormScale: true, InitOne: true},
			{Name: "wq", Shape: []int{D, D}},
			{Name: "wk", Shape: []int{D, D}},
			{Name: "wv", Shape: []int{D, D}},
			{Name: "wo", Shape: []int{D, D}},
			{Name: "ff1", Shape: []int{D, ffn}},
			{Name: "ff2", Shape: []int{ffn, D}},
		}, nil

	case "token_blend":
		return []WeightMeta{
			{Name: "w_gate", Shape: []int{D, D}},
		}, nil

	case "custom":
		heads := spec.Heads
		if heads <= 0 {
			heads = 1
		}
		metas := make([]WeightMeta, len(spec.Weights))
		for i, ws := range spec.Weights {
			resolved, err := resolveShapeSymbol(ws.Shape, D, heads, T, B, V)
			if err != nil {
				return nil, fmt.Errorf("custom block %q weight %q: %w", spec.Name, ws.Name, err)
			}
			metas[i] = WeightMeta{Name: ws.Name, Shape: resolved}
		}
		return metas, nil

	default:
		return nil, fmt.Errorf("unsupported block type %q", spec.Type)
	}
}

func parallelBlockWeightShapes(spec BlockSpec, pairedSecond bool, D, T, B, V int, mlpMult float64, blockScales, residMix bool) ([]WeightMeta, error) {
	metas, err := blockWeightShapes(spec, D, T, B, V, mlpMult, blockScales, residMix)
	if err != nil {
		return nil, err
	}
	if pairedSecond && isParallelResidualFFNSecond(spec) {
		if len(metas) == 0 || (metas[0].Name != "ffn_norm_scale" && metas[0].Name != "moe_norm_scale") {
			return nil, fmt.Errorf("parallel_residual FFN block has unexpected weight layout")
		}
		metas = metas[1:]
	}
	return metas, nil
}

func streamWeightShapesWithRefsAndParallelOptions(specs []BlockSpec, refs []int, D, T, B, V int, opts EmitOptions, parallelResidual bool) ([]WeightMeta, error) {
	plan, err := newParallelResidualPlan(specs, parallelResidual)
	if err != nil {
		return nil, err
	}
	if !plan.any {
		var all []WeightMeta
		for i, spec := range specs {
			if refs[i] != i {
				continue
			}
			metas, err := blockWeightShapesWithEmitOptions(spec, D, T, B, V, opts)
			if err != nil {
				return nil, err
			}
			annotateGPTBERTOutputScale(metas, spec, i)
			all = append(all, metas...)
		}
		return all, nil
	}
	if err := validateParallelResidualRefs(plan, refs); err != nil {
		return nil, err
	}
	var all []WeightMeta
	for i, spec := range specs {
		if refs[i] != i {
			continue
		}
		metas, err := parallelBlockWeightShapes(spec, plan.secondAt(i), D, T, B, V, opts.MLPMult, opts.BlockScales, opts.ResidMix)
		if err != nil {
			return nil, err
		}
		annotateGPTBERTOutputScale(metas, spec, i)
		all = append(all, metas...)
	}
	return all, nil
}

func blockRangeWeightShapesWithRefs(specs []BlockSpec, refs []int, start, end, D, T, B, V int, mlpMult float64, blockScales, residMix bool) ([]WeightMeta, error) {
	var all []WeightMeta
	for i := start; i < end; i++ {
		if refs[i] != i {
			continue
		}
		metas, err := blockWeightShapes(specs[i], D, T, B, V, mlpMult, blockScales, residMix)
		if err != nil {
			return nil, err
		}
		annotateGPTBERTOutputScale(metas, specs[i], i)
		all = append(all, metas...)
	}
	return all, nil
}

func annotateGPTBERTOutputScale(metas []WeightMeta, spec BlockSpec, blockIndex int) {
	if blockIndex < 0 {
		blockIndex = 0
	}
	scale := float32(math.Sqrt(1.0 / (2.0 * float64(blockIndex+1))))
	for i := range metas {
		if isGPTBERTOutputProjectionName(spec, metas[i].Name) {
			metas[i].GPTBERTScale = scale
		}
	}
}

func isGPTBERTOutputProjectionName(spec BlockSpec, name string) bool {
	switch blockTypeKey(spec) {
	case "plain":
		return name == "wo" || name == "ff2"
	case "swiglu", "geglu", "mlp":
		return name == "w_down"
	case "moe":
		return strings.HasSuffix(name, "_w_down")
	default:
		return false
	}
}

func bigramWeightShapes(modelDim, bigramVocabSize, bigramDim int) []WeightMeta {
	if bigramVocabSize <= 0 {
		return nil
	}
	D := modelDim
	if bigramDim <= 0 {
		bigramDim = D
	}
	shapes := []WeightMeta{
		{Name: "bigram_table", Shape: []int{bigramVocabSize, bigramDim}},
	}
	if bigramDim != D {
		shapes = append(shapes, WeightMeta{Name: "bigram_proj", Shape: []int{bigramDim, D}})
	}
	shapes = append(shapes, WeightMeta{Name: "bigram_scale", Shape: []int{1}, InitOne: true})
	return shapes
}

func trigramWeightShapes(modelDim, trigramVocabSize, trigramDim int) []WeightMeta {
	if trigramVocabSize <= 0 {
		return nil
	}
	D := modelDim
	if trigramDim <= 0 {
		trigramDim = D
	}
	shapes := []WeightMeta{
		{Name: "trigram_table", Shape: []int{trigramVocabSize, trigramDim}},
	}
	if trigramDim != D {
		shapes = append(shapes, WeightMeta{Name: "trigram_proj", Shape: []int{trigramDim, D}})
	}
	shapes = append(shapes, WeightMeta{Name: "trigram_scale", Shape: []int{1}, InitOne: true})
	return shapes
}

func charWeightShapes(modelDim, charVocabSize, charDim int) []WeightMeta {
	if charVocabSize <= 0 {
		return nil
	}
	D := modelDim
	if charDim <= 0 {
		charDim = D
	}
	shapes := []WeightMeta{
		{Name: "char_table", Shape: []int{charVocabSize, charDim}},
	}
	if charDim != D {
		shapes = append(shapes, WeightMeta{Name: "char_proj", Shape: []int{charDim, D}})
	}
	shapes = append(shapes, WeightMeta{Name: "char_scale", Shape: []int{1}, InitOne: true})
	return shapes
}

// CollectWeightShapes returns the complete ordered list of WeightMeta for an
// architecture configuration. The order matches the weight indices used by
// BuildIRProgram: embed, head, final_norm, then block weights in emission order.
//
// This is the single source of truth for weight shapes. Both IR building and
// weight initialization should derive from this function.
func CollectWeightShapes(
	modelDim, vocabSize, seqLen int,
	mlpMult float64,
	tieEmbeddings bool,
	blockScales, residMix bool,
	unet bool,
	blocks []BlockSpec,
) ([]WeightMeta, error) {
	return CollectWeightShapesWithBigram(modelDim, vocabSize, seqLen, mlpMult, tieEmbeddings, blockScales, residMix, unet, 0, 0, blocks)
}

// CollectWeightShapesWithBigram returns ordered weight metadata including
// optional model-level bigram embedding weights.
func CollectWeightShapesWithBigram(
	modelDim, vocabSize, seqLen int,
	mlpMult float64,
	tieEmbeddings bool,
	blockScales, residMix bool,
	unet bool,
	bigramVocabSize, bigramDim int,
	blocks []BlockSpec,
) ([]WeightMeta, error) {
	return CollectWeightShapesWithBigramAndRecurrence(modelDim, vocabSize, seqLen, mlpMult, tieEmbeddings, blockScales, residMix, unet, bigramVocabSize, bigramDim, blocks, nil)
}

// CollectWeightShapesWithBigramAndRecurrence returns ordered weight metadata
// including optional bigram weights and only original sequential block weights.
func CollectWeightShapesWithBigramAndRecurrence(
	modelDim, vocabSize, seqLen int,
	mlpMult float64,
	tieEmbeddings bool,
	blockScales, residMix bool,
	unet bool,
	bigramVocabSize, bigramDim int,
	blocks []BlockSpec,
	recurrence []int,
) ([]WeightMeta, error) {
	return CollectWeightShapesWithBigramRecurrenceAndParallel(modelDim, vocabSize, seqLen, mlpMult, tieEmbeddings, blockScales, residMix, unet, false, bigramVocabSize, bigramDim, blocks, recurrence)
}

// CollectWeightShapesWithBigramRecurrenceAndParallel returns ordered weight
// metadata including optional bigram weights, original sequential block
// weights, and parallel residual norm sharing.
func CollectWeightShapesWithBigramRecurrenceAndParallel(
	modelDim, vocabSize, seqLen int,
	mlpMult float64,
	tieEmbeddings bool,
	blockScales, residMix bool,
	unet bool,
	parallelResidual bool,
	bigramVocabSize, bigramDim int,
	blocks []BlockSpec,
	recurrence []int,
) ([]WeightMeta, error) {
	refs, err := normalizeWeightRefs(blocks, recurrence)
	if err != nil {
		return nil, fmt.Errorf("blocks: %w", err)
	}
	return collectWeightShapesWithRefs(modelDim, vocabSize, seqLen, mlpMult, tieEmbeddings, blockScales, residMix, unet, parallelResidual, bigramVocabSize, bigramDim, 0, 0, blocks, refs)
}

// CollectWeightShapesWithNgramsRecurrenceAndParallel returns ordered weight
// metadata including optional bigram and trigram weights, original sequential
// block weights, and parallel residual norm sharing.
func CollectWeightShapesWithNgramsRecurrenceAndParallel(
	modelDim, vocabSize, seqLen int,
	mlpMult float64,
	tieEmbeddings bool,
	blockScales, residMix bool,
	unet bool,
	parallelResidual bool,
	bigramVocabSize, bigramDim int,
	trigramVocabSize, trigramDim int,
	blocks []BlockSpec,
	recurrence []int,
) ([]WeightMeta, error) {
	refs, err := normalizeWeightRefs(blocks, recurrence)
	if err != nil {
		return nil, fmt.Errorf("blocks: %w", err)
	}
	return collectWeightShapesWithRefs(modelDim, vocabSize, seqLen, mlpMult, tieEmbeddings, blockScales, residMix, unet, parallelResidual, bigramVocabSize, bigramDim, trigramVocabSize, trigramDim, blocks, refs)
}

// CollectWeightShapesFromConfig returns ordered weight metadata for the full
// config, including MTP untie schedules that reserve a future LM head weight.
func CollectWeightShapesFromConfig(cfg *ArchConfig) ([]WeightMeta, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
	}
	refs, err := normalizeWeightRefs(cfg.Blocks, cfg.Recurrence)
	if err != nil {
		return nil, fmt.Errorf("blocks: %w", err)
	}
	metas, err := collectWeightShapesWithRefsHeadLayoutFeaturesNorm(
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
		refs,
		cfg.EffectiveNormSpec(),
		cfg.EffectiveNormPlacement(),
		cfg.FFNInternalNorm,
	)
	if err != nil {
		return nil, err
	}
	smearMetas, err := smearEmbeddingWeightShapes(cfg.ModelDim, cfg.SeqLen, cfg.smearEmbeddingOptions())
	if err != nil {
		return nil, err
	}
	backoutMetas := backoutWeightShapes(cfg.Backout)
	data2VecMetas := data2VecWeightShapes(cfg.ModelDim, cfg.Training.Data2Vec)
	if len(smearMetas) == 0 && len(backoutMetas) == 0 && len(data2VecMetas) == 0 {
		return metas, nil
	}
	fixed := fixedWeightCountWithHeadAndNorm(cfg.ReservesUntiedHeadWeight(), cfg.EffectiveNormSpec())
	out := make([]WeightMeta, 0, len(metas)+len(smearMetas)+len(backoutMetas)+len(data2VecMetas))
	out = append(out, metas[:fixed]...)
	out = append(out, smearMetas...)
	out = append(out, metas[fixed:]...)
	out = append(out, backoutMetas...)
	out = append(out, data2VecMetas...)
	return out, nil
}

func collectWeightShapesWithRefs(
	modelDim, vocabSize, seqLen int,
	mlpMult float64,
	tieEmbeddings bool,
	blockScales, residMix bool,
	unet bool,
	parallelResidual bool,
	bigramVocabSize, bigramDim int,
	trigramVocabSize, trigramDim int,
	blocks []BlockSpec,
	refs []int,
) ([]WeightMeta, error) {
	return collectWeightShapesWithRefsHeadLayout(modelDim, vocabSize, seqLen, mlpMult, !tieEmbeddings, blockScales, residMix, unet, parallelResidual, bigramVocabSize, bigramDim, trigramVocabSize, trigramDim, blocks, refs)
}

func collectWeightShapesWithRefsHeadLayout(
	modelDim, vocabSize, seqLen int,
	mlpMult float64,
	reserveHead bool,
	blockScales, residMix bool,
	unet bool,
	parallelResidual bool,
	bigramVocabSize, bigramDim int,
	trigramVocabSize, trigramDim int,
	blocks []BlockSpec,
	refs []int,
) ([]WeightMeta, error) {
	return collectWeightShapesWithRefsHeadLayoutFeatures(modelDim, vocabSize, seqLen, mlpMult, reserveHead, blockScales, residMix, unet, parallelResidual, 0, 0, bigramVocabSize, bigramDim, trigramVocabSize, trigramDim, blocks, refs)
}

func collectWeightShapesWithRefsHeadLayoutFeatures(
	modelDim, vocabSize, seqLen int,
	mlpMult float64,
	reserveHead bool,
	blockScales, residMix bool,
	unet bool,
	parallelResidual bool,
	charVocabSize, charDim int,
	bigramVocabSize, bigramDim int,
	trigramVocabSize, trigramDim int,
	blocks []BlockSpec,
	refs []int,
) ([]WeightMeta, error) {
	return collectWeightShapesWithRefsHeadLayoutFeaturesNorm(modelDim, vocabSize, seqLen, mlpMult, reserveHead, blockScales, residMix, unet, parallelResidual, charVocabSize, charDim, bigramVocabSize, bigramDim, trigramVocabSize, trigramDim, blocks, refs, defaultNormSpec(), NormPlacementPre, false)
}

func collectWeightShapesWithRefsHeadLayoutFeaturesNorm(
	modelDim, vocabSize, seqLen int,
	mlpMult float64,
	reserveHead bool,
	blockScales, residMix bool,
	unet bool,
	parallelResidual bool,
	charVocabSize, charDim int,
	bigramVocabSize, bigramDim int,
	trigramVocabSize, trigramDim int,
	blocks []BlockSpec,
	refs []int,
	norm NormSpec,
	normPlacement string,
	ffnInternalNorm bool,
) ([]WeightMeta, error) {
	D := modelDim
	V := vocabSize
	T := seqLen
	if D <= 0 {
		return nil, fmt.Errorf("invalid model_dim=%d", D)
	}
	if V <= 0 {
		return nil, fmt.Errorf("invalid vocab_size=%d", V)
	}
	if charVocabSize < 0 {
		return nil, fmt.Errorf("invalid char_vocab_size=%d", charVocabSize)
	}
	if charVocabSize > 0 && charVocabSize < 257 {
		return nil, fmt.Errorf("invalid char_vocab_size=%d", charVocabSize)
	}
	if charDim < 0 {
		return nil, fmt.Errorf("invalid char_dim=%d", charDim)
	}
	if bigramVocabSize < 0 {
		return nil, fmt.Errorf("invalid bigram_vocab_size=%d", bigramVocabSize)
	}
	if bigramVocabSize == 1 {
		return nil, fmt.Errorf("invalid bigram_vocab_size=%d", bigramVocabSize)
	}
	if bigramDim < 0 {
		return nil, fmt.Errorf("invalid bigram_dim=%d", bigramDim)
	}
	if trigramVocabSize < 0 {
		return nil, fmt.Errorf("invalid trigram_vocab_size=%d", trigramVocabSize)
	}
	if trigramVocabSize == 1 {
		return nil, fmt.Errorf("invalid trigram_vocab_size=%d", trigramVocabSize)
	}
	if trigramDim < 0 {
		return nil, fmt.Errorf("invalid trigram_dim=%d", trigramDim)
	}
	if len(refs) != len(blocks) {
		return nil, fmt.Errorf("invalid weight refs length=%d for blocks length=%d", len(refs), len(blocks))
	}
	norm = normSpecOrDefault(norm)
	normPlacement = normPlacementOrDefault(normPlacement)

	var shapes []WeightMeta

	// Fixed weights: embed + optional head + final norm.
	shapes = append(shapes, WeightMeta{Name: "embed", Shape: []int{V, D}})
	if reserveHead {
		shapes = append(shapes, WeightMeta{Name: "head", Shape: []int{D, V}})
	}
	shapes = append(shapes, normWeights("final_norm", D, norm)...)
	shapes = append(shapes, charWeightShapes(D, charVocabSize, charDim)...)
	shapes = append(shapes, bigramWeightShapes(D, bigramVocabSize, bigramDim)...)
	shapes = append(shapes, trigramWeightShapes(D, trigramVocabSize, trigramDim)...)
	sharedRel, err := sharedRelativeAttentionWeightShapes(D, blocks)
	if err != nil {
		return nil, err
	}
	shapes = append(shapes, sharedRel...)

	plan, err := newParallelResidualPlan(blocks, parallelResidual)
	if err != nil {
		return nil, err
	}
	if plan.any && unet {
		return nil, fmt.Errorf("parallel_residual is not supported with unet")
	}
	if err := validateParallelResidualRefs(plan, refs); err != nil {
		return nil, err
	}
	if unet {
		numEncoder, numSkip := unetLayout(len(blocks))
		enc, err := blockRangeWeightShapesWithRefs(blocks, refs, 0, numEncoder, D, T, 1, V, mlpMult, blockScales, residMix)
		if err != nil {
			return nil, fmt.Errorf("blocks: %w", err)
		}
		shapes = append(shapes, enc...)
		for i := 0; i < numSkip; i++ {
			shapes = append(shapes, WeightMeta{Name: fmt.Sprintf("skip_weight_%d", i), Shape: []int{D}, InitOne: true})
		}
		dec, err := blockRangeWeightShapesWithRefs(blocks, refs, numEncoder, len(blocks), D, T, 1, V, mlpMult, blockScales, residMix)
		if err != nil {
			return nil, fmt.Errorf("blocks: %w", err)
		}
		shapes = append(shapes, dec...)
	} else {
		blk, err := streamWeightShapesWithRefsAndParallelOptions(blocks, refs, D, T, 1, V, EmitOptions{
			MLPMult:         mlpMult,
			BlockScales:     blockScales,
			ResidMix:        residMix,
			Norm:            norm,
			NormPlacement:   normPlacement,
			FFNInternalNorm: ffnInternalNorm,
		}, parallelResidual)
		if err != nil {
			return nil, fmt.Errorf("blocks: %w", err)
		}
		shapes = append(shapes, blk...)
	}

	return shapes, nil
}
