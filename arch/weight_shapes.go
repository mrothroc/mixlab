package arch

import (
	"fmt"
	"math"
	"strings"
)

// DefaultFFNMultiplier is the default FFN hidden dimension as a multiple of
// model dim. Used by plain attention and swiglu blocks when no override is set.
const DefaultFFNMultiplier = 2.67

// WeightMeta describes a single weight tensor's shape and initialization hint.
type WeightMeta struct {
	Name        string
	Shape       []int
	IsNormScale bool // true for normalization scales (init 1.0); false for other 1-D params (init 0.0)
	InitOne     bool // true for non-norm weights that should initialize to 1.0
	InitValue   float32
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

func builtinBlockWeightShapes(spec BlockSpec, D, T, B, V int, mlpMult float64, blockScales, residMix bool) ([]WeightMeta, error) {
	switch strings.ToLower(strings.TrimSpace(spec.Type)) {
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
		metas := []WeightMeta{
			{Name: "norm_scale", Shape: []int{D}, IsNormScale: true, InitOne: true},
			{Name: "wq", Shape: []int{D, D}},
		}
		if spec.KVSource <= 0 {
			metas = append(metas,
				WeightMeta{Name: "wk", Shape: []int{D, kvProjDim}},
				WeightMeta{Name: "wv", Shape: []int{D, kvProjDim}},
			)
		}
		if spec.QKGain > 0 {
			metas = append(metas, WeightMeta{Name: "qk_gain", Shape: []int{heads}, InitValue: float32(spec.QKGain)})
		}
		metas = append(metas, WeightMeta{Name: "wo", Shape: []int{D, D}})
		if blockScales {
			metas = append(metas, WeightMeta{Name: "attn_scale", Shape: []int{D}, InitOne: true})
		}
		metas = append(metas,
			WeightMeta{Name: "ff1", Shape: []int{D, ffn}},
			WeightMeta{Name: "ff2", Shape: []int{ffn, D}},
		)
		if blockScales {
			metas = append(metas, WeightMeta{Name: "mlp_scale", Shape: []int{D}, InitOne: true})
		}
		if residMix {
			metas = append([]WeightMeta{{Name: "resid_mix", Shape: []int{2, D}}}, metas...)
		}
		return metas, nil

	case "swiglu":
		ffn := ffnDim(D, mlpMult)
		metas := []WeightMeta{
			{Name: "ffn_norm_scale", Shape: []int{D}, IsNormScale: true, InitOne: true},
			{Name: "w_gate", Shape: []int{D, ffn}},
			{Name: "w_up", Shape: []int{D, ffn}},
			{Name: "w_down", Shape: []int{ffn, D}},
		}
		if blockScales {
			metas = append(metas, WeightMeta{Name: "mlp_scale", Shape: []int{D}, InitOne: true})
		}
		return metas, nil

	case "mlp":
		ffn := ffnDim(D, mlpMult)
		return []WeightMeta{
			{Name: "ffn_norm_scale", Shape: []int{D}, IsNormScale: true, InitOne: true},
			{Name: "w_up", Shape: []int{D, ffn}},
			{Name: "w_down", Shape: []int{ffn, D}},
		}, nil

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

	case "mamba3":
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

func streamWeightShapesWithRefs(specs []BlockSpec, refs []int, D, T, B, V int, mlpMult float64, blockScales, residMix bool) ([]WeightMeta, error) {
	var all []WeightMeta
	for i, spec := range specs {
		if refs[i] != i {
			continue
		}
		metas, err := blockWeightShapes(spec, D, T, B, V, mlpMult, blockScales, residMix)
		if err != nil {
			return nil, err
		}
		all = append(all, metas...)
	}
	return all, nil
}

func parallelBlockWeightShapes(spec BlockSpec, blockIdx, D, T, B, V int, mlpMult float64, blockScales, residMix bool) ([]WeightMeta, error) {
	metas, err := blockWeightShapes(spec, D, T, B, V, mlpMult, blockScales, residMix)
	if err != nil {
		return nil, err
	}
	if blockIdx%2 == 1 && blockTypeKey(spec) == "swiglu" {
		if len(metas) == 0 || metas[0].Name != "ffn_norm_scale" {
			return nil, fmt.Errorf("parallel_residual swiglu block %d has unexpected weight layout", blockIdx)
		}
		metas = metas[1:]
	}
	return metas, nil
}

func streamWeightShapesWithRefsAndParallel(specs []BlockSpec, refs []int, D, T, B, V int, mlpMult float64, blockScales, residMix, parallelResidual bool) ([]WeightMeta, error) {
	if !parallelResidual {
		return streamWeightShapesWithRefs(specs, refs, D, T, B, V, mlpMult, blockScales, residMix)
	}
	if err := validateParallelResidualBlocks(specs); err != nil {
		return nil, err
	}
	var all []WeightMeta
	for i, spec := range specs {
		if refs[i] != i {
			continue
		}
		metas, err := parallelBlockWeightShapes(spec, i, D, T, B, V, mlpMult, blockScales, residMix)
		if err != nil {
			return nil, err
		}
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
		all = append(all, metas...)
	}
	return all, nil
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
	D := modelDim
	V := vocabSize
	T := seqLen
	if D <= 0 {
		return nil, fmt.Errorf("invalid model_dim=%d", D)
	}
	if V <= 0 {
		return nil, fmt.Errorf("invalid vocab_size=%d", V)
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

	var shapes []WeightMeta

	// Fixed weights: embed + optional head + final_norm.
	shapes = append(shapes, WeightMeta{Name: "embed", Shape: []int{V, D}})
	if !tieEmbeddings {
		shapes = append(shapes, WeightMeta{Name: "head", Shape: []int{D, V}})
	}
	shapes = append(shapes, WeightMeta{Name: "final_norm", Shape: []int{D}, IsNormScale: true, InitOne: true})
	shapes = append(shapes, bigramWeightShapes(D, bigramVocabSize, bigramDim)...)
	shapes = append(shapes, trigramWeightShapes(D, trigramVocabSize, trigramDim)...)

	if parallelResidual {
		if err := validateParallelResidualBlocks(blocks); err != nil {
			return nil, err
		}
		if unet {
			return nil, fmt.Errorf("parallel_residual is not supported with unet")
		}
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
		blk, err := streamWeightShapesWithRefsAndParallel(blocks, refs, D, T, 1, V, mlpMult, blockScales, residMix, parallelResidual)
		if err != nil {
			return nil, fmt.Errorf("blocks: %w", err)
		}
		shapes = append(shapes, blk...)
	}

	return shapes, nil
}
