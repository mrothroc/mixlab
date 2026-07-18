package arch

import "fmt"

// discreteTokenInputOptions describes the existing token-id input adapter.
// It stays internal until Mixlab has a second useful adapter implementation.
type discreteTokenInputOptions struct {
	BatchSize           int
	SeqLen              int
	ModelDim            int
	TokenWeightIndex    int
	NextWeightIndex     int
	PositionalEmbedding string
	MaxPositions        int
	Smear               smearEmbeddingOptions
	CharVocabSize       int
	CharDim             int
	CharMaxPerToken     int
	BigramVocabSize     int
	BigramDim           int
	TrigramVocabSize    int
	TrigramDim          int
	EmbeddingDropout    float32
}

// emitDiscreteTokenInputIR lowers token lookup and every model-level embedding
// feature into the canonical flattened `x` backbone input. The weight indices
// are explicit because single-head layouts reserve output/final-norm weights
// before model-level feature weights while multihead layouts do not.
func emitDiscreteTokenInputIR(prog *Program, opts discreteTokenInputOptions) (int, error) {
	if prog == nil {
		return 0, fmt.Errorf("nil IR program")
	}
	if opts.BatchSize <= 0 || opts.SeqLen <= 0 || opts.ModelDim <= 0 {
		return 0, fmt.Errorf("invalid discrete token input shape B=%d T=%d D=%d", opts.BatchSize, opts.SeqLen, opts.ModelDim)
	}
	B, T, D := opts.BatchSize, opts.SeqLen, opts.ModelDim
	wi := opts.NextWeightIndex
	prog.Embed(weightName(opts.TokenWeightIndex), "tokens", "x_embed")
	embedState := "x_embed"
	var err error
	if normalizePositionalEmbedding(opts.PositionalEmbedding) == PositionalEmbeddingLearnedAbsolute {
		embedState, wi, err = emitLearnedPositionEmbeddingIR(prog, embedState, B, T, D, wi, opts.MaxPositions)
		if err != nil {
			return 0, err
		}
	}
	if opts.Smear.Enabled {
		embedState, wi, err = emitSmearEmbeddingIR(prog, embedState, T, D, wi, opts.Smear)
		if err != nil {
			return 0, err
		}
	}

	xState := "x"
	if opts.CharVocabSize > 0 || opts.BigramVocabSize > 0 || opts.TrigramVocabSize > 0 {
		xState = "x_tok"
	}
	prog.Reshape(embedState, []int{B * T, D}, xState)
	featureBase := xState
	wi = emitCharIR(prog, featureBase, B, T, D, wi, opts.CharVocabSize, opts.CharDim, opts.CharMaxPerToken)
	if opts.CharVocabSize > 0 {
		featureBase = "x"
	}
	wi = emitBigramIR(prog, featureBase, B, T, D, wi, opts.BigramVocabSize, opts.BigramDim)
	if opts.BigramVocabSize > 0 {
		featureBase = "x"
	}
	wi = emitTrigramIR(prog, featureBase, B, T, D, wi, opts.TrigramVocabSize, opts.TrigramDim)
	if opts.EmbeddingDropout > 0 {
		prog.Dropout("x", opts.EmbeddingDropout, "x")
	}
	return wi, nil
}
