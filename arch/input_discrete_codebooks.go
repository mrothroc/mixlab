package arch

import "fmt"

type discreteCodebookInputOptions struct {
	BatchSize           int
	SeqLen              int
	ModelDim            int
	NumCodebooks        int
	CodebookVocabSize   int
	Fusion              string
	FusionHiddenDim     int
	EmbeddingIndex      int
	NextWeightIndex     int
	Norm                string
	NormEps             float32
	PositionalEmbedding string
	MaxPositions        int
	EmbeddingDropout    float32
}

// emitDiscreteCodebookInputIR embeds Q synchronized discrete codebooks and
// fuses their per-timestep vectors into the canonical [B*T,D] backbone input.
func emitDiscreteCodebookInputIR(prog *Program, opts discreteCodebookInputOptions) (int, error) {
	if prog == nil {
		return 0, fmt.Errorf("nil IR program")
	}
	if opts.BatchSize <= 0 || opts.SeqLen <= 0 || opts.ModelDim <= 0 ||
		opts.NumCodebooks <= 0 || opts.CodebookVocabSize <= 0 {
		return 0, fmt.Errorf(
			"invalid discrete codebook input shape B=%d T=%d Q=%d V=%d D=%d",
			opts.BatchSize, opts.SeqLen, opts.NumCodebooks, opts.CodebookVocabSize, opts.ModelDim,
		)
	}
	B, T, Q, D := opts.BatchSize, opts.SeqLen, opts.NumCodebooks, opts.ModelDim
	wi := opts.NextWeightIndex
	prog.CodebookOffset("codebook_tokens", Q, opts.CodebookVocabSize, "codebook_indices")
	prog.Embed(weightName(opts.EmbeddingIndex), "codebook_indices", "codebook_embeddings")

	state := "codebook_fused"
	switch normalizeInputAdapterFusion(opts.Fusion) {
	case InputAdapterFusionAttentionMLP:
		prog.Reshape("codebook_embeddings", []int{B * T * Q, D}, "codebook_embeddings_flat")
		prog.MatMul("codebook_embeddings_flat", weightName(wi), "codebook_attn_hidden_linear")
		prog.Add("codebook_attn_hidden_linear", weightName(wi+1), "codebook_attn_hidden_biased")
		prog.ReLU("codebook_attn_hidden_biased", "codebook_attn_hidden")
		prog.MatMul("codebook_attn_hidden", weightName(wi+2), "codebook_attn_scores_flat")
		wi += 3
		prog.Reshape("codebook_attn_scores_flat", []int{B * T, Q}, "codebook_attn_scores")
		prog.Softmax("codebook_attn_scores", -1, "codebook_attn_weights")
		prog.Reshape("codebook_attn_weights", []int{B * T, 1, Q}, "codebook_attn_weights_b1q")
		prog.Reshape("codebook_embeddings", []int{B * T, Q, D}, "codebook_embeddings_bqd")
		prog.MatMul("codebook_attn_weights_b1q", "codebook_embeddings_bqd", "codebook_fused_b1d")
		prog.Reshape("codebook_fused_b1d", []int{B, T, D}, state)
	case InputAdapterFusionMean:
		prog.MeanAxis("codebook_embeddings", 2, state)
	default:
		return 0, fmt.Errorf("unsupported discrete codebook fusion %q", opts.Fusion)
	}

	switch normalizeInputAdapterNorm(opts.Norm) {
	case InputAdapterNormNone:
	case InputAdapterNormLayerNorm:
		eps := opts.NormEps
		if eps <= 0 {
			eps = 1e-5
		}
		prog.LayerNorm(state, weightName(wi), weightName(wi+1), "codebook_fused_norm", eps)
		state = "codebook_fused_norm"
		wi += 2
	default:
		return 0, fmt.Errorf("unsupported discrete codebook input norm %q", opts.Norm)
	}

	var err error
	if normalizePositionalEmbedding(opts.PositionalEmbedding) == PositionalEmbeddingLearnedAbsolute {
		state, wi, err = emitLearnedPositionEmbeddingIR(prog, state, B, T, D, wi, opts.MaxPositions)
		if err != nil {
			return 0, err
		}
	}
	prog.Reshape(state, []int{B * T, D}, "x")
	if opts.EmbeddingDropout > 0 {
		prog.Dropout("x", opts.EmbeddingDropout, "x")
	}
	return wi, nil
}
