package arch

import "fmt"

type linearFramesInputOptions struct {
	BatchSize           int
	SeqLen              int
	ModelDim            int
	FeatureDim          int
	ProjectionIndex     int
	NextWeightIndex     int
	Bias                bool
	Norm                string
	NormEps             float32
	PositionalEmbedding string
	MaxPositions        int
	EmbeddingDropout    float32
}

// emitLinearFramesInputIR projects generic continuous feature sequences into
// the canonical flattened x stream consumed by every backbone block.
func emitLinearFramesInputIR(prog *Program, opts linearFramesInputOptions) (int, error) {
	if prog == nil {
		return 0, fmt.Errorf("nil IR program")
	}
	if opts.BatchSize <= 0 || opts.SeqLen <= 0 || opts.ModelDim <= 0 || opts.FeatureDim <= 0 {
		return 0, fmt.Errorf(
			"invalid linear frame input shape B=%d T=%d F=%d D=%d",
			opts.BatchSize, opts.SeqLen, opts.FeatureDim, opts.ModelDim,
		)
	}
	B, T, D := opts.BatchSize, opts.SeqLen, opts.ModelDim
	wi := opts.NextWeightIndex
	prog.MatMul("continuous_frames", weightName(opts.ProjectionIndex), "x_frame_projected")
	state := "x_frame_projected"
	if opts.Bias {
		prog.Add(state, weightName(wi), "x_frame_biased")
		state = "x_frame_biased"
		wi++
	}
	switch normalizeInputAdapterNorm(opts.Norm) {
	case InputAdapterNormNone:
	case InputAdapterNormLayerNorm:
		eps := opts.NormEps
		if eps <= 0 {
			eps = 1e-5
		}
		prog.LayerNorm(state, weightName(wi), weightName(wi+1), "x_frame_norm", eps)
		state = "x_frame_norm"
		wi += 2
	default:
		return 0, fmt.Errorf("unsupported linear frame input norm %q", opts.Norm)
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
