package train

import (
	"fmt"
	"runtime"
)

var errInferenceSessionClosed = fmt.Errorf("inference session is closed")

// InferenceSession holds a loaded model ready for repeated forward passes.
// It encapsulates the loaded config and GPU trainer so callers can evaluate
// caller-supplied token arrays without reloading weights each time.
//
// Sessions are bound to the OS thread that created them because the underlying
// MLX trainer requires thread affinity. Callers should create, use, and close a
// session from the same goroutine.
type InferenceSession struct {
	cfg         *ArchConfig
	trainer     GPUTrainer
	closed      bool
	weightCount int
}

// NewInferenceSession loads a config and weights, builds the IR program, and
// initializes the GPU trainer. The returned session must be closed via
// (*InferenceSession).Close to release GPU resources.
//
// Example:
//
//	sess, err := train.NewInferenceSession("model.json", "weights.safetensors")
//	if err != nil {
//	    return err
//	}
//	defer sess.Close()
//	nlls, err := sess.EvalTokens(myTokens)
func NewInferenceSession(configPath, safetensorsLoad string) (*InferenceSession, error) {
	runtime.LockOSThread()

	if configPath == "" {
		runtime.UnlockOSThread()
		return nil, fmt.Errorf("config path is required")
	}
	if safetensorsLoad == "" {
		runtime.UnlockOSThread()
		return nil, fmt.Errorf("safetensors path is required")
	}

	cfg, err := LoadArchConfig(configPath)
	if err != nil {
		runtime.UnlockOSThread()
		return nil, err
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		runtime.UnlockOSThread()
		return nil, fmt.Errorf("build IR program: %w", err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		runtime.UnlockOSThread()
		return nil, fmt.Errorf("compute weight shapes: %w", err)
	}
	loadedWeights, err := loadSafetensorsWeights(safetensorsLoad, shapes)
	if err != nil {
		runtime.UnlockOSThread()
		return nil, fmt.Errorf("load safetensors %q: %w", safetensorsLoad, err)
	}
	trainer, err := initGPUTrainer(prog, cfg, loadedWeights)
	if err != nil {
		runtime.UnlockOSThread()
		return nil, fmt.Errorf("init GPU trainer: %w", err)
	}

	return &InferenceSession{
		cfg:         cfg,
		trainer:     trainer,
		weightCount: len(loadedWeights),
	}, nil
}

// Config returns the loaded config. Callers should treat the returned value as
// read-only.
func (s *InferenceSession) Config() *ArchConfig {
	if s == nil {
		return nil
	}
	return s.cfg
}

// EvalTokens runs next-token evaluation on a flat token stream and returns one
// per-token NLL for each adjacent token pair in the input.
//
// The caller passes a flat token slice `[t0, t1, ..., tN]`. The session scores
// each prediction `(t0 -> t1), (t1 -> t2), ...`, so the returned slice has
// length `len(tokens)-1`.
//
// Because the underlying IR program is built for a fixed batch shape, each call
// must provide a whole number of configured eval batches:
// `len(tokens)-1` must be a positive multiple of `Config().Training.BatchTokens`.
func (s *InferenceSession) EvalTokens(tokens []uint16) ([]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("nil inference session")
	}
	if s.closed || s.trainer == nil {
		return nil, errInferenceSessionClosed
	}
	if s.cfg == nil {
		return nil, fmt.Errorf("inference session has no config")
	}

	batchTokens := s.cfg.Training.BatchTokens
	seqLen := s.cfg.SeqLen
	if batchTokens <= 0 {
		return nil, fmt.Errorf("invalid batch_tokens=%d", batchTokens)
	}
	if seqLen <= 0 {
		return nil, fmt.Errorf("invalid seq_len=%d", seqLen)
	}
	if batchTokens%seqLen != 0 {
		return nil, fmt.Errorf("batch_tokens (%d) must be divisible by seq_len (%d)", batchTokens, seqLen)
	}
	if len(tokens) < 2 {
		return nil, fmt.Errorf("need at least 2 tokens, got %d", len(tokens))
	}

	pairs := len(tokens) - 1
	if pairs%batchTokens != 0 {
		return nil, fmt.Errorf("token pair count (%d) must be a multiple of batch_tokens (%d)", pairs, batchTokens)
	}

	batchSize := batchTokens / seqLen
	out := make([]float32, 0, pairs)
	for start := 0; start < pairs; start += batchTokens {
		window := tokens[start : start+batchTokens+1]
		xTok := make([]int, batchTokens)
		yTok := make([]int, batchTokens)
		for i := 0; i < batchTokens; i++ {
			xTok[i] = int(window[i])
			yTok[i] = int(window[i+1])
		}
		nlls, err := s.trainer.EvaluatePerTokenGPU(xTok, yTok, batchSize, seqLen)
		if err != nil {
			return nil, err
		}
		if len(nlls) != batchTokens {
			return nil, fmt.Errorf("per-token eval length mismatch: got=%d want=%d", len(nlls), batchTokens)
		}
		out = append(out, nlls...)
	}

	return out, nil
}

// Close releases GPU resources. It is safe to call multiple times.
func (s *InferenceSession) Close() error {
	if s == nil || s.closed {
		return nil
	}
	if s.trainer != nil {
		s.trainer.CloseTrainer()
		s.trainer = nil
	}
	s.closed = true
	runtime.UnlockOSThread()
	return nil
}
