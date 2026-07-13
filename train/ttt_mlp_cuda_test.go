//go:build mlx && cgo && linux

package train

import (
	"math"
	"testing"
)

func TestTTTMLPCUDAStatefulPrimitiveSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX CUDA backend not available")
	}
	configPath, weightsPath, _ := newTTTMLPInferenceFixture(t)
	tokens := []int{1, 2, 3, 4, 5, 6, 7}
	t.Setenv("MIXLAB_TTT_MLP_DISABLE_CUDA_PRIMITIVE", "0")
	session, err := NewTTTMLPInferenceSession(configPath, weightsPath)
	if err != nil {
		t.Fatalf("NewTTTMLPInferenceSession: %v", err)
	}
	defer session.Close()
	state, err := session.NewState()
	if err != nil {
		t.Fatalf("NewState: %v", err)
	}
	defer state.Close()
	logits, err := session.Prefill(state, tokens)
	if err != nil {
		t.Fatalf("CUDA stateful prefill: %v", err)
	}
	if len(logits) != len(tokens)*session.Config().VocabSize {
		t.Fatalf("logits=%d, want %d", len(logits), len(tokens)*session.Config().VocabSize)
	}
	for i, value := range logits {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("logits[%d]=%g is non-finite", i, value)
		}
	}
	stats := session.Stats()
	if stats.Tokens != uint64(len(tokens)) || stats.Evaluations < 2 {
		t.Fatalf("stateful CUDA stats=%+v", stats)
	}
}
