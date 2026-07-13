//go:build mlx && cgo && (darwin || linux)

package train

import (
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/gpu"
)

type tttMLPParityBlockState struct {
	MLP      []float32 `json:"mlp"`
	Gradient []float32 `json:"gradient"`
	Conv     []float32 `json:"conv"`
	Offset   int       `json:"offset"`
}

func writeTTTMLPCachedParityFixture(
	t *testing.T,
	cfgPath, weightsPath, outDir string,
	inputIDs []int,
	split int,
) {
	t.Helper()
	if split <= 0 || split >= len(inputIDs) {
		t.Fatalf("invalid TTT parity split=%d for %d tokens", split, len(inputIDs))
	}
	session, err := NewTTTMLPInferenceSession(cfgPath, weightsPath)
	if err != nil {
		t.Fatalf("NewTTTMLPInferenceSession: %v", err)
	}
	defer session.Close()
	state, err := session.NewState()
	if err != nil {
		t.Fatalf("NewState: %v", err)
	}
	defer state.Close()
	if _, err := session.Prefill(state, inputIDs[:split]); err != nil {
		t.Fatalf("TTT native prefix: %v", err)
	}
	continuation, err := session.Prefill(state, inputIDs[split:])
	if err != nil {
		t.Fatalf("TTT native continuation: %v", err)
	}
	blocks := make([]tttMLPParityBlockState, len(state.handles))
	for i, handles := range state.handles {
		mlp, err := gpu.ReadHandle(handles.mlp)
		if err != nil {
			t.Fatalf("read TTT block %d MLP state: %v", i, err)
		}
		gradient, err := gpu.ReadHandle(handles.grad)
		if err != nil {
			t.Fatalf("read TTT block %d gradient state: %v", i, err)
		}
		conv, err := gpu.ReadHandle(handles.conv)
		if err != nil {
			t.Fatalf("read TTT block %d convolution state: %v", i, err)
		}
		blocks[i] = tttMLPParityBlockState{
			MLP: mlp, Gradient: gradient, Conv: conv, Offset: state.offsets[i],
		}
	}
	if err := writeJSONFile(filepath.Join(outDir, "parity_ttt_state.json"), map[string]any{
		"split":               split,
		"continuation_logits": continuation,
		"blocks":              blocks,
	}); err != nil {
		t.Fatalf("write TTT cached parity fixture: %v", err)
	}
}
