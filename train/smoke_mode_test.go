package train

import (
	"testing"

	"github.com/mrothroc/mixlab/gpu"
)

// TestRunSmoke_StubPath verifies that runSmoke returns a clear error (not a
// panic) when the GPU backend is unavailable (the stub build path).
func TestRunSmoke_StubPath(t *testing.T) {
	if gpu.Available() {
		t.Skip("skipping stub-path test: MLX backend is available")
	}

	err := runSmoke()
	if err == nil {
		t.Fatal("expected runSmoke to return an error on the stub path, got nil")
	}
	t.Logf("runSmoke stub-path error (expected): %v", err)
}

// TestRunIRHealthCheck_StubPath verifies the IR health check returns a clear
// error when the GPU backend is unavailable.
func TestRunIRHealthCheck_StubPath(t *testing.T) {
	if gpu.Available() {
		t.Skip("skipping stub-path test: MLX backend is available")
	}

	err := runIRHealthCheck()
	if err == nil {
		t.Fatal("expected runIRHealthCheck to return an error on the stub path, got nil")
	}
	t.Logf("runIRHealthCheck stub-path error (expected): %v", err)
}
