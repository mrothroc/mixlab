package gpu

import (
	"strings"
	"testing"
)

func TestExplainBridgeErrorKeepsMLXCacheThrashText(t *testing.T) {
	const upstream = "Cache thrashing is happening, please set the environment variable " +
		"MLX_CUDA_GRAPH_CACHE_SIZE to a larger value than 400 to fix degraded performance."
	env := map[string]string{MLXMaxOpsPerBufferEnv: "64"}
	got := explainBridgeError(upstream, func(key string) (string, bool) {
		v, ok := env[key]
		return v, ok
	})
	if !strings.HasPrefix(got, upstream) {
		t.Fatalf("MLX's original message must lead the error, got %q", got)
	}
	for _, want := range []string{
		"MLX_MAX_OPS_PER_BUFFER (current 64)",
		"MLX_MAX_MB_PER_BUFFER (current unset)",
		"MLX_CUDA_GRAPH_CACHE_SIZE (current unset)",
		"canonical Mamba3",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("hint missing %q in %q", want, got)
		}
	}
}

func TestExplainBridgeErrorPassesOtherErrorsThrough(t *testing.T) {
	for _, message := range []string{
		"",
		"cudaGraphAddDependencies( graph_, from_nodes_.data(), to_nodes_.data(), from_nodes_.size()) failed: invalid argument",
		"[gather_axis] Cannot calculate VJP with respect to indices",
	} {
		if got := explainBridgeError(message, func(string) (string, bool) { return "", false }); got != message {
			t.Errorf("explainBridgeError(%q) = %q, want it unchanged", message, got)
		}
	}
}

func TestExplainBridgeErrorTreatsEmptyEnvAsUnset(t *testing.T) {
	got := explainBridgeError("Cache thrashing is happening", func(string) (string, bool) { return "", true })
	if !strings.Contains(got, "MLX_CUDA_GRAPH_CACHE_SIZE (current unset)") {
		t.Fatalf("empty variable should read as unset: %q", got)
	}
}
