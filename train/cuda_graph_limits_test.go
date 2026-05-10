package train

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/gpu"
)

func TestMergeCUDAGraphLimits_PrioritizesSmallGraphCap(t *testing.T) {
	got := mergeCUDAGraphLimits(
		gpu.CUDAGraphLimits{MaxOpsPerBuffer: 16000},
		gpu.CUDAGraphLimits{MaxOpsPerBuffer: 64, MaxMBPerBuffer: 128, GraphCacheSize: 1024},
	)

	if got.MaxOpsPerBuffer != 64 {
		t.Fatalf("max ops=%d, want small graph cap 64", got.MaxOpsPerBuffer)
	}
	if got.MaxMBPerBuffer != 128 {
		t.Fatalf("max MB=%d, want 128", got.MaxMBPerBuffer)
	}
	if got.GraphCacheSize != 1024 {
		t.Fatalf("graph cache size=%d, want 1024", got.GraphCacheSize)
	}
}

func TestCUDAGraphLimitsForSelection_LoadsConfigDir(t *testing.T) {
	dir := t.TempDir()
	writeTestConfig(t, filepath.Join(dir, "plain.json"), `{
		"name": "plain_ops_tune",
		"model_dim": 128,
		"vocab_size": 256,
		"seq_len": 1024,
		"blocks": [
			{"type": "plain", "heads": 4},
			{"type": "swiglu"}
		],
		"training": {"steps": 1, "lr": 0.001, "batch_tokens": 1024}
	}`)
	writeTestConfig(t, filepath.Join(dir, "mamba3.json"), `{
		"name": "mamba3_cuda_graph_cap",
		"model_dim": 448,
		"vocab_size": 1024,
		"seq_len": 4096,
		"tie_embeddings": false,
		"blocks": [
			{"type": "mamba3-canonical"}, {"type": "swiglu"},
			{"type": "mamba3-canonical"}, {"type": "swiglu"},
			{"type": "mamba3-canonical"}, {"type": "swiglu"},
			{"type": "mamba3-canonical"}, {"type": "swiglu"},
			{"type": "mamba3-canonical"}, {"type": "swiglu"},
			{"type": "mamba3-canonical"}, {"type": "swiglu"},
			{"type": "mamba3-canonical"}, {"type": "swiglu"},
			{"type": "mamba3-canonical"}, {"type": "swiglu"}
		],
		"training": {"steps": 1, "lr": 0.001, "batch_tokens": 4096}
	}`)

	got := cudaGraphLimitsForSelection("", dir)
	if got.MaxOpsPerBuffer != 64 {
		t.Fatalf("max ops=%d, want Mamba3 cap 64", got.MaxOpsPerBuffer)
	}
	if got.MaxMBPerBuffer != 128 {
		t.Fatalf("max MB=%d, want Mamba3 cap 128", got.MaxMBPerBuffer)
	}
	if got.GraphCacheSize != 1024 {
		t.Fatalf("graph cache size=%d, want Mamba3 graph cache 1024", got.GraphCacheSize)
	}
}

func TestConfigureCUDAGraphLimits_AppliesConfigWithoutOverridingUserEnv(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "mamba3.json")
	writeTestConfig(t, path, `{
		"name": "mamba3_cuda_graph_cap",
		"model_dim": 448,
		"vocab_size": 1024,
		"seq_len": 4096,
		"tie_embeddings": false,
		"blocks": [
			{"type": "mamba3-canonical"}, {"type": "swiglu"},
			{"type": "mamba3-canonical"}, {"type": "swiglu"},
			{"type": "mamba3-canonical"}, {"type": "swiglu"},
			{"type": "mamba3-canonical"}, {"type": "swiglu"},
			{"type": "mamba3-canonical"}, {"type": "swiglu"},
			{"type": "mamba3-canonical"}, {"type": "swiglu"},
			{"type": "mamba3-canonical"}, {"type": "swiglu"},
			{"type": "mamba3-canonical"}, {"type": "swiglu"}
		],
		"training": {"steps": 1, "lr": 0.001, "batch_tokens": 4096}
	}`)

	t.Setenv(gpu.MLXMaxOpsPerBufferEnv, "999")
	t.Setenv(gpu.MLXCUDAGraphCacheEnv, "777")
	unsetEnvForTest(t, gpu.MLXMaxMBPerBufferEnv)

	ConfigureCUDAGraphLimits(path, "")

	if got := os.Getenv(gpu.MLXMaxOpsPerBufferEnv); got != "999" {
		t.Fatalf("%s=%q, want user value 999", gpu.MLXMaxOpsPerBufferEnv, got)
	}
	if got := os.Getenv(gpu.MLXMaxMBPerBufferEnv); got != "128" {
		t.Fatalf("%s=%q, want 128", gpu.MLXMaxMBPerBufferEnv, got)
	}
	if got := os.Getenv(gpu.MLXCUDAGraphCacheEnv); got != "777" {
		t.Fatalf("%s=%q, want user value 777", gpu.MLXCUDAGraphCacheEnv, got)
	}
}

func writeTestConfig(t *testing.T, path, contents string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(contents), 0644); err != nil {
		t.Fatalf("WriteFile(%s): %v", path, err)
	}
}

func unsetEnvForTest(t *testing.T, key string) {
	t.Helper()
	old, existed := os.LookupEnv(key)
	if err := os.Unsetenv(key); err != nil {
		t.Fatalf("Unsetenv(%s): %v", key, err)
	}
	t.Cleanup(func() {
		if existed {
			_ = os.Setenv(key, old)
		} else {
			_ = os.Unsetenv(key)
		}
	})
}
