package gpu

import (
	"os"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestTuneCUDAGraphLimits_UsesGatedDeltaNetFloor(t *testing.T) {
	cfg, err := arch.ParseArchConfig([]byte(`{
		"name": "gdn_ops_floor",
		"model_dim": 128,
		"vocab_size": 256,
		"seq_len": 4096,
		"blocks": [
			{"type": "gated_deltanet", "heads": 4, "d_k": 16}
		],
		"training": {"steps": 1, "lr": 0.001, "batch_tokens": 4096}
	}`), "gdn_ops_floor")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	prog, err := arch.BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	rawTune := len(prog.Ops) * 3
	if rawTune >= minGatedDeltaNetMaxOpsPerBuffer {
		t.Fatalf("test fixture raw tune=%d, want below floor=%d", rawTune, minGatedDeltaNetMaxOpsPerBuffer)
	}

	got := TuneCUDAGraphLimits(prog)
	if got.MaxOpsPerBuffer != minGatedDeltaNetMaxOpsPerBuffer {
		t.Fatalf("max ops=%d, want %d", got.MaxOpsPerBuffer, minGatedDeltaNetMaxOpsPerBuffer)
	}
	if got.MaxMBPerBuffer != 0 {
		t.Fatalf("max MB=%d, want 0", got.MaxMBPerBuffer)
	}
}

func TestTuneCUDAGraphLimits_KeepsPlainIRTune(t *testing.T) {
	cfg, err := arch.ParseArchConfig([]byte(`{
		"name": "plain_ops_tune",
		"model_dim": 128,
		"vocab_size": 256,
		"seq_len": 1024,
		"blocks": [
			{"type": "plain", "heads": 4},
			{"type": "swiglu"}
		],
		"training": {"steps": 1, "lr": 0.001, "batch_tokens": 1024}
	}`), "plain_ops_tune")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	prog, err := arch.BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	want := len(prog.Ops) * 3

	got := TuneCUDAGraphLimits(prog)
	if got.MaxOpsPerBuffer != want {
		t.Fatalf("max ops=%d, want raw IR tune %d", got.MaxOpsPerBuffer, want)
	}
	if got.MaxMBPerBuffer != 0 {
		t.Fatalf("max MB=%d, want 0", got.MaxMBPerBuffer)
	}
}

func TestTuneCUDAGraphLimits_CapsCanonicalMamba3Scan(t *testing.T) {
	cfg, err := arch.ParseArchConfig([]byte(`{
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
	}`), "mamba3_cuda_graph_cap")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	prog, err := arch.BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	rawTune := len(prog.Ops) * 3
	if rawTune <= maxMamba3SelectiveScanOpsPerBuffer {
		t.Fatalf("test fixture raw tune=%d, want above cap=%d", rawTune, maxMamba3SelectiveScanOpsPerBuffer)
	}
	if !programHasOp(prog, arch.OpMamba3CanonicalBlock) {
		t.Fatal("test fixture missing canonical Mamba3 block op")
	}

	got := TuneCUDAGraphLimits(prog)
	if got.MaxOpsPerBuffer != maxMamba3SelectiveScanOpsPerBuffer {
		t.Fatalf("max ops=%d, want Mamba3 cap %d", got.MaxOpsPerBuffer, maxMamba3SelectiveScanOpsPerBuffer)
	}
	if got.MaxMBPerBuffer != maxMamba3SelectiveScanMBPerBuffer {
		t.Fatalf("max MB=%d, want Mamba3 cap %d", got.MaxMBPerBuffer, maxMamba3SelectiveScanMBPerBuffer)
	}
}

func TestApplyCUDAGraphLimits_PreservesUserEnv(t *testing.T) {
	t.Setenv(MLXMaxOpsPerBufferEnv, "999")
	unsetEnvForTest(t, MLXMaxMBPerBufferEnv)

	ApplyCUDAGraphLimits(CUDAGraphLimits{
		MaxOpsPerBuffer: 64,
		MaxMBPerBuffer:  128,
	})

	if got := os.Getenv(MLXMaxOpsPerBufferEnv); got != "999" {
		t.Fatalf("%s=%q, want user value 999", MLXMaxOpsPerBufferEnv, got)
	}
	if got := os.Getenv(MLXMaxMBPerBufferEnv); got != "128" {
		t.Fatalf("%s=%q, want 128", MLXMaxMBPerBufferEnv, got)
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
