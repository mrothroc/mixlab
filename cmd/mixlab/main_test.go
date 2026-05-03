package main

import (
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestTunedMaxOpsForProgram_UsesGatedDeltaNetFloor(t *testing.T) {
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
	if got := tunedMaxOpsForProgram(cfg, prog); got != minGatedDeltaNetMaxOpsPerBuffer {
		t.Fatalf("tuned max ops=%d, want %d", got, minGatedDeltaNetMaxOpsPerBuffer)
	}
}

func TestTunedMaxOpsForProgram_KeepsPlainIRTune(t *testing.T) {
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
	if got := tunedMaxOpsForProgram(cfg, prog); got != want {
		t.Fatalf("tuned max ops=%d, want raw IR tune %d", got, want)
	}
}
