package arch

import (
	"strings"
	"testing"
)

func validBatchNormConfigJSON() []byte {
	return []byte(`{
		"name":"batchnorm_s4d_tiny",
		"model_dim":8,
		"vocab_size":32,
		"seq_len":4,
		"norm_type":"batchnorm",
		"blocks":[{"type":"s4d","state_size":8},{"type":"swiglu"}],
		"training":{
			"objective":"classification",
			"classification":{"num_labels":2},
			"batch_tokens":8,
			"steps":2,
			"lr":0.001
		}
	}`)
}

func TestBatchNormConfigAndPersistentBuffers(t *testing.T) {
	cfg, err := ParseArchConfig(validBatchNormConfigJSON(), "batchnorm-test")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	norm := cfg.EffectiveNormSpec()
	if norm.Type != NormTypeBatchNorm || norm.Eps != 1e-5 || norm.Momentum != 0.1 || !norm.Affine {
		t.Fatalf("effective norm=%+v", norm)
	}

	metas, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatalf("CollectWeightShapesFromConfig: %v", err)
	}
	var bufferElements int64
	var bufferCount int
	var trainableElements int64
	for _, meta := range metas {
		n := int64(1)
		for _, dim := range meta.Shape {
			n *= int64(dim)
		}
		if meta.IsBuffer {
			bufferCount++
			bufferElements += n
			if !strings.HasSuffix(meta.Name, "_running_mean") && !strings.HasSuffix(meta.Name, "_running_var") {
				t.Fatalf("unexpected model buffer %q", meta.Name)
			}
		} else {
			trainableElements += n
		}
	}
	if bufferCount != 6 {
		t.Fatalf("buffer tensors=%d want 6 (pre-norm for two blocks plus final norm)", bufferCount)
	}
	params, expanded, err := ParameterCountsFromConfig(cfg)
	if err != nil {
		t.Fatalf("ParameterCountsFromConfig: %v", err)
	}
	if params != trainableElements || expanded != trainableElements {
		t.Fatalf("parameter counts=(%d,%d) trainable metadata=%d buffers=%d", params, expanded, trainableElements, bufferElements)
	}

	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	if got := countOps(prog, OpBatchNorm); got != 3 {
		t.Fatalf("BatchNorm ops=%d want 3", got)
	}
	for _, op := range prog.Ops {
		if op.Code == OpBatchNorm && (len(op.Inputs) != 5 || len(op.Outputs) != 3) {
			t.Fatalf("BatchNorm op contract inputs=%d outputs=%d", len(op.Inputs), len(op.Outputs))
		}
	}
}

func TestBatchNormValidationBoundaries(t *testing.T) {
	cases := []struct {
		name    string
		replace string
		with    string
		want    string
	}{
		{"bad momentum", `"norm_type":"batchnorm",`, `"norm_type":"batchnorm","batchnorm_momentum":1.1,`, "batchnorm_momentum"},
		{"non classification", `"objective":"classification",`, `"objective":"causal",`, "classification"},
		{"post norm", `"norm_type":"batchnorm",`, `"norm_type":"batchnorm","norm_placement":"post",`, "norm_placement"},
		{"affine false", `"norm_type":"batchnorm",`, `"norm_type":"batchnorm","norm_affine":false,`, "norm_affine"},
		{"one sample", `"batch_tokens":8,`, `"batch_tokens":1,`, "batch_tokens > 1"},
		{"swa", `"steps":2,`, `"steps":2,"swa_start":1,`, "SWA"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw := strings.Replace(string(validBatchNormConfigJSON()), tc.replace, tc.with, 1)
			_, err := ParseArchConfig([]byte(raw), tc.name)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error=%v want substring %q", err, tc.want)
			}
		})
	}
}

func TestBatchNormDisabledLeavesLegacyNormLayoutUnchanged(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"model_dim":8,"vocab_size":32,"seq_len":4,
		"blocks":[{"type":"s4d","state_size":8}],
		"training":{"objective":"causal","batch_tokens":4}
	}`), "legacy-norm")
	if err != nil {
		t.Fatal(err)
	}
	metas, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, meta := range metas {
		if meta.IsBuffer {
			t.Fatalf("legacy config unexpectedly emitted buffer %q", meta.Name)
		}
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if got := countOps(prog, OpBatchNorm); got != 0 {
		t.Fatalf("legacy config emitted %d BatchNorm ops", got)
	}
}
