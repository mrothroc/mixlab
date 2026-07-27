package train

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestBatchNormBuffersInitializeAndRoundTripInNativeSafetensors(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"name":"batchnorm_checkpoint",
		"model_dim":8,
		"vocab_size":32,
		"seq_len":4,
		"norm_type":"batchnorm",
		"blocks":[{"type":"s4d","state_size":8}],
		"training":{
			"objective":"classification",
			"classification":{"num_labels":2},
			"batch_tokens":8,
			"steps":1,
			"lr":0.001
		}
	}`), "batchnorm-checkpoint")
	if err != nil {
		t.Fatal(err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	weights := initWeightData(shapes, 42, "", 0)
	var buffers int
	for i, shape := range shapes {
		if !shape.IsBuffer {
			continue
		}
		buffers++
		if strings.HasSuffix(shape.Name, "_running_var") {
			for j, value := range weights[i] {
				if value != 1 {
					t.Fatalf("%s[%d]=%g want 1", shape.Name, j, value)
				}
			}
		}
	}
	if buffers != 4 {
		t.Fatalf("buffers=%d want 4", buffers)
	}

	path := filepath.Join(t.TempDir(), "batchnorm.safetensors")
	if err := exportSafetensors(path, cfg, shapes, weights); err != nil {
		t.Fatal(err)
	}
	loaded, err := loadSafetensorsWeights(path, shapes)
	if err != nil {
		t.Fatal(err)
	}
	for i, shape := range shapes {
		if !shape.IsBuffer {
			continue
		}
		for j := range weights[i] {
			if loaded[i][j] != weights[i][j] {
				t.Fatalf("%s[%d] round trip=%g want %g", shape.Name, j, loaded[i][j], weights[i][j])
			}
		}
	}
}
