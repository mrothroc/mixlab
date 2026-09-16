package train

import (
	"math"
	"testing"
)

const clsContinuousConfig = `{
 "name":"cls_continuous","model_dim":8,"seq_len":6,"max_positions":7,
 "positional_embedding":"learned_absolute","norm_type":"layernorm","norm_affine":true,
 "input_adapter":{"kind":"linear_frames","feature_dim":1,"norm":"none"},
 "blocks":[{"type":"plain","heads":2,"attention_mask":"bidirectional","ffn_activation":"gelu","ffn_pre_norm":true,"ffn_bias":true}],
 "training":{"objective":"classification","classification":{"num_labels":3,"pooling":"cls","classifier_dropout":0},
 "optimizer":"adamw","steps":30,"batch_tokens":6,"lr":0.001,"grad_clip":1,"weight_decay":0,"seed":23}
}`

func TestCLSPoolingExportAndInitialization(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(clsContinuousConfig), t.Name())
	if err != nil {
		t.Fatal(err)
	}
	if err := validateHFExportConfig(cfg); err != nil {
		t.Fatal(err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	mapping, err := buildHFWeightMap(cfg, shapes)
	if err != nil {
		t.Fatal(err)
	}
	i := weightShapeIndex(shapes, "cls_token")
	if i < 0 {
		t.Fatal("missing CLS")
	}
	if mapping[i].HF != "cls_token" {
		t.Fatalf("mapping %v", mapping[i])
	}
	a := initWeightData(shapes, 23, "normal", 0.02)
	b := initWeightData(shapes, 23, "normal", 1)
	for j := range a[i] {
		if math.Abs(float64(a[i][j]*50-b[i][j])) > 1e-6 {
			t.Fatal("normal std ignored")
		}
	}
	// An unset weight_init_std falls back to the class token's own default
	// rather than collapsing the tensor to zero.
	d := initWeightData(shapes, 23, "normal", 0)
	for j := range a[i] {
		if math.Abs(float64(a[i][j]-d[i][j])) > 1e-6 {
			t.Fatalf("class token default std not applied: %v vs %v", a[i][j], d[i][j])
		}
	}
}
