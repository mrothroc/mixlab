package train

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExportHFGPT2NativeFormat(t *testing.T) {
	dir := t.TempDir()
	cfgJSON := `{
		"name":"gpt2_native_tiny",
		"model_dim":6,
		"vocab_size":7,
		"seq_len":3,
		"mlp_mult":2.0,
		"tie_embeddings":true,
		"norm_type":"layernorm",
		"norm_affine":true,
		"positional_embedding":"learned_absolute",
		"max_positions":5,
		"embedding_dropout":0.1,
		"hidden_dropout":0.2,
		"attn_dropout":0.3,
		"hf_export_format":"gpt2",
		"blocks":[
			{"type":"plain","heads":2,"attention_mask":"causal","attn_bias":true,"ffn_activation":"gelu_new","ffn_pre_norm":true,"ffn_bias":true}
		],
		"training":{"batch_tokens":3,"objective":"causal","seed":7,"weight_init_std":0.02}
	}`
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, cfgJSON)
	outDir := filepath.Join(dir, "hf_gpt2")
	if err := RunExportHF(ExportHFOptions{
		ConfigPath:      cfgPath,
		SafetensorsLoad: weightsPath,
		OutputDir:       outDir,
		TokenizerSource: tokenizerDir,
	}); err != nil {
		t.Fatalf("RunExportHF: %v", err)
	}
	if _, err := os.Stat(filepath.Join(outDir, "configuration_mixlab.py")); !os.IsNotExist(err) {
		t.Fatalf("native GPT-2 export should not write custom Mixlab template, stat err=%v", err)
	}
	var cfg map[string]any
	readJSON(t, filepath.Join(outDir, "config.json"), &cfg)
	if got := cfg["model_type"]; got != "gpt2" {
		t.Fatalf("model_type=%v want gpt2", got)
	}
	if got := int(cfg["n_positions"].(float64)); got != 5 {
		t.Fatalf("n_positions=%d want 5", got)
	}
	if got := cfg["activation_function"]; got != "gelu_new" {
		t.Fatalf("activation_function=%v want gelu_new", got)
	}
	if got := cfg["architectures"].([]any)[0]; got != "GPT2LMHeadModel" {
		t.Fatalf("architectures[0]=%v want GPT2LMHeadModel", got)
	}
	tensors, err := loadSafetensors(filepath.Join(outDir, "model.safetensors"))
	if err != nil {
		t.Fatalf("load exported model.safetensors: %v", err)
	}
	wantShapes := map[string][]int{
		"transformer.wte.weight":             {7, 6},
		"transformer.wpe.weight":             {5, 6},
		"transformer.h.0.ln_1.weight":        {6},
		"transformer.h.0.ln_1.bias":          {6},
		"transformer.h.0.attn.c_attn.weight": {6, 18},
		"transformer.h.0.attn.c_attn.bias":   {18},
		"transformer.h.0.attn.c_proj.weight": {6, 6},
		"transformer.h.0.attn.c_proj.bias":   {6},
		"transformer.h.0.ln_2.weight":        {6},
		"transformer.h.0.ln_2.bias":          {6},
		"transformer.h.0.mlp.c_fc.weight":    {6, 12},
		"transformer.h.0.mlp.c_fc.bias":      {12},
		"transformer.h.0.mlp.c_proj.weight":  {12, 6},
		"transformer.h.0.mlp.c_proj.bias":    {6},
		"transformer.ln_f.weight":            {6},
		"transformer.ln_f.bias":              {6},
		"lm_head.weight":                     {7, 6},
	}
	for name, shape := range wantShapes {
		got, ok := tensors[name]
		if !ok {
			t.Fatalf("missing GPT-2 tensor %q", name)
		}
		if !sameIntSlice(got.Shape, shape) {
			t.Fatalf("%s shape=%v want %v", name, got.Shape, shape)
		}
	}
	cfgObj, err := LoadArchConfig(cfgPath)
	if err != nil {
		t.Fatalf("LoadArchConfig: %v", err)
	}
	shapes, err := computeWeightShapes(cfgObj)
	if err != nil {
		t.Fatalf("computeWeightShapes: %v", err)
	}
	nativeWeights, err := loadSafetensorsWeights(weightsPath, shapes)
	if err != nil {
		t.Fatalf("loadSafetensorsWeights: %v", err)
	}
	wq := nativeWeights[nthWeightShapeIndex(t, shapes, "wq")]
	wk := nativeWeights[nthWeightShapeIndex(t, shapes, "wk")]
	wv := nativeWeights[nthWeightShapeIndex(t, shapes, "wv")]
	wantCATTN := concatMatrixColumns(6, 6, wq, wk, wv)
	gotCATTN, err := decodeSafetensorFloat32("transformer.h.0.attn.c_attn.weight", []int{6, 18}, tensors)
	if err != nil {
		t.Fatalf("decode c_attn.weight: %v", err)
	}
	if maxAbsDiffFloat32(gotCATTN, wantCATTN) != 0 {
		t.Fatal("packed c_attn.weight does not match concatenated Mixlab Q/K/V weights")
	}
	gotHead, err := decodeSafetensorFloat32("lm_head.weight", []int{7, 6}, tensors)
	if err != nil {
		t.Fatalf("decode lm_head.weight: %v", err)
	}
	if maxAbsDiffFloat32(gotHead, nativeWeights[nthWeightShapeIndex(t, shapes, "embed")]) != 0 {
		t.Fatal("lm_head.weight is not tied to exported token embeddings")
	}
}

func TestExportHFGPT2RejectsNonStrictPlainBlock(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
		"name":"bad_gpt2",
		"model_dim":6,
		"vocab_size":7,
		"seq_len":3,
		"mlp_mult":2.0,
		"tie_embeddings":true,
		"norm_type":"layernorm",
		"norm_affine":true,
		"positional_embedding":"learned_absolute",
		"hf_export_format":"gpt2",
		"blocks":[
			{"type":"plain","heads":2,"attention_mask":"causal","attn_bias":true,"ffn_activation":"gelu_new","ffn_pre_norm":true}
		],
		"training":{"batch_tokens":3,"objective":"causal","seed":7}
	}`)
	err := RunExportHF(ExportHFOptions{
		ConfigPath:      cfgPath,
		SafetensorsLoad: weightsPath,
		OutputDir:       filepath.Join(dir, "out"),
		TokenizerSource: tokenizerDir,
	})
	if err == nil || !strings.Contains(err.Error(), "ffn_bias") {
		t.Fatalf("RunExportHF error=%v want ffn_bias strict rejection", err)
	}
}

func nthWeightShapeIndex(t *testing.T, shapes []WeightShape, name string) int {
	t.Helper()
	for i, shape := range shapes {
		if shape.Name == name {
			return i
		}
	}
	t.Fatalf("missing weight %q", name)
	return -1
}

func maxAbsDiffFloat32(a, b []float32) float32 {
	if len(a) != len(b) {
		return float32(math.Inf(1))
	}
	var max float32
	for i := range a {
		d := float32(math.Abs(float64(a[i] - b[i])))
		if d > max {
			max = d
		}
	}
	return max
}
