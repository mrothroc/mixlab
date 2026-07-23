package train

import (
	"math"
	"os"
	"os/exec"
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

func TestExportHFGPT2WritesNullForUnresolvedSpecialTokens(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, strictGPT2SpecialTokenTestConfig())
	removeTokenizerSpecialDeclarations(t, tokenizerDir)
	outDir := filepath.Join(dir, "hf_gpt2")
	if err := RunExportHF(ExportHFOptions{
		ConfigPath:      cfgPath,
		SafetensorsLoad: weightsPath,
		OutputDir:       outDir,
		TokenizerSource: tokenizerDir,
	}); err != nil {
		t.Fatalf("RunExportHF: %v", err)
	}
	var cfg map[string]any
	readJSON(t, filepath.Join(outDir, "config.json"), &cfg)
	for _, key := range []string{"bos_token_id", "eos_token_id", "pad_token_id"} {
		value, ok := cfg[key]
		if !ok {
			t.Fatalf("config omitted %s; GPT2Config would restore its legacy out-of-vocab default", key)
		}
		if value != nil {
			t.Fatalf("config %s=%v want null", key, value)
		}
	}
}

func TestExportHFGPT2SpecialTokensLoadInTransformers(t *testing.T) {
	if os.Getenv("HF_PARITY") != "1" {
		t.Skip("set HF_PARITY=1 to run the Transformers integration check")
	}
	python := os.Getenv("HF_PARITY_PYTHON")
	if python == "" {
		python = "python3"
	}
	if err := exec.Command(python, "-c", "import torch, transformers, safetensors").Run(); err != nil {
		t.Skipf("python HF dependencies unavailable via %q: %v", python, err)
	}

	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, strictGPT2SpecialTokenTestConfig())
	removeTokenizerSpecialDeclarations(t, tokenizerDir)
	writeHFExportDatasetManifest(t, tokenizerDir, 7, map[string]int{
		"pad": 0,
		"bos": 3,
		"eos": 2,
	})
	outDir := filepath.Join(dir, "hf_gpt2")
	if err := RunExportHF(ExportHFOptions{
		ConfigPath:      cfgPath,
		SafetensorsLoad: weightsPath,
		OutputDir:       outDir,
		TokenizerSource: tokenizerDir,
	}); err != nil {
		t.Fatalf("RunExportHF: %v", err)
	}
	if err := os.Remove(filepath.Join(tokenizerDir, "mixlab.dataset.json")); err != nil {
		t.Fatal(err)
	}
	nullOutDir := filepath.Join(dir, "hf_gpt2_unresolved")
	if err := RunExportHF(ExportHFOptions{
		ConfigPath:      cfgPath,
		SafetensorsLoad: weightsPath,
		OutputDir:       nullOutDir,
		TokenizerSource: tokenizerDir,
	}); err != nil {
		t.Fatalf("RunExportHF unresolved specials: %v", err)
	}
	script := `
import torch
from transformers import AutoModelForCausalLM, GPT2Config, LogitsProcessor
model = AutoModelForCausalLM.from_pretrained(r"` + outDir + `")
assert model.config.bos_token_id == 3
assert model.config.eos_token_id == 2
assert model.config.pad_token_id == 0
assert model.generation_config.bos_token_id == 3
assert model.generation_config.eos_token_id == 2
assert model.generation_config.pad_token_id == 0
class ForceEOS(LogitsProcessor):
    def __call__(self, input_ids, scores):
        scores.fill_(-float("inf"))
        scores[:, 2] = 0
        return scores
generated = model.generate(
    torch.tensor([[3]]),
    do_sample=True,
    max_new_tokens=2,
    logits_processor=[ForceEOS()],
)
assert generated.tolist() == [[3, 2]]
unresolved = GPT2Config.from_pretrained(r"` + nullOutDir + `")
assert unresolved.bos_token_id is None
assert unresolved.eos_token_id is None
assert unresolved.pad_token_id is None
`
	output, err := exec.Command(python, "-c", script).CombinedOutput()
	if err != nil {
		t.Fatalf("Transformers failed to load native GPT-2 special-token metadata: %v\n%s", err, output)
	}
	if strings.Contains(string(output), "50256") || strings.Contains(strings.ToLower(string(output)), "out of range") {
		t.Fatalf("Transformers emitted an invalid-special-token warning:\n%s", output)
	}
}

func strictGPT2SpecialTokenTestConfig() string {
	return `{
		"name":"gpt2_specials",
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
			{"type":"plain","heads":2,"attention_mask":"causal","attn_bias":true,"ffn_activation":"gelu_new","ffn_pre_norm":true,"ffn_bias":true}
		],
		"training":{"batch_tokens":3,"objective":"causal","seed":7}
	}`
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

func TestExportHFGPT2RejectsPerBlockSettingsNotRepresentableInGPT2Config(t *testing.T) {
	tests := []struct {
		name       string
		blocksJSON string
		want       string
	}{
		{
			name: "mixed_heads",
			blocksJSON: `[
				{"type":"plain","heads":2,"attention_mask":"causal","attn_bias":true,"ffn_activation":"gelu_new","ffn_pre_norm":true,"ffn_bias":true},
				{"type":"plain","heads":3,"attention_mask":"causal","attn_bias":true,"ffn_activation":"gelu_new","ffn_pre_norm":true,"ffn_bias":true}
			]`,
			want: "head count",
		},
		{
			name: "mixed_activations",
			blocksJSON: `[
				{"type":"plain","heads":2,"attention_mask":"causal","attn_bias":true,"ffn_activation":"gelu_new","ffn_pre_norm":true,"ffn_bias":true},
				{"type":"plain","heads":2,"attention_mask":"causal","attn_bias":true,"ffn_activation":"gelu","ffn_pre_norm":true,"ffn_bias":true}
			]`,
			want: "FFN activation",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
				"name":"bad_gpt2_per_block",
				"model_dim":6,
				"vocab_size":7,
				"seq_len":3,
				"mlp_mult":2.0,
				"tie_embeddings":true,
				"norm_type":"layernorm",
				"norm_affine":true,
				"positional_embedding":"learned_absolute",
				"hf_export_format":"gpt2",
				"blocks":`+tt.blocksJSON+`,
				"training":{"batch_tokens":3,"objective":"causal","seed":7}
			}`)
			err := RunExportHF(ExportHFOptions{
				ConfigPath:      cfgPath,
				SafetensorsLoad: weightsPath,
				OutputDir:       filepath.Join(dir, "out"),
				TokenizerSource: tokenizerDir,
			})
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("RunExportHF error=%v want substring %q", err, tt.want)
			}
		})
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
