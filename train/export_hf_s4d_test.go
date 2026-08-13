package train

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const hfS4DContinuousConfig = `{
  "name":"s4d_continuous_export",
  "model_dim":8,
  "seq_len":6,
  "positional_embedding":"none",
  "dropout":0.1,
  "tie_dropout":true,
  "norm_type":"layernorm",
  "norm_placement":"post_residual",
  "final_norm":false,
  "input_adapter":{"kind":"linear_frames","feature_dim":1,"bias":true,"norm":"none"},
  "blocks":[{
    "type":"s4d","state_size":4,"n_ssm":2,"bidirectional":true,
    "discretization":"bilinear","trainable_b":true,"output_transform":"glu",
    "freq_scale":3,"sobolev_filter":{"beta_init":-0.25,"learning_rate":0.004}
  }],
  "training":{
    "objective":"classification",
    "classification":{"num_labels":3,"pooling":"mean","classifier_dropout":0},
    "steps":1,"batch_tokens":6,"seed":20260807
  }
}`

func TestExportHFS4DContinuousConfigAndWeightMap(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(hfS4DContinuousConfig), "s4d-continuous-hf")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if err := validateHFExportConfig(cfg); err != nil {
		t.Fatalf("validateHFExportConfig: %v", err)
	}
	blocks := hfBlockEntries(cfg, false)
	if len(blocks) != 1 {
		t.Fatalf("blocks=%d, want 1", len(blocks))
	}
	for key, want := range map[string]any{
		"type": "s4d", "state_size": 4, "n_ssm": 2,
		"bidirectional": true, "discretization": "bilinear",
		"trainable_b": true, "output_transform": "glu", "tie_dropout": true,
		"freq_scale": 3.0,
	} {
		if got := blocks[0][key]; got != want {
			t.Fatalf("block %s=%v, want %v", key, got, want)
		}
	}
	sobolev, ok := blocks[0]["sobolev_filter"].(map[string]any)
	if !ok || sobolev["beta_init"] != -0.25 || sobolev["learning_rate"] != 0.004 ||
		sobolev["trainable"] != true || sobolev["weight_decay"] != 0.0 || sobolev["granularity"] != "channel" {
		t.Fatalf("sobolev_filter=%#v", blocks[0]["sobolev_filter"])
	}

	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes: %v", err)
	}
	weights := make([][]float32, len(shapes))
	for i, shape := range shapes {
		weights[i] = make([]float32, shapeProduct(shape.Shape))
	}
	exportShapes, _, err := materializeHFExportWeights(cfg, cfg, shapes, weights)
	if err != nil {
		t.Fatalf("materializeHFExportWeights: %v", err)
	}
	mapping, err := buildHFWeightMap(cfg, exportShapes)
	if err != nil {
		t.Fatalf("buildHFWeightMap: %v", err)
	}
	byMixlab := make(map[string]string, len(mapping))
	for _, item := range mapping {
		underscore := strings.IndexByte(item.Mixlab, '_')
		if underscore >= 0 {
			byMixlab[item.Mixlab[underscore+1:]] = item.HF
		}
	}
	for name, want := range map[string]string{
		"input_adapter_proj":           "input_adapter.weight",
		"input_adapter_bias":           "input_adapter.bias",
		"s4d_log_dt":                   "blocks.0.log_dt",
		"s4d_B_real":                   "blocks.0.B_real",
		"s4d_C_backward_real":          "blocks.0.C_backward_real",
		"s4d_sobolev_beta":             "blocks.0.sobolev_beta",
		"s4d_out_proj":                 "blocks.0.out_proj.weight",
		"s4d_post_residual_norm_scale": "blocks.0.post_residual_norm.weight",
		"head_classifier_proj":         "classifier.weight",
	} {
		if got := byMixlab[name]; got != want {
			t.Fatalf("mapping %s=%q, want %q", name, got, want)
		}
	}
	for _, forbidden := range []string{"embed", "head", "final_norm_scale"} {
		if _, ok := byMixlab[forbidden]; ok {
			t.Fatalf("continuous final_norm=false export mapped forbidden weight %q", forbidden)
		}
	}
}

func TestRunExportHFS4DContinuousIsTokenizerFree(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, _ := writeHFExportFixture(t, dir, hfS4DContinuousConfig)
	outDir := filepath.Join(dir, "hf_out")
	if err := RunExportHF(ExportHFOptions{
		ConfigPath: cfgPath, SafetensorsLoad: weightsPath, OutputDir: outDir,
	}); err != nil {
		t.Fatalf("RunExportHF: %v", err)
	}
	var doc hfConfigJSON
	readJSON(t, filepath.Join(outDir, "config.json"), &doc)
	if doc.InputAdapter == nil || doc.InputAdapter.Kind != "linear_frames" || doc.InputAdapter.FeatureDim != 1 {
		t.Fatalf("input_adapter=%+v", doc.InputAdapter)
	}
	if doc.FinalNorm == nil || *doc.FinalNorm {
		t.Fatalf("final_norm=%v, want explicit false", doc.FinalNorm)
	}
	if _, ok := doc.AutoMap["AutoModelForCausalLM"]; ok {
		t.Fatal("continuous classifier unexpectedly advertises AutoModelForCausalLM")
	}
	if got := doc.AutoMap["AutoModelForSequenceClassification"]; got != "modeling_mixlab.MixlabForSequenceClassification" {
		t.Fatalf("sequence classifier auto_map=%q", got)
	}
	for _, name := range []string{"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"} {
		if _, err := os.Stat(filepath.Join(outDir, name)); !os.IsNotExist(err) {
			t.Fatalf("continuous export unexpectedly wrote %s", name)
		}
	}
	tensors, err := loadSafetensors(filepath.Join(outDir, "model.safetensors"))
	if err != nil {
		t.Fatalf("load exported tensors: %v", err)
	}
	for _, forbidden := range []string{"embed_tokens.weight", "lm_head_weight", "final_norm.weight"} {
		if _, ok := tensors[forbidden]; ok {
			t.Fatalf("continuous export contains %q", forbidden)
		}
	}
}

func TestRunExportHFS4DContinuousRejectsTokenOptions(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, hfS4DContinuousConfig)
	if err := RunExportHF(ExportHFOptions{
		ConfigPath: cfgPath, SafetensorsLoad: weightsPath,
		OutputDir: filepath.Join(dir, "hf_out"), TokenizerSource: tokenizerDir,
	}); err == nil || !strings.Contains(err.Error(), "does not use tokenizer") {
		t.Fatalf("error=%v, want continuous tokenizer-option rejection", err)
	}
}

func TestExportHFS4DContinuousValidationBoundary(t *testing.T) {
	tests := []struct {
		name    string
		config  string
		wantErr string
	}{
		{
			name:    "token s4d remains gated",
			config:  `{"model_dim":8,"vocab_size":11,"seq_len":4,"blocks":[{"type":"s4d","state_size":4}],"training":{"steps":1,"batch_tokens":4}}`,
			wantErr: "requires a linear_frames",
		},
		{
			name: "batchnorm remains gated",
			config: strings.Replace(
				strings.Replace(hfS4DContinuousConfig, `"norm_type":"layernorm"`, `"norm_type":"batchnorm"`, 1),
				`"norm_placement":"post_residual"`, `"norm_placement":"pre"`, 1,
			),
			wantErr: "BatchNorm running statistics",
		},
		{
			name: "mixed backbone remains gated",
			config: strings.Replace(
				strings.Replace(hfS4DContinuousConfig, `"blocks":[{`, `"blocks":[{"type":"mlp"},{`, 1),
				`"norm_placement":"post_residual"`, `"norm_placement":"pre"`, 1,
			),
			wantErr: "s4d-only stacks",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg, err := ParseArchConfig([]byte(tc.config), tc.name)
			if err != nil {
				t.Fatalf("ParseArchConfig: %v", err)
			}
			err = validateHFExportConfig(cfg)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error=%v, want containing %q", err, tc.wantErr)
			}
		})
	}
}

func TestExportHFS4DTemplateContainsReferenceMath(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("hf_templates", "s4d_mixlab.py"))
	if err != nil {
		t.Fatalf("read template: %v", err)
	}
	source := string(data)
	for _, want := range []string{
		"class MixlabS4DBlock", "def _discretize", "def _kernel",
		"torch.fft.rfft", "backward_kernel.flip(1)",
		"self.sobolev_beta", "self.sobolev_bounds", "torch.tanh(effective_beta)",
		"frequency_product = frequency_product * frequency_filter",
		"value * torch.sigmoid(gate)", "self.post_residual_norm(output)",
	} {
		if !strings.Contains(source, want) {
			t.Fatalf("S4D template missing %q", want)
		}
	}
}
