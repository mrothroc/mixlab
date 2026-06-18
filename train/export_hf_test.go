package train

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExportHFWeightMapAndDirectoryMetadata(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
		"name": "hf_tiny",
		"model_dim": 4,
		"vocab_size": 7,
		"seq_len": 3,
		"mlp_mult": 1.0,
		"blocks": [
			{"type": "plain", "heads": 2, "rope_dims": 2},
			{"type": "swiglu"}
		],
		"training": {"steps": 1, "batch_tokens": 3, "seed": 123}
	}`)
	outDir := filepath.Join(dir, "hf_out")
	if err := RunExportHF(ExportHFOptions{
		ConfigPath:      cfgPath,
		SafetensorsLoad: weightsPath,
		OutputDir:       outDir,
		TokenizerSource: tokenizerDir,
	}); err != nil {
		t.Fatalf("RunExportHF: %v", err)
	}

	for _, name := range []string{
		"config.json",
		"configuration_mixlab.py",
		"modeling_mixlab.py",
		"model.safetensors",
		"weight_map.json",
		"tokenizer.json",
		"tokenizer_config.json",
		"special_tokens_map.json",
	} {
		if _, err := os.Stat(filepath.Join(outDir, name)); err != nil {
			t.Fatalf("missing exported %s: %v", name, err)
		}
	}

	var cfg map[string]any
	readJSON(t, filepath.Join(outDir, "config.json"), &cfg)
	autoMap, ok := cfg["auto_map"].(map[string]any)
	if !ok {
		t.Fatalf("config.json missing auto_map: %#v", cfg["auto_map"])
	}
	if got := autoMap["AutoModelForCausalLM"]; got != "modeling_mixlab.MixlabForCausalLM" {
		t.Fatalf("AutoModelForCausalLM auto_map=%v", got)
	}
	if got := autoMap["AutoModel"]; got != "modeling_mixlab.MixlabModel" {
		t.Fatalf("AutoModel auto_map=%v", got)
	}
	if _, ok := autoMap["AutoModelForMaskedLM"]; ok {
		t.Fatalf("causal-only export unexpectedly registered AutoModelForMaskedLM: %#v", autoMap)
	}

	var mapping []hfWeightMapping
	readJSON(t, filepath.Join(outDir, "weight_map.json"), &mapping)
	if len(mapping) == 0 {
		t.Fatal("empty weight_map.json")
	}
	if mapping[0].Mixlab != "w0_embed" || mapping[0].HF != "embed_tokens.weight" {
		t.Fatalf("first weight mapping = %#v", mapping[0])
	}
	if !containsHFWeight(mapping, "blocks.0.wq.weight") || !containsHFWeight(mapping, "blocks.1.w_gate.weight") {
		t.Fatalf("weight map missing plain/swiglu entries: %#v", mapping)
	}

	tensors, err := loadSafetensors(filepath.Join(outDir, "model.safetensors"))
	if err != nil {
		t.Fatalf("load exported model.safetensors: %v", err)
	}
	meta := readSafetensorsMetadata(t, filepath.Join(outDir, "model.safetensors"))
	if got := meta["format"]; got != "pt" {
		t.Fatalf("model.safetensors __metadata__.format=%q, want pt", got)
	}
	if got := meta["name"]; got != "hf_tiny" {
		t.Fatalf("model.safetensors __metadata__.name=%q, want hf_tiny", got)
	}
	for _, name := range []string{"embed_tokens.weight", "lm_head_weight", "final_norm.weight", "blocks.0.wq.weight", "blocks.1.w_down.weight"} {
		if _, ok := tensors[name]; !ok {
			t.Fatalf("missing HF tensor %q", name)
		}
	}
}

func TestRunExportHFRequiresInputs(t *testing.T) {
	tests := []struct {
		name string
		opts ExportHFOptions
		want string
	}{
		{name: "config", opts: ExportHFOptions{}, want: "-config"},
		{name: "weights", opts: ExportHFOptions{ConfigPath: "config.json"}, want: "-safetensors-load"},
		{name: "output", opts: ExportHFOptions{ConfigPath: "config.json", SafetensorsLoad: "weights.safetensors"}, want: "-output"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := RunExportHF(tt.opts)
			if err == nil {
				t.Fatal("RunExportHF succeeded")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("RunExportHF error = %v, want %s", err, tt.want)
			}
		})
	}
}

func TestExportHFUnsupportedValidation(t *testing.T) {
	tests := []struct {
		name    string
		config  string
		wantErr string
	}{
		{
			name: "block type",
			config: `{
				"model_dim": 4, "vocab_size": 7, "seq_len": 3,
				"blocks": [{"type": "mamba"}],
				"training": {"steps": 1, "batch_tokens": 3}
			}`,
			wantErr: "blocks[0].type",
		},
		{
			name: "kv source",
			config: `{
				"model_dim": 4, "vocab_size": 7, "seq_len": 3,
				"blocks": [
					{"type": "plain", "heads": 2},
					{"type": "plain", "heads": 2, "kv_source": 1}
				],
				"training": {"steps": 1, "batch_tokens": 3}
			}`,
			wantErr: "blocks[1].kv_source",
		},
		{
			name: "hgrn2 remains gated advanced",
			config: `{
				"model_dim": 4, "vocab_size": 7, "seq_len": 3,
				"blocks": [{"type": "hgrn2", "heads": 2}],
				"training": {"steps": 1, "batch_tokens": 3}
			}`,
			wantErr: "blocks[0].type",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg, err := ParseArchConfig([]byte(tt.config), tt.name)
			if err != nil {
				t.Fatalf("ParseArchConfig: %v", err)
			}
			err = validateHFExportConfig(cfg)
			if err == nil {
				t.Fatal("validateHFExportConfig succeeded")
			}
			if !strings.Contains(err.Error(), "unsupported") || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("error = %v, want unsupported %s", err, tt.wantErr)
			}
		})
	}
}

func TestHFModelingPartialRoPEAdjacentPair(t *testing.T) {
	headDim := 6
	seqLen := 3
	input := []float64{
		0.10, 0.20, 0.30, 0.40, 0.50, 0.60,
		0.70, 0.80, 0.90, 1.00, 1.10, 1.20,
		1.30, 1.40, 1.50, 1.60, 1.70, 1.80,
	}
	got := hfAdjacentRoPEForTest(input, 0, 1, seqLen, headDim, 4)
	half := halfRotationRoPEForTest(input, seqLen, headDim, 4)
	if math.Abs(got[6]-half[6]) < 1e-6 && math.Abs(got[7]-half[7]) < 1e-6 {
		t.Fatalf("adjacent-pair RoPE unexpectedly matched half-rotation on nontrivial fixture")
	}
	for tPos := 0; tPos < seqLen; tPos++ {
		for d := 4; d < headDim; d++ {
			idx := tPos*headDim + d
			if got[idx] != input[idx] {
				t.Fatalf("partial RoPE changed pass-through dim token=%d dim=%d: got=%g want=%g", tPos, d, got[idx], input[idx])
			}
		}
	}

	modeling, err := os.ReadFile(filepath.Join("hf_templates", "modeling_mixlab.py"))
	if err != nil {
		t.Fatalf("read modeling template: %v", err)
	}
	source := string(modeling)
	for _, want := range []string{"rotate_adjacent_rope", "rotate_half_rope", "rope_convention", "rope_dims", "MixlabForMaskedLM", "MaskedLMOutput", "ignore_index=-100", "torch.stack((even * cos_t - odd * sin_t"} {
		if !strings.Contains(source, want) {
			t.Fatalf("modeling template missing %q", want)
		}
	}
}

func TestExportHFNativeHFParityCPUOracle(t *testing.T) {
	runExportHFParityCase(t, `{
		"name": "hf_parity",
		"model_dim": 4,
		"vocab_size": 7,
		"seq_len": 3,
		"mlp_mult": 1.0,
		"blocks": [
			{"type": "plain", "heads": 2, "rope_dims": 2},
			{"type": "swiglu"}
		],
		"training": {"steps": 1, "batch_tokens": 3, "seed": 99}
	}`, [][]int{{0, 1, 2}}, [][]int{{1, 2, 3}}, nil)
}

func TestExportHFPlainGatedFFNTailParityCPUOracle(t *testing.T) {
	for _, activation := range []string{"geglu", "swiglu"} {
		t.Run(activation, func(t *testing.T) {
			runExportHFParityCase(t, fmt.Sprintf(`{
				"name": "hf_plain_%[1]s_tail",
				"model_dim": 4,
				"vocab_size": 9,
				"seq_len": 4,
				"mlp_mult": 1.25,
				"blocks": [
					{"type": "plain", "heads": 2, "rope_dims": 2, "ffn_activation": "%[1]s"}
				],
				"training": {"steps": 1, "batch_tokens": 4, "seed": 102}
			}`, activation), [][]int{{0, 1, 2, 3}}, [][]int{{1, 2, 3, 4}}, func(t *testing.T, outDir string) {
				var cfg map[string]any
				readJSON(t, filepath.Join(outDir, "config.json"), &cfg)
				blocks, ok := cfg["blocks"].([]any)
				if !ok || len(blocks) != 1 {
					t.Fatalf("exported blocks = %#v", cfg["blocks"])
				}
				block, ok := blocks[0].(map[string]any)
				if !ok {
					t.Fatalf("exported block = %#v", blocks[0])
				}
				if got := block["ffn_activation"]; got != activation {
					t.Fatalf("exported ffn_activation=%v want %s", got, activation)
				}
				var mapping []hfWeightMapping
				readJSON(t, filepath.Join(outDir, "weight_map.json"), &mapping)
				if !containsHFWeight(mapping, "blocks.0.ff_gate.weight") {
					t.Fatalf("weight map missing ff_gate: %#v", mapping)
				}
			})
		})
	}
}

func TestExportHFHalfRotationRoPEParityCPUOracle(t *testing.T) {
	runExportHFParityCase(t, `{
		"name": "hf_half_rotation_rope",
		"model_dim": 8,
		"vocab_size": 11,
		"seq_len": 4,
		"mlp_mult": 1.0,
		"blocks": [
			{"type": "plain", "heads": 2, "rope_dims": 4, "rope_convention": "half_rotation"},
			{"type": "swiglu"}
		],
		"training": {"steps": 1, "batch_tokens": 4, "seed": 100}
	}`, [][]int{{0, 1, 2, 3}}, [][]int{{1, 2, 3, 4}}, nil)
}

func TestExportHFParityGEGLUAndMLPCPUOracle(t *testing.T) {
	runExportHFParityCase(t, `{
		"name": "hf_geglu_mlp",
		"model_dim": 4,
		"vocab_size": 9,
		"seq_len": 4,
		"mlp_mult": 1.25,
		"blocks": [
			{"type": "plain", "heads": 2, "rope_dims": 2},
			{"type": "geglu"},
			{"type": "mlp", "activation": "leaky_relu_sq", "leaky_slope": 0.25}
		],
		"training": {"steps": 1, "batch_tokens": 4, "seed": 101}
	}`, [][]int{{0, 1, 2, 3}}, [][]int{{1, 2, 3, 4}}, nil)
}

func TestExportHFAttentionMaskGQAQKGainWindowParityCPUOracle(t *testing.T) {
	runExportHFParityCase(t, `{
		"name": "hf_attention_core",
		"model_dim": 8,
		"vocab_size": 11,
		"seq_len": 4,
		"mlp_mult": 1.0,
		"blocks": [
			{"type": "plain", "heads": 4, "kv_heads": 2, "qk_gain": 1.5, "rope_dims": 2, "window_size": 2},
			{"type": "plain", "heads": 4, "attention_mask": "bidirectional", "rope_dims": 2},
			{"type": "plain", "heads": 4, "attention_mask": "none", "rope_dims": 2}
		],
		"training": {"steps": 1, "batch_tokens": 4, "seed": 202}
	}`, [][]int{{0, 1, 2, 3}}, [][]int{{1, 2, 3, 4}}, nil)
}

func TestExportHFQKNormParityCPUOracle(t *testing.T) {
	runExportHFParityCase(t, `{
		"name": "hf_qk_norm",
		"model_dim": 8,
		"vocab_size": 11,
		"seq_len": 4,
		"mlp_mult": 1.0,
		"blocks": [
			{"type": "plain", "heads": 4, "kv_heads": 2, "qk_norm": true, "qk_gain": 1.25, "rope_dims": 2},
			{"type": "plain", "heads": 4, "attention_mask": "bidirectional", "qk_norm": true, "relative_attention": "deberta_p2c_c2p", "relative_attention_window": 2}
		],
		"training": {"steps": 1, "batch_tokens": 4, "seed": 207}
	}`, [][]int{{0, 1, 2, 3}}, [][]int{{1, 2, 3, 4}}, nil)
}

func TestExportHFDebertaRelativeAttentionParityCPUOracle(t *testing.T) {
	runExportHFParityCase(t, `{
		"name": "hf_deberta_relative",
		"model_dim": 8,
		"vocab_size": 11,
		"seq_len": 4,
		"mlp_mult": 1.0,
		"blocks": [
			{"type": "plain", "heads": 4, "attention_mask": "bidirectional", "relative_attention": "deberta_p2c_c2p", "relative_attention_window": 2, "qk_gain": 1.25}
		],
		"training": {"steps": 1, "batch_tokens": 4, "seed": 212}
	}`, [][]int{{0, 1, 2, 3}}, [][]int{{1, 2, 3, 4}}, nil)
}

func TestExportHFXSASparseGateParityCPUOracle(t *testing.T) {
	runExportHFParityCase(t, `{
		"name": "hf_xsa_sparse_gate",
		"model_dim": 8,
		"vocab_size": 11,
		"seq_len": 4,
		"mlp_mult": 1.0,
		"blocks": [
			{"type": "plain", "heads": 2, "rope_dims": 4, "xsa": true, "sparse_attn_gate": true, "qk_gain": 1.2}
		],
		"training": {"steps": 1, "batch_tokens": 4, "seed": 214}
	}`, [][]int{{0, 1, 2, 3}}, [][]int{{1, 2, 3, 4}}, nil)
}

func TestExportHFMoETopKRoutingParityCPUOracle(t *testing.T) {
	for _, topK := range []int{1, 2} {
		t.Run(fmt.Sprintf("top_k_%d", topK), func(t *testing.T) {
			runExportHFParityCase(t, fmt.Sprintf(`{
				"name": "hf_moe_topk_%d",
				"model_dim": 6,
				"vocab_size": 13,
				"seq_len": 4,
				"mlp_mult": 1.0,
				"blocks": [
					{"type": "moe", "num_experts": 3, "top_k": %d, "expert_block": {"type": "swiglu"}}
				],
				"training": {"steps": 1, "batch_tokens": 4, "seed": 313}
			}`, topK, topK), [][]int{{0, 1, 2, 3}}, [][]int{{1, 2, 3, 4}}, nil)
		})
	}
}

func TestExportHFMoEExpertFFNParityCPUOracle(t *testing.T) {
	cases := []struct {
		name   string
		expert string
	}{
		{name: "geglu", expert: `{"type": "geglu"}`},
		{name: "mlp", expert: `{"type": "mlp", "activation": "leaky_relu_sq", "leaky_slope": 0.2}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			runExportHFParityCase(t, fmt.Sprintf(`{
				"name": "hf_moe_%s",
				"model_dim": 6,
				"vocab_size": 13,
				"seq_len": 4,
				"mlp_mult": 1.0,
				"blocks": [
					{"type": "moe", "num_experts": 2, "top_k": 2, "expert_block": %s}
				],
				"training": {"steps": 1, "batch_tokens": 4, "seed": 323}
			}`, tc.name, tc.expert), [][]int{{0, 1, 2, 3}}, [][]int{{1, 2, 3, 4}}, nil)
		})
	}
}

func TestExportHFAdvancedParityCPUOracle(t *testing.T) {
	runExportHFParityCase(t, `{
		"name": "hf_advanced_parity",
		"model_dim": 8,
		"vocab_size": 17,
		"seq_len": 4,
		"mlp_mult": 1.0,
		"blocks": [
			{"type": "plain", "heads": 4, "relative_attention": "deberta_p2c_c2p", "relative_attention_window": 2},
			{"type": "moe", "num_experts": 2, "top_k": 2, "expert_block": {"type": "geglu"}}
		],
		"training": {"steps": 1, "batch_tokens": 4, "seed": 329}
	}`, [][]int{{0, 1, 2, 3}}, [][]int{{1, 2, 3, 4}}, nil)
}

func TestExportHFHybridObjectiveExportsCausalEval(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
		"name": "hf_hybrid_eval",
		"model_dim": 4,
		"vocab_size": 9,
		"seq_len": 3,
		"mlp_mult": 1.0,
		"blocks": [{"type": "plain", "heads": 2, "attention_mask": "bidirectional"}],
		"training": {
			"steps": 1,
			"batch_tokens": 3,
			"seed": 333,
			"objective": "hybrid",
			"mlm_mask_token_id": 1,
			"hybrid_secondary_objective": "mlm"
		}
	}`)
	outDir := filepath.Join(dir, "hf_out")
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
	blocks := cfg["blocks"].([]any)
	block := blocks[0].(map[string]any)
	if got := block["attention_mask"]; got != "causal" {
		t.Fatalf("exported hybrid attention_mask=%v, want causal", got)
	}
}

func TestExportHFAdvancedUnsupportedPolicies(t *testing.T) {
	tests := []struct {
		name    string
		block   string
		wantErr string
	}{
		{name: "hgrn2", block: `{"type": "hgrn2", "heads": 2}`, wantErr: "hgrn2"},
		{name: "mlstm", block: `{"type": "mlstm", "heads": 2, "d_k": 2, "d_v": 3}`, wantErr: "mlstm"},
		{name: "gated_deltanet", block: `{"type": "gated_deltanet", "heads": 2, "d_k": 2}`, wantErr: "gated_deltanet"},
		{name: "mamba", block: `{"type": "mamba"}`, wantErr: "mamba"},
		{name: "mamba3_canonical", block: `{"type": "mamba3-canonical", "inner_dim": 8, "state_size": 4, "n_groups": 2}`, wantErr: "mamba3-canonical"},
		{name: "retnet", block: `{"type": "retnet", "heads": 2}`, wantErr: "retnet"},
		{name: "custom", block: `{"type": "custom", "weights": [], "ops": []}`, wantErr: "custom"},
		{name: "unsupported_moe_expert", block: `{"type": "moe", "num_experts": 2, "expert_block": {"type": "plain", "heads": 2}}`, wantErr: "expert_block.type"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg, err := ParseArchConfig([]byte(fmt.Sprintf(`{
				"model_dim": 4,
				"vocab_size": 9,
				"seq_len": 3,
				"mlp_mult": 1.0,
				"blocks": [%s],
				"training": {"steps": 1, "batch_tokens": 3}
			}`, tt.block)), tt.name)
			if err != nil {
				if strings.Contains(err.Error(), tt.wantErr) {
					return
				}
				t.Fatalf("ParseArchConfig: %v", err)
			}
			err = validateHFExportConfig(cfg)
			if err == nil {
				t.Fatal("validateHFExportConfig succeeded")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("error = %v, want %s", err, tt.wantErr)
			}
		})
	}
}

func TestExportHFHGRN2UnsupportedPolicy(t *testing.T) {
	assertHFExportUnsupportedBlock(t, "hgrn2", `{"type": "hgrn2", "heads": 2}`, "hgrn2")
}

func TestExportHFMLSTMUnsupportedPolicy(t *testing.T) {
	assertHFExportUnsupportedBlock(t, "mlstm", `{"type": "mlstm", "heads": 2, "d_k": 2, "d_v": 3}`, "mlstm")
}

func TestExportHFRecurrentPolicy(t *testing.T) {
	for _, tc := range []struct {
		name  string
		block string
		want  string
	}{
		{name: "gated_deltanet", block: `{"type": "gated_deltanet", "heads": 2, "d_k": 2}`, want: "gated_deltanet"},
		{name: "mamba", block: `{"type": "mamba"}`, want: "mamba"},
		{name: "mamba3_canonical", block: `{"type": "mamba3-canonical", "inner_dim": 8, "state_size": 4, "n_groups": 2}`, want: "mamba3-canonical"},
		{name: "retnet", block: `{"type": "retnet", "heads": 2}`, want: "retnet"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			assertHFExportUnsupportedBlock(t, tc.name, tc.block, tc.want)
		})
	}
}

func assertHFExportUnsupportedBlock(t *testing.T, name, block, want string) {
	t.Helper()
	cfg, err := ParseArchConfig([]byte(fmt.Sprintf(`{
		"model_dim": 4,
		"vocab_size": 9,
		"seq_len": 3,
		"mlp_mult": 1.0,
		"blocks": [%s],
		"training": {"steps": 1, "batch_tokens": 3}
	}`, block)), name)
	if err != nil {
		if strings.Contains(err.Error(), want) {
			return
		}
		t.Fatalf("ParseArchConfig: %v", err)
	}
	err = validateHFExportConfig(cfg)
	if err == nil {
		t.Fatal("validateHFExportConfig succeeded")
	}
	if !strings.Contains(err.Error(), want) {
		t.Fatalf("error = %v, want %s", err, want)
	}
}

func TestExportHFPartialAdvancedRegistryAndSupportMatrix(t *testing.T) {
	capabilities := hfExportCapabilities()
	if len(capabilities) == 0 {
		t.Fatal("empty HF export capability registry")
	}
	want := map[string]hfExportSupportStatus{
		"plain.attn_bias":                                           hfExportSupported,
		"plain.attn_value_gate":                                     hfExportSupported,
		"plain.attn_post_norm":                                      hfExportSupported,
		"plain.qk_norm":                                             hfExportSupported,
		"plain.ffn_activation=geglu":                                hfExportSupported,
		"plain.ffn_activation=swiglu":                               hfExportSupported,
		"plain.relative_attention=deberta_p2c_c2p":                  hfExportSupported,
		"plain.relative_attention_parameterization=shared_qk_reuse": hfExportSupported,
		"plain.relative_attention_embedding_norm=layernorm":         hfExportSupported,
		"moe":            hfExportSupported,
		"hgrn2":          hfExportGated,
		"mlstm":          hfExportGated,
		"gated_deltanet": hfExportGated,
	}
	for feature, status := range want {
		got := capabilityByFeature(feature)
		if got.Status != status {
			t.Fatalf("capability %s status=%s want %s", feature, got.Status, status)
		}
		if strings.TrimSpace(got.Reason) == "" {
			t.Fatalf("capability %s missing reason", feature)
		}
	}
	doc, err := os.ReadFile(filepath.Join("..", "docs", "hf-export-support-matrix.md"))
	if err != nil {
		t.Fatalf("read support matrix: %v", err)
	}
	source := string(doc)
	for feature := range want {
		if !strings.Contains(source, feature) {
			t.Fatalf("support matrix missing %s", feature)
		}
	}
}

func TestExportHFExamplesDocumentation(t *testing.T) {
	for _, path := range []string{
		filepath.Join("..", "docs", "hf-export.md"),
		filepath.Join("..", "docs", "hf-export-support-matrix.md"),
		filepath.Join("..", "examples", "README.md"),
	} {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v", path, err)
		}
		source := string(data)
		if strings.Contains(source, "TODO") || strings.Contains(source, "TBD") || strings.Contains(source, "placeholder") {
			t.Fatalf("%s contains unfinished support-matrix wording", path)
		}
	}
}

func TestExportHFBigramTrigramCharParityCPUOracle(t *testing.T) {
	runExportHFParityCase(t, `{
		"name": "hf_features",
		"model_dim": 6,
		"vocab_size": 8,
		"seq_len": 4,
		"mlp_mult": 1.0,
		"char_vocab_size": 257,
		"char_dim": 3,
		"char_max_per_token": 3,
		"bigram_vocab_size": 17,
		"bigram_dim": 4,
		"trigram_vocab_size": 19,
		"trigram_dim": 5,
		"blocks": [
			{"type": "plain", "heads": 3, "rope_dims": 2},
			{"type": "swiglu"}
		],
		"training": {"steps": 1, "batch_tokens": 4, "seed": 303}
	}`, [][]int{{0, 1, 2, 3}}, [][]int{{1, 2, 3, 4}}, func(t *testing.T, outDir string) {
		if _, err := os.Stat(filepath.Join(outDir, "char_features.bin")); err != nil {
			t.Fatalf("missing exported char_features.bin: %v", err)
		}
	})
}

func TestNativeHFTrainedParityCPUOracle(t *testing.T) {
	runExportHFParityCase(t, `{
		"name": "hf_trained_parity",
		"model_dim": 4,
		"vocab_size": 7,
		"seq_len": 3,
		"mlp_mult": 1.0,
		"blocks": [
			{"type": "plain", "heads": 2, "rope_dims": 2},
			{"type": "geglu"}
		],
		"training": {"steps": 1, "batch_tokens": 3, "seed": 404}
	}`, [][]int{{0, 1, 2}}, [][]int{{1, 2, 3}}, nil, scaleHFExportWeightsToTrainedMagnitude)
}

func TestNativeHFDWALayerAggregationParityCPUOracle(t *testing.T) {
	runExportHFParityCase(t, `{
		"name": "hf_dwa_parity",
		"model_dim": 4,
		"vocab_size": 7,
		"seq_len": 3,
		"mlp_mult": 1.0,
		"layer_aggregation": "dwa",
		"blocks": [
			{"type": "plain", "heads": 2, "rope_dims": 2},
			{"type": "geglu"}
		],
		"training": {"steps": 1, "batch_tokens": 3, "seed": 606}
	}`, [][]int{{0, 1, 2}}, [][]int{{1, 2, 3}}, func(t *testing.T, outDir string) {
		var cfg hfConfigJSON
		readJSON(t, filepath.Join(outDir, "config.json"), &cfg)
		if cfg.LayerAggregation != "dwa" {
			t.Fatalf("config layer_aggregation=%q, want dwa", cfg.LayerAggregation)
		}
		var mapping []hfWeightMapping
		readJSON(t, filepath.Join(outDir, "weight_map.json"), &mapping)
		for _, name := range []string{"dwa_alphas.0", "dwa_alphas.1", "dwa_alphas.2"} {
			if !containsHFWeight(mapping, name) {
				t.Fatalf("weight_map missing %s", name)
			}
		}
	})
}

func TestScaleHFExportWeightsToTrainedMagnitude(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"name": "hf_trained_magnitude_fixture",
		"model_dim": 8,
		"vocab_size": 11,
		"seq_len": 4,
		"mlp_mult": 1.0,
		"blocks": [
			{"type": "plain", "heads": 2, "qk_norm": true, "rope_dims": 2},
			{"type": "geglu"}
		],
		"training": {"steps": 1, "batch_tokens": 4, "seed": 505, "weight_init": "normal", "weight_init_std": 0.02}
	}`), "hf_trained_magnitude_fixture")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes: %v", err)
	}
	weights := initWeightData(shapes, cfg.Training.Seed, cfg.Training.WeightInit, cfg.Training.WeightInitStd)
	before := make([]float64, len(weights))
	for i := range weights {
		before[i] = weightRMS(weights[i])
	}
	if err := scaleHFExportWeightsToTrainedMagnitude(weights, shapes); err != nil {
		t.Fatalf("scaleHFExportWeightsToTrainedMagnitude: %v", err)
	}
	checked := 0
	for i, ws := range shapes {
		if !isTrainedMagnitudeCandidate(ws) {
			continue
		}
		after := weightRMS(weights[i])
		if after < trainedFixtureMinRMS {
			t.Fatalf("%s RMS=%g, want >= %g", ws.Name, after, trainedFixtureMinRMS)
		}
		if before[i] > 0 && after < before[i]*trainedFixtureMinScaleRatio*0.99 {
			t.Fatalf("%s RMS before=%g after=%g, want trained-scale increase", ws.Name, before[i], after)
		}
		checked++
	}
	if checked == 0 {
		t.Fatal("no trained-magnitude tensors checked")
	}
}

type hfExportWeightMutator func([][]float32, []WeightShape) error

func runExportHFParityCase(t *testing.T, config string, tokens, targets [][]int, afterExport func(*testing.T, string), mutators ...hfExportWeightMutator) {
	t.Helper()
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixtureWithMutators(t, dir, config, mutators...)
	cfg, err := LoadArchConfig(cfgPath)
	if err != nil {
		t.Fatalf("LoadArchConfig: %v", err)
	}
	if err := configureCharFeaturesForHFExport(cfg, cfgPath, weightsPath, tokenizerDir); err != nil {
		t.Fatalf("configureCharFeaturesForHFExport: %v", err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes: %v", err)
	}
	nativeWeights, err := loadSafetensorsWeights(weightsPath, shapes)
	if err != nil {
		t.Fatalf("load native weights: %v", err)
	}

	outDir := filepath.Join(dir, "hf_out")
	if err := RunExportHF(ExportHFOptions{ConfigPath: cfgPath, SafetensorsLoad: weightsPath, OutputDir: outDir, TokenizerSource: tokenizerDir}); err != nil {
		t.Fatalf("RunExportHF: %v", err)
	}
	if afterExport != nil {
		afterExport(t, outDir)
	}
	hfWeights, err := loadHFWeightsForParity(filepath.Join(outDir, "model.safetensors"))
	if err != nil {
		t.Fatalf("loadHFWeightsForParity: %v", err)
	}

	nativeLogits := runNativeCPUForward(t, cfg, nativeWeights, tokens)
	hfLogits := runHFCPUForward(t, cfg, hfWeights, tokens)
	maxDiff := maxAbsDiff3D(nativeLogits, hfLogits)
	nativeLoss := cpuCrossEntropy(nativeLogits, targets)
	hfLoss := cpuCrossEntropy(hfLogits, targets)
	if math.Abs(nativeLoss-hfLoss) >= 1e-4 {
		t.Fatalf("mean loss delta = %.8f, want < 1e-4", math.Abs(nativeLoss-hfLoss))
	}
	if maxDiff >= 1e-3 {
		t.Fatalf("max per-logit abs diff = %.8f, want < 1e-3", maxDiff)
	}
}

func writeHFExportFixture(t *testing.T, dir, config string) (configPath, weightsPath, tokenizerDir string) {
	t.Helper()
	return writeHFExportFixtureWithMutators(t, dir, config)
}

func writeHFExportFixtureWithMutators(t *testing.T, dir, config string, mutators ...hfExportWeightMutator) (configPath, weightsPath, tokenizerDir string) {
	t.Helper()
	configPath = filepath.Join(dir, "config.json")
	if err := os.WriteFile(configPath, []byte(config), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	cfg, err := LoadArchConfig(configPath)
	if err != nil {
		t.Fatalf("LoadArchConfig: %v", err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatalf("computeWeightShapes: %v", err)
	}
	weights := initWeightData(shapes, cfg.Training.Seed, cfg.Training.WeightInit, cfg.Training.WeightInitStd)
	for _, mutate := range mutators {
		if mutate != nil {
			if err := mutate(weights, shapes); err != nil {
				t.Fatalf("mutate HF export weights: %v", err)
			}
		}
	}
	weightsPath = filepath.Join(dir, "weights.safetensors")
	if err := exportSafetensors(weightsPath, cfg, shapes, weights); err != nil {
		t.Fatalf("exportSafetensors: %v", err)
	}
	if cfg.CharVocabSize > 0 {
		writeSyntheticCharFeaturesForExportTest(t, filepath.Join(dir, "char_features.bin"), cfg)
	}

	tokenizerDir = filepath.Join(dir, "tokenizer")
	if err := os.MkdirAll(tokenizerDir, 0o755); err != nil {
		t.Fatal(err)
	}
	tokenizerJSON := `{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<|pad|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "<|unk|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 2, "content": "<|eos|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 3, "content": "<|bos|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": "<|unk|>",
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "byte_fallback": false,
    "vocab": {
      "<|pad|>": 0,
      "<|unk|>": 1,
      "<|eos|>": 2,
      "<|bos|>": 3,
      "a": 4,
      "b": 5,
      "c": 6
    },
    "merges": []
  }
}`
	if err := os.WriteFile(filepath.Join(tokenizerDir, "tokenizer.json"), []byte(tokenizerJSON), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tokenizerDir, "tokenizer_config.json"), []byte(`{"tokenizer_class":"PreTrainedTokenizerFast","model_max_length":3}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tokenizerDir, "special_tokens_map.json"), []byte(`{"bos_token":"<s>"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	return configPath, weightsPath, tokenizerDir
}

const (
	trainedFixtureMinRMS        = 0.15
	trainedFixtureMinScaleRatio = 1.50
)

func writeSyntheticCharFeaturesForExportTest(t *testing.T, path string, cfg *ArchConfig) {
	t.Helper()
	rows := make([]uint16, cfg.VocabSize*cfg.EffectiveCharMaxPerToken())
	for tok := 0; tok < cfg.VocabSize; tok++ {
		base := tok * cfg.EffectiveCharMaxPerToken()
		rows[base] = uint16(tok%256 + 1)
		if cfg.EffectiveCharMaxPerToken() > 1 && tok%2 == 1 {
			rows[base+1] = uint16((tok*3)%256 + 1)
		}
	}
	writeTestCharFeatures(t, path, cfg.VocabSize, cfg.CharVocabSize, cfg.EffectiveCharMaxPerToken(), rows)
}

func readJSON(t *testing.T, path string, v any) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if err := json.Unmarshal(data, v); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
}

func readSafetensorsMetadata(t *testing.T, path string) map[string]string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if len(data) < 8 {
		t.Fatalf("safetensors file %s is too short", path)
	}
	headerLen := binary.LittleEndian.Uint64(data[:8])
	if headerLen == 0 || uint64(len(data)-8) < headerLen {
		t.Fatalf("safetensors file %s has invalid header length %d", path, headerLen)
	}
	var header map[string]json.RawMessage
	if err := json.Unmarshal(data[8:8+headerLen], &header); err != nil {
		t.Fatalf("parse safetensors header %s: %v", path, err)
	}
	raw, ok := header["__metadata__"]
	if !ok {
		t.Fatalf("safetensors file %s missing __metadata__", path)
	}
	var meta map[string]string
	if err := json.Unmarshal(raw, &meta); err != nil {
		t.Fatalf("parse safetensors metadata %s: %v", path, err)
	}
	return meta
}

func containsHFWeight(mapping []hfWeightMapping, name string) bool {
	for _, m := range mapping {
		if m.HF == name {
			return true
		}
	}
	return false
}

func loadHFWeightsForParity(path string) (map[string][]float64, error) {
	tensors, err := loadSafetensors(path)
	if err != nil {
		return nil, err
	}
	out := make(map[string][]float64, len(tensors))
	for name, tensor := range tensors {
		data, err := decodeSafetensorFloat32(name, tensor.Shape, tensors)
		if err != nil {
			return nil, err
		}
		out[name] = toFloat64(data)
	}
	return out, nil
}
