package train

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestExportHFMamba3TemplateContainsCanonicalScan(t *testing.T) {
	sourceBytes, err := os.ReadFile(filepath.Join("hf_templates", "mamba3_mixlab.py"))
	if err != nil {
		t.Fatalf("read Mamba-3 HF template: %v", err)
	}
	source := string(sourceBytes)
	for _, want := range []string{
		"class MixlabMamba3CanonicalBlock",
		"delta = F.softplus(dt_raw)",
		"interpolation = torch.sigmoid(lambda_raw)",
		"candidate_phase = phase + delta_step.unsqueeze(-1) * theta[:, step]",
		"previous = beta * previous_b * previous_x",
		"candidate_state = alpha * state + previous + current",
		"gated = self.post_norm(scanned) * F.silu(self.w_gate(x_norm))",
	} {
		if !strings.Contains(source, want) {
			t.Fatalf("Mamba-3 HF template missing %q", want)
		}
	}
}

func TestExportHFMamba3ReferenceFixture(t *testing.T) {
	if os.Getenv("HF_PARITY") != "1" {
		t.Skip("set HF_PARITY=1 to run the PyTorch Mamba-3 reference fixture")
	}
	python := os.Getenv("HF_PARITY_PYTHON")
	if python == "" {
		python = "python3"
	}
	if err := exec.Command(python, "-c", "import torch").Run(); err != nil {
		t.Skipf("PyTorch unavailable via %q: %v", python, err)
	}
	script := filepath.Join("testdata", "hf_mamba3_reference.py")
	template := filepath.Join("hf_templates", "mamba3_mixlab.py")
	fixture := filepath.Join("testdata", "mamba3_full_block_reference.json")
	output, err := exec.Command(
		python, script, "--template", template, "--fixture", fixture,
	).CombinedOutput()
	t.Logf("hf_mamba3_reference.py output:\n%s", output)
	if err != nil {
		t.Fatalf("Mamba-3 HF reference parity failed: %v", err)
	}
}

func TestExportHFMamba3ConfigAndWeightMap(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"model_dim": 8,
		"vocab_size": 13,
		"seq_len": 6,
		"mlp_mult": 2.0,
		"tie_embeddings": true,
		"blocks": [
			{"type": "mamba3-canonical", "inner_dim": 8, "state_size": 4, "n_groups": 2, "dt_rank": 2, "conv_kernel": 3},
			{"type": "swiglu"}
		],
		"training": {"steps": 1, "batch_tokens": 6}
	}`), "mamba3_hf")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if err := validateHFExportConfig(cfg); err != nil {
		t.Fatalf("validateHFExportConfig: %v", err)
	}

	blocks := hfBlockEntries(cfg, false)
	if len(blocks) != 2 {
		t.Fatalf("block entries=%d, want 2", len(blocks))
	}
	for key, want := range map[string]any{
		"type":        "mamba3-canonical",
		"inner_dim":   8,
		"state_size":  4,
		"n_groups":    2,
		"dt_rank":     2,
		"conv_kernel": 3,
		"use_conv":    true,
	} {
		if got := blocks[0][key]; got != want {
			t.Fatalf("mamba3 block %s=%v, want %v", key, got, want)
		}
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
	wantMappings := map[string]string{
		"pre_norm_scale": "blocks.0.pre_norm.weight",
		"conv_w":         "blocks.0.conv_weight",
		"w_theta_high":   "blocks.0.w_theta_high.weight",
		"B_norm_scale":   "blocks.0.B_norm.weight",
		"A_log":          "blocks.0.A_log",
		"w_out":          "blocks.0.w_out.weight",
	}
	for suffix, wantHF := range wantMappings {
		found := false
		for _, item := range mapping {
			separator := strings.IndexByte(item.Mixlab, '_')
			if separator >= 0 && item.Mixlab[separator+1:] == suffix {
				found = true
				if item.HF != wantHF {
					t.Fatalf("mapping %s=%s, want %s", item.Mixlab, item.HF, wantHF)
				}
			}
		}
		if !found {
			t.Fatalf("missing mapping for Mamba-3 weight %s", suffix)
		}
	}
}

func TestExportHFMamba3RejectsConfigurableOuterNormUntilParityCoverage(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"model_dim":8,"vocab_size":13,"seq_len":6,
		"norm_type":"layernorm","norm_placement":"pre",
		"blocks":[{"type":"mamba3-canonical","inner_dim":8,"state_size":4,"n_groups":2,"dt_rank":2,"use_conv":false}],
		"training":{"objective":"classification","classification":{"num_labels":2},"batch_tokens":6}
	}`), "mamba3_layernorm_hf")
	if err != nil {
		t.Fatal(err)
	}
	err = validateHFExportConfig(cfg)
	if err == nil || !strings.Contains(err.Error(), "configurable outer norms are native-only") {
		t.Fatalf("validateHFExportConfig error=%v", err)
	}
}

func TestExportHFMamba3WithoutShortConv(t *testing.T) {
	useConv := false
	cfg := &ArchConfig{
		ModelDim:  8,
		VocabSize: 13,
		SeqLen:    4,
		Blocks: []BlockSpec{{
			Type: "mamba3-canonical", InnerDim: 8, StateSize: 4, NGroups: 2, DTRank: 2, UseConv: &useConv,
		}},
		Training: TrainingSpec{Steps: 1, BatchTokens: 4},
	}
	parsed := parseHFTestConfig(t, cfg)
	if err := validateHFExportConfig(parsed); err != nil {
		t.Fatalf("validateHFExportConfig: %v", err)
	}
	blocks := hfBlockEntries(parsed, false)
	if got := blocks[0]["use_conv"]; got != false {
		t.Fatalf("use_conv=%v, want false", got)
	}
	if _, ok := blocks[0]["conv_kernel"]; ok {
		t.Fatal("no-conv Mamba-3 config unexpectedly exports conv_kernel")
	}
}

func TestExportHFBidirectionalMamba3IsExplicitlyNativeOnly(t *testing.T) {
	useConv := false
	cfg := &ArchConfig{
		ModelDim: 8, VocabSize: 13, SeqLen: 4,
		Blocks: []BlockSpec{{
			Type: "mamba3-canonical", InnerDim: 8, StateSize: 4, NGroups: 2, DTRank: 2,
			UseConv: &useConv, Bidirectional: true,
		}},
		Training: TrainingSpec{
			Objective: arch.ObjectiveClassification, Classification: &arch.ClassificationSpec{NumLabels: 2}, Steps: 1, BatchTokens: 4,
		},
	}
	parsed := parseHFTestConfig(t, cfg)
	err := validateHFExportConfig(parsed)
	if err == nil || !strings.Contains(err.Error(), "valid-prefix reversal") {
		t.Fatalf("validateHFExportConfig error=%v want valid-prefix reversal parity error", err)
	}
}

func TestExportHFMamba3KeepsOtherRecurrentMixersGated(t *testing.T) {
	for _, blockType := range []string{
		"legacy_mamba", "gated_linear_ssm", "mamba3", "rwkv", "retnet",
		"hgrn2", "mlstm", "gated_deltanet",
	} {
		t.Run(blockType, func(t *testing.T) {
			cfg := &ArchConfig{
				ModelDim: 8, VocabSize: 13, SeqLen: 4,
				Blocks:   []BlockSpec{{Type: blockType, Heads: 2, DK: 4, DV: 4}},
				Training: TrainingSpec{Steps: 1, BatchTokens: 4},
			}
			parsed := parseHFTestConfig(t, cfg)
			if err := validateHFExportConfig(parsed); err == nil {
				t.Fatalf("%s unexpectedly became export-supported", blockType)
			}
		})
	}
}

func TestExportHFNativeMamba3ClassifierMetadataAndWeights(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixtureWithMutators(t, dir, `{
		"name": "mamba3_native_classifier",
		"model_dim": 8,
		"vocab_size": 13,
		"seq_len": 6,
		"mlp_mult": 2.0,
		"tie_embeddings": true,
		"blocks": [
			{"type": "mamba3-canonical", "inner_dim": 8, "state_size": 4, "n_groups": 2, "dt_rank": 2, "conv_kernel": 3},
			{"type": "swiglu"}
		],
		"training": {
			"objective": "classification",
			"classification": {"num_labels": 3, "pooling": "mean", "classifier_dropout": 0.0},
			"steps": 1,
			"batch_tokens": 6,
			"seed": 123
		}
	}`, func(weights [][]float32, shapes []WeightShape) error {
		proj := weightShapeIndex(shapes, "head_classifier_proj")
		bias := weightShapeIndex(shapes, "head_classifier_bias")
		for i := range weights[proj] {
			weights[proj][i] = float32(i+1) / 100
		}
		copy(weights[bias], []float32{0.1, -0.2, 0.3})
		return nil
	})

	outDir := filepath.Join(dir, "hf_out")
	if err := RunExportHF(ExportHFOptions{
		ConfigPath: cfgPath, SafetensorsLoad: weightsPath, OutputDir: outDir, TokenizerSource: tokenizerDir,
	}); err != nil {
		t.Fatalf("RunExportHF: %v", err)
	}
	var doc hfConfigJSON
	readJSON(t, filepath.Join(outDir, "config.json"), &doc)
	if len(doc.Architectures) != 1 || doc.Architectures[0] != "MixlabForSequenceClassification" {
		t.Fatalf("architectures=%v", doc.Architectures)
	}
	if doc.NumLabels != 3 || doc.SequenceClassificationPooling != "mean" {
		t.Fatalf("classifier metadata num_labels=%d pooling=%q", doc.NumLabels, doc.SequenceClassificationPooling)
	}
	if doc.ClassifierDropout == nil || *doc.ClassifierDropout != 0 {
		t.Fatalf("classifier_dropout=%v, want explicit zero", doc.ClassifierDropout)
	}

	tensors, err := loadSafetensors(filepath.Join(outDir, "model.safetensors"))
	if err != nil {
		t.Fatalf("load exported tensors: %v", err)
	}
	proj := tensors["classifier.weight"]
	bias := tensors["classifier.bias"]
	if !equalIntShape(proj.Shape, []int{3, 8}) || !equalIntShape(bias.Shape, []int{3}) {
		t.Fatalf("classifier shapes weight=%v bias=%v", proj.Shape, bias.Shape)
	}
	values := decodeF32Blob(t, proj.Data)
	for label := 0; label < 3; label++ {
		for dim := 0; dim < 8; dim++ {
			want := float32(dim*3+label+1) / 100
			if got := values[label*8+dim]; got != want {
				t.Fatalf("classifier[%d,%d]=%g, want %g", label, dim, got, want)
			}
		}
	}
}

func decodeF32Blob(t *testing.T, data []byte) []float32 {
	t.Helper()
	if len(data)%4 != 0 {
		t.Fatalf("float32 blob has %d bytes", len(data))
	}
	out := make([]float32, len(data)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return out
}

func parseHFTestConfig(t *testing.T, cfg *ArchConfig) *ArchConfig {
	t.Helper()
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal config: %v", err)
	}
	parsed, err := ParseArchConfig(data, "hf_test")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	return parsed
}
