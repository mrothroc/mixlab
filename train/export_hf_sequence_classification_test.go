package train

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExportHFSequenceClassificationMetadata(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
		"name": "hf_sequence_classifier",
		"model_dim": 4,
		"vocab_size": 7,
		"seq_len": 3,
		"mlp_mult": 1.0,
		"blocks": [
			{"type": "plain", "heads": 2},
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

	var doc hfConfigJSON
	readJSON(t, filepath.Join(outDir, "config.json"), &doc)
	if got := doc.AutoMap["AutoModelForSequenceClassification"]; got != "modeling_mixlab.MixlabForSequenceClassification" {
		t.Fatalf("AutoModelForSequenceClassification auto_map=%q", got)
	}
	if got := doc.SequenceClassificationPooling; got != "last" {
		t.Fatalf("sequence_classification_pooling=%q, want last", got)
	}
	if _, err := os.Stat(filepath.Join(outDir, "pooling_mixlab.py")); err != nil {
		t.Fatalf("missing exported pooling_mixlab.py: %v", err)
	}

	tensors, err := loadSafetensors(filepath.Join(outDir, "model.safetensors"))
	if err != nil {
		t.Fatalf("load exported model.safetensors: %v", err)
	}
	for name := range tensors {
		if strings.HasPrefix(name, "classifier.") {
			t.Fatalf("pretraining checkpoint unexpectedly contains task-head tensor %q", name)
		}
	}

	modeling, err := os.ReadFile(filepath.Join(outDir, "modeling_mixlab.py"))
	if err != nil {
		t.Fatalf("read modeling template: %v", err)
	}
	source := string(modeling)
	classStart := strings.Index(source, "class MixlabForSequenceClassification")
	if classStart < 0 {
		t.Fatal("modeling template is missing MixlabForSequenceClassification")
	}
	classSource := source[classStart:]
	if !strings.Contains(classSource, "pool_sequence(") {
		t.Fatal("sequence-classification head does not call shared pool_sequence")
	}
	if strings.Contains(classSource, "[:, -1]") {
		t.Fatal("sequence-classification head contains unsafe bare final-position pooling")
	}
}

func TestHFSequenceClassificationPoolingInference(t *testing.T) {
	tests := []struct {
		name         string
		blocks       []BlockSpec
		maskedBlocks []map[string]any
		want         string
	}{
		{
			name:   "causal plain",
			blocks: []BlockSpec{{Type: "plain", Heads: 2}},
			want:   "last",
		},
		{
			name:   "bidirectional plain",
			blocks: []BlockSpec{{Type: "plain", Heads: 2, AttentionMask: "bidirectional"}},
			want:   "mean",
		},
		{
			name:   "recurrent",
			blocks: []BlockSpec{{Type: "ttt_mlp", Heads: 2}},
			want:   "last",
		},
		{
			name: "mixed masks require override",
			blocks: []BlockSpec{
				{Type: "plain", Heads: 2, AttentionMask: "causal"},
				{Type: "plain", Heads: 2, AttentionMask: "bidirectional"},
			},
			want: "",
		},
		{
			name:         "masked export view",
			blocks:       []BlockSpec{{Type: "plain", Heads: 2}},
			maskedBlocks: []map[string]any{{"type": "plain", "heads": 2, "attention_mask": "bidirectional"}},
			want:         "mean",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &ArchConfig{Blocks: tt.blocks}
			if got := hfSequenceClassificationPooling(cfg, tt.maskedBlocks); got != tt.want {
				t.Fatalf("hfSequenceClassificationPooling()=%q, want %q", got, tt.want)
			}
		})
	}
}

func TestHFSequenceClassificationAmbiguityDoesNotBlockExport(t *testing.T) {
	dir := t.TempDir()
	cfgPath, weightsPath, tokenizerDir := writeHFExportFixture(t, dir, `{
		"name": "hf_sequence_classifier_ambiguous",
		"model_dim": 4,
		"vocab_size": 7,
		"seq_len": 3,
		"mlp_mult": 1.0,
		"blocks": [
			{"type": "plain", "heads": 2, "attention_mask": "causal"},
			{"type": "plain", "heads": 2, "attention_mask": "bidirectional"}
		],
		"training": {"steps": 1, "batch_tokens": 3, "seed": 456}
	}`)
	outDir := filepath.Join(dir, "hf_out")
	if err := RunExportHF(ExportHFOptions{
		ConfigPath:      cfgPath,
		SafetensorsLoad: weightsPath,
		OutputDir:       outDir,
		TokenizerSource: tokenizerDir,
	}); err != nil {
		t.Fatalf("ambiguous classification pooling must not block export: %v", err)
	}

	var doc hfConfigJSON
	readJSON(t, filepath.Join(outDir, "config.json"), &doc)
	if doc.SequenceClassificationPooling != "" {
		t.Fatalf("ambiguous sequence_classification_pooling=%q, want omitted", doc.SequenceClassificationPooling)
	}
	if _, ok := doc.AutoMap["AutoModelForSequenceClassification"]; !ok {
		t.Fatal("ambiguous export should remain loadable with an explicit pooling override")
	}
}
