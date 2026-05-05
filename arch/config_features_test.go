package arch

import (
	"encoding/json"
	"io"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestPlainBlockInvalidKVHeads(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 8, KVHeads: 3}},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for heads % kv_heads != 0")
	}
	if !strings.Contains(err.Error(), "heads % kv_heads == 0") {
		t.Errorf("error should mention kv_heads divisibility: %v", err)
	}
}

func TestParseArchConfig_BigramDefaultsToModelDim(t *testing.T) {
	cfg := ArchConfig{
		Name:            "bigram",
		ModelDim:        64,
		VocabSize:       256,
		SeqLen:          32,
		BigramVocabSize: 512,
		Blocks:          []BlockSpec{{Type: "plain", Heads: 4}},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	got, err := ParseArchConfig(data, "test_bigram")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if got.BigramDim != 64 {
		t.Fatalf("BigramDim=%d want 64", got.BigramDim)
	}
}

func TestParseArchConfig_BigramDisabledZerosDim(t *testing.T) {
	cfg := ArchConfig{
		Name:      "no_bigram",
		ModelDim:  64,
		VocabSize: 256,
		SeqLen:    32,
		BigramDim: 13,
		Blocks:    []BlockSpec{{Type: "plain", Heads: 4}},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	got, err := ParseArchConfig(data, "test_bigram_disabled")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if got.BigramDim != 0 {
		t.Fatalf("BigramDim=%d want 0 when bigram disabled", got.BigramDim)
	}
}

func TestParseArchConfig_TrigramDefaultsAndTrainingKnobs(t *testing.T) {
	cfg := ArchConfig{
		Name:             "trigram",
		ModelDim:         64,
		VocabSize:        256,
		SeqLen:           32,
		BigramVocabSize:  512,
		BigramDim:        24,
		TrigramVocabSize: 1024,
		Blocks:           []BlockSpec{{Type: "plain", Heads: 4}},
		Training: TrainingSpec{
			NewtonSchulzVariant: "polar_express",
			MinLRFraction:       0.1,
		},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	got, err := ParseArchConfig(data, "test_trigram_defaults")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if got.TrigramDim != 24 {
		t.Fatalf("TrigramDim=%d want 24", got.TrigramDim)
	}
	if got.Training.NewtonSchulzVariant != "polar_express" {
		t.Fatalf("NewtonSchulzVariant=%q want polar_express", got.Training.NewtonSchulzVariant)
	}
	if got.Training.MinLRFraction != 0.1 {
		t.Fatalf("MinLRFraction=%g want 0.1", got.Training.MinLRFraction)
	}
}

func TestParseArchConfig_RejectsInvalidTrigramAndTrainingKnobs(t *testing.T) {
	tests := []struct {
		name string
		cfg  ArchConfig
		want string
	}{
		{
			name: "bad_trigram_vocab",
			cfg: ArchConfig{
				ModelDim:         64,
				VocabSize:        256,
				SeqLen:           32,
				TrigramVocabSize: 1,
				Blocks:           []BlockSpec{{Type: "plain", Heads: 4}},
			},
			want: "trigram_vocab_size",
		},
		{
			name: "bad_variant",
			cfg: ArchConfig{
				ModelDim:  64,
				VocabSize: 256,
				SeqLen:    32,
				Blocks:    []BlockSpec{{Type: "plain", Heads: 4}},
				Training:  TrainingSpec{NewtonSchulzVariant: "weird"},
			},
			want: "newton_schulz_variant",
		},
		{
			name: "bad_min_lr_fraction",
			cfg: ArchConfig{
				ModelDim:  64,
				VocabSize: 256,
				SeqLen:    32,
				Blocks:    []BlockSpec{{Type: "plain", Heads: 4}},
				Training:  TrainingSpec{MinLRFraction: 1},
			},
			want: "min_lr_fraction",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.cfg)
			if err != nil {
				t.Fatalf("Marshal: %v", err)
			}
			_, err = ParseArchConfig(data, tt.name)
			if err == nil {
				t.Fatal("expected validation error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error %q missing %q", err, tt.want)
			}
		})
	}
}

func TestParseArchConfig_LogitSoftcapPreserved(t *testing.T) {
	cfg := ArchConfig{
		Name:         "softcap",
		ModelDim:     64,
		VocabSize:    256,
		SeqLen:       32,
		LogitSoftcap: 12.5,
		Blocks:       []BlockSpec{{Type: "plain", Heads: 4}},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	got, err := ParseArchConfig(data, "test_softcap")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if got.LogitSoftcap != 12.5 {
		t.Fatalf("LogitSoftcap=%g want 12.5", got.LogitSoftcap)
	}
}

func TestParseArchConfig_AcceptsGatedLinearSSM(t *testing.T) {
	cfg := ArchConfig{
		Name:      "gated_linear_ssm",
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks: []BlockSpec{
			{Type: "gated_linear_ssm", InnerDim: 192},
		},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := ParseArchConfig(data, "test_gated_linear_ssm")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got.Blocks[0].Type != "gated_linear_ssm" {
		t.Fatalf("type=%q want gated_linear_ssm", got.Blocks[0].Type)
	}
	if got.Blocks[0].InnerDim != 192 {
		t.Fatalf("inner_dim=%d want 192", got.Blocks[0].InnerDim)
	}
}

func TestParseArchConfig_Mamba3DeprecatedAliasWarnsAndMatchesGatedLinearSSM(t *testing.T) {
	gated := ArchConfig{
		Name:        "gated_linear_ssm",
		ModelDim:    64,
		VocabSize:   256,
		SeqLen:      16,
		BlockScales: true,
		Blocks: []BlockSpec{
			{Type: "gated_linear_ssm", InnerDim: 96},
		},
		Training: TrainingSpec{BatchTokens: 16},
	}
	legacy := gated
	legacy.Name = "legacy_mamba3"
	legacy.Blocks = []BlockSpec{{Type: "mamba3", InnerDim: 96}}

	gatedData, err := json.Marshal(gated)
	if err != nil {
		t.Fatalf("marshal gated: %v", err)
	}
	legacyData, err := json.Marshal(legacy)
	if err != nil {
		t.Fatalf("marshal legacy: %v", err)
	}

	mamba3AliasWarningSeen.Store(false)
	warned, stderr := parseConfigCapturingStderr(t, legacyData, "legacy_mamba3")
	if !strings.Contains(stderr, mamba3AliasWarning) {
		t.Fatalf("stderr %q does not contain warning %q", stderr, mamba3AliasWarning)
	}
	if warned.Blocks[0].Type != "mamba3" {
		t.Fatalf("legacy type=%q want preserved mamba3", warned.Blocks[0].Type)
	}
	legacyForIR := *warned

	gatedParsed, err := ParseArchConfig(gatedData, "gated_linear_ssm")
	if err != nil {
		t.Fatalf("parse gated: %v", err)
	}
	warned.Name = gatedParsed.Name
	warned.Blocks[0].Type = gatedParsed.Blocks[0].Type
	if !reflect.DeepEqual(warned, gatedParsed) {
		t.Fatalf("legacy config differs after normalizing alias\nlegacy=%+v\ngated=%+v", warned, gatedParsed)
	}

	legacyProg, err := BuildIRProgramFromConfig(&legacyForIR)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig legacy: %v", err)
	}
	gatedProg, err := BuildIRProgramFromConfig(gatedParsed)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig gated: %v", err)
	}
	if !reflect.DeepEqual(legacyProg, gatedProg) {
		t.Fatalf("legacy and gated_linear_ssm IR programs differ")
	}
}

func parseConfigCapturingStderr(t *testing.T, data []byte, source string) (*ArchConfig, string) {
	t.Helper()

	oldStderr := os.Stderr
	readEnd, writeEnd, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe: %v", err)
	}
	os.Stderr = writeEnd
	defer func() {
		os.Stderr = oldStderr
	}()

	cfg, parseErr := ParseArchConfig(data, source)
	if closeErr := writeEnd.Close(); closeErr != nil {
		t.Fatalf("close stderr pipe writer: %v", closeErr)
	}
	out, readErr := io.ReadAll(readEnd)
	if readErr != nil {
		t.Fatalf("read stderr pipe: %v", readErr)
	}
	if closeErr := readEnd.Close(); closeErr != nil {
		t.Fatalf("close stderr pipe reader: %v", closeErr)
	}
	if parseErr != nil {
		t.Fatalf("ParseArchConfig: %v", parseErr)
	}
	return cfg, string(out)
}

func TestParseArchConfig_AcceptsMamba3Canonical(t *testing.T) {
	useConv := false
	cfg := ArchConfig{
		Name:      "mamba3_canonical",
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks: []BlockSpec{{
			Type:       "mamba3-canonical",
			InnerDim:   192,
			StateSize:  16,
			NGroups:    4,
			DTRank:     8,
			ConvKernel: 4,
			UseConv:    &useConv,
			DTMin:      0.001,
			DTMax:      0.1,
		}},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := ParseArchConfig(data, "test_mamba3_canonical")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got.Blocks[0].Type != "mamba3-canonical" || got.Blocks[0].InnerDim != 192 || got.Blocks[0].UseConv == nil || *got.Blocks[0].UseConv {
		t.Fatalf("parsed block = %+v", got.Blocks[0])
	}
}

func TestParseArchConfig_AcceptsCrossAttention(t *testing.T) {
	cfg := ArchConfig{
		Name:      "xattn",
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks: []BlockSpec{
			{Type: "plain", Heads: 4},
			{Type: "cross_attention", Heads: 4, SourceStream: "x"},
		},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := ParseArchConfig(data, "test_cross_attention")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got.Blocks[1].SourceStream != "x" {
		t.Fatalf("source_stream=%q want x", got.Blocks[1].SourceStream)
	}
}

func TestParseArchConfig_RejectsCrossAttentionWithoutSourceStream(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks: []BlockSpec{
			{Type: "plain", Heads: 4},
			{Type: "cross_attention", Heads: 4},
		},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test_cross_attention_missing")
	if err == nil {
		t.Fatal("expected error for cross_attention without source_stream")
	}
	if !strings.Contains(err.Error(), "source_stream") {
		t.Fatalf("error should mention source_stream: %v", err)
	}
}

func TestParseArchConfig_AcceptsKVSource(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    64,
		Blocks: []BlockSpec{
			{Type: "plain", Heads: 8, KVHeads: 4},
			{Type: "swiglu"},
			{Type: "plain", Heads: 8, KVHeads: 4, KVSource: 1},
		},
	}
	data, _ := json.Marshal(cfg)
	got, err := ParseArchConfig(data, "kv_source_ok")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if got.Blocks[2].KVSource != 1 {
		t.Fatalf("kv_source=%d want 1", got.Blocks[2].KVSource)
	}
}

func TestParseArchConfig_RejectsInvalidKVSource(t *testing.T) {
	tests := []struct {
		name   string
		blocks []BlockSpec
		want   string
	}{
		{
			name: "forward ref",
			blocks: []BlockSpec{
				{Type: "plain", Heads: 8, KVHeads: 4, KVSource: 2},
				{Type: "plain", Heads: 8, KVHeads: 4},
			},
			want: "must reference an earlier block",
		},
		{
			name: "non plain source",
			blocks: []BlockSpec{
				{Type: "swiglu"},
				{Type: "plain", Heads: 8, KVHeads: 4, KVSource: 1},
			},
			want: "want plain",
		},
		{
			name: "heads mismatch",
			blocks: []BlockSpec{
				{Type: "plain", Heads: 4, KVHeads: 4},
				{Type: "plain", Heads: 8, KVHeads: 4, KVSource: 1},
			},
			want: "source heads=4",
		},
		{
			name: "kv heads mismatch",
			blocks: []BlockSpec{
				{Type: "plain", Heads: 8, KVHeads: 2},
				{Type: "plain", Heads: 8, KVHeads: 4, KVSource: 1},
			},
			want: "source kv_heads=2",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := ArchConfig{
				ModelDim:  128,
				VocabSize: 1024,
				SeqLen:    64,
				Blocks:    tc.blocks,
			}
			data, _ := json.Marshal(cfg)
			_, err := ParseArchConfig(data, "kv_source_bad")
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q does not contain %q", err, tc.want)
			}
		})
	}
}

func TestValidCustomBlockConfig(t *testing.T) {
	cfg := ArchConfig{
		Name:      "test_custom",
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks: []BlockSpec{
			{Type: "plain", Heads: 4},
			{
				Type: "custom",
				Name: "geglu",
				Weights: []CustomWeightSpec{
					{Name: "w_gate", Shape: []string{"D", "FFN"}},
					{Name: "w_up", Shape: []string{"D", "FFN"}},
					{Name: "w_down", Shape: []string{"FFN", "D"}},
				},
				Ops: []CustomOpSpec{
					{Op: "matmul", Inputs: []string{"x", "w_gate"}, Output: "gate"},
					{Op: "silu", Inputs: []string{"gate"}, Output: "gate_act"},
					{Op: "matmul", Inputs: []string{"x", "w_up"}, Output: "up"},
					{Op: "mul", Inputs: []string{"gate_act", "up"}, Output: "ff"},
					{Op: "matmul", Inputs: []string{"ff", "w_down"}, Output: "ff_out"},
					{Op: "add", Inputs: []string{"x", "ff_out"}, Output: "x"},
				},
			},
		},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := ParseArchConfig(data, "test_custom")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(got.Blocks) != 2 {
		t.Errorf("blocks len = %d, want 2", len(got.Blocks))
	}
	if got.Blocks[1].Name != "geglu" {
		t.Errorf("custom block name = %q, want geglu", got.Blocks[1].Name)
	}
	if len(got.Blocks[1].Weights) != 3 {
		t.Errorf("custom weights len = %d, want 3", len(got.Blocks[1].Weights))
	}
	if len(got.Blocks[1].Ops) != 6 {
		t.Errorf("custom ops len = %d, want 6", len(got.Blocks[1].Ops))
	}
}

func TestCustomBlockMissingName(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks: []BlockSpec{
			{
				Type: "custom",
				Weights: []CustomWeightSpec{
					{Name: "w", Shape: []string{"D", "D"}},
				},
				Ops: []CustomOpSpec{
					{Op: "matmul", Inputs: []string{"x", "w"}, Output: "y"},
				},
			},
		},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for custom block without name")
	}
	if !strings.Contains(err.Error(), "requires a name") {
		t.Errorf("error should mention name: %v", err)
	}
}

func TestCustomBlockMissingWeights(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks: []BlockSpec{
			{
				Type: "custom",
				Name: "empty_weights",
				Ops: []CustomOpSpec{
					{Op: "add", Inputs: []string{"x", "x"}, Output: "y"},
				},
			},
		},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for custom block without weights")
	}
	if !strings.Contains(err.Error(), "at least one weight") {
		t.Errorf("error should mention weights: %v", err)
	}
}

func TestCustomBlockMissingOps(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks: []BlockSpec{
			{
				Type: "custom",
				Name: "empty_ops",
				Weights: []CustomWeightSpec{
					{Name: "w", Shape: []string{"D", "D"}},
				},
			},
		},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for custom block without ops")
	}
	if !strings.Contains(err.Error(), "at least one op") {
		t.Errorf("error should mention ops: %v", err)
	}
}

func TestCustomBlockWeightMissingName(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks: []BlockSpec{
			{
				Type: "custom",
				Name: "bad_weight",
				Weights: []CustomWeightSpec{
					{Name: "", Shape: []string{"D", "D"}},
				},
				Ops: []CustomOpSpec{
					{Op: "matmul", Inputs: []string{"x", "w"}, Output: "y"},
				},
			},
		},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for weight with empty name")
	}
	if !strings.Contains(err.Error(), "missing name") {
		t.Errorf("error should mention missing name: %v", err)
	}
}

func TestCustomBlockWeightMissingShape(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks: []BlockSpec{
			{
				Type: "custom",
				Name: "bad_shape",
				Weights: []CustomWeightSpec{
					{Name: "w", Shape: nil},
				},
				Ops: []CustomOpSpec{
					{Op: "matmul", Inputs: []string{"x", "w"}, Output: "y"},
				},
			},
		},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for weight with missing shape")
	}
	if !strings.Contains(err.Error(), "missing shape") {
		t.Errorf("error should mention missing shape: %v", err)
	}
}

func TestCustomBlockOpMissingOutput(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks: []BlockSpec{
			{
				Type: "custom",
				Name: "bad_op",
				Weights: []CustomWeightSpec{
					{Name: "w", Shape: []string{"D", "D"}},
				},
				Ops: []CustomOpSpec{
					{Op: "matmul", Inputs: []string{"x", "w"}},
				},
			},
		},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for op with missing output")
	}
	if !strings.Contains(err.Error(), "missing output") {
		t.Errorf("error should mention missing output: %v", err)
	}
}

func TestCustomBlockOpMissingOpName(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks: []BlockSpec{
			{
				Type: "custom",
				Name: "bad_op",
				Weights: []CustomWeightSpec{
					{Name: "w", Shape: []string{"D", "D"}},
				},
				Ops: []CustomOpSpec{
					{Op: "", Inputs: []string{"x", "w"}, Output: "y"},
				},
			},
		},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "test")
	if err == nil {
		t.Fatal("expected error for op with empty name")
	}
	if !strings.Contains(err.Error(), "missing op name") {
		t.Errorf("error should mention missing op name: %v", err)
	}
}
