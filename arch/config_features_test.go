package arch

import (
	"encoding/json"
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

func TestParseArchConfig_AcceptsMamba3(t *testing.T) {
	cfg := ArchConfig{
		Name:      "mamba3",
		ModelDim:  128,
		VocabSize: 1024,
		SeqLen:    128,
		Blocks: []BlockSpec{
			{Type: "mamba3", InnerDim: 192},
		},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := ParseArchConfig(data, "test_mamba3")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got.Blocks[0].Type != "mamba3" {
		t.Fatalf("type=%q want mamba3", got.Blocks[0].Type)
	}
	if got.Blocks[0].InnerDim != 192 {
		t.Fatalf("inner_dim=%d want 192", got.Blocks[0].InnerDim)
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
