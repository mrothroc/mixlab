package arch

import (
	"fmt"
	"strings"
	"testing"
)

func TestGPT2CompatDefaultsPreserveRoPE(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"name":"default_rope",
		"model_dim":16,
		"vocab_size":32,
		"seq_len":4,
		"tie_embeddings":true,
		"blocks":[{"type":"plain","heads":4}],
		"training":{"batch_tokens":8}
	}`), "default_rope")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if got := cfg.EffectivePositionalEmbedding(); got != PositionalEmbeddingRope {
		t.Fatalf("EffectivePositionalEmbedding=%q want rope", got)
	}
	if got := cfg.EffectiveMaxPositions(); got != cfg.SeqLen {
		t.Fatalf("EffectiveMaxPositions=%d want %d", got, cfg.SeqLen)
	}
	metas, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatalf("CollectWeightShapesFromConfig: %v", err)
	}
	if idx := weightMetaIndex(metas, "position_embeddings"); idx >= 0 {
		t.Fatalf("unexpected position_embeddings at index %d", idx)
	}
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig: %v", err)
	}
	if got := countOps(prog, OpRoPE); got == 0 {
		t.Fatal("default plain config did not emit RoPE")
	}
}

func TestGPT2CompatLearnedAbsoluteIRAndWeights(t *testing.T) {
	cfg := mustParseGPT2CompatConfig(t, 16, 32, 4, 8, 4, 4, 1)
	metas, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatalf("CollectWeightShapesFromConfig: %v", err)
	}
	assertWeightShape(t, metas, "position_embeddings", []int{8, 16})
	assertWeightShape(t, metas, "ffn_norm_scale", []int{16})
	assertWeightShape(t, metas, "ffn_norm_bias", []int{16})
	assertWeightShape(t, metas, "ff1_bias", []int{64})
	assertWeightShape(t, metas, "ff2_bias", []int{16})
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig: %v", err)
	}
	if got := countOps(prog, OpRoPE); got != 0 {
		t.Fatalf("RoPE ops=%d want 0 for learned absolute positions", got)
	}
	if got := countOps(prog, OpEmbed); got < 2 {
		t.Fatalf("Embed ops=%d want at least token + position embedding", got)
	}
	if got := countOps(prog, OpDropout); got == 0 {
		t.Fatal("missing embedding dropout op")
	}
	if got := countOps(prog, OpGELU); got == 0 {
		t.Fatal("missing gelu_new/tanh GELU op")
	}
	if got := countOps(prog, OpGELUExact); got != 0 {
		t.Fatalf("GELUExact ops=%d want 0 for gelu_new", got)
	}
	if got := countOps(prog, OpLayerNorm); got < 3 {
		t.Fatalf("LayerNorm ops=%d want final + attention norm + FFN norm", got)
	}
}

func TestGPT2CompatExactGELUIR(t *testing.T) {
	cfg := mustParseGPT2CompatConfig(t, 16, 32, 4, 8, 4, 4, 1)
	cfg.Blocks[0].FFNActivation = "gelu"
	prog, err := BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{})
	if err != nil {
		t.Fatalf("BuildTrainingIRProgramFromConfig: %v", err)
	}
	if got := countOps(prog, OpGELUExact); got != 1 {
		t.Fatalf("GELUExact ops=%d want 1 for ffn_activation=gelu", got)
	}
}

func TestGPT2StrictSmallParameterCount(t *testing.T) {
	cfg := mustParseGPT2CompatConfig(t, 768, 16384, 1024, 1024, 12, 4, 12)
	metas, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatalf("CollectWeightShapesFromConfig: %v", err)
	}
	if got, want := totalParams(metas), 98425344; got != want {
		t.Fatalf("strict-small trainable params=%d want %d", got, want)
	}
	counted, expanded, err := ParameterCountsFromConfig(cfg)
	if err != nil {
		t.Fatalf("ParameterCountsFromConfig: %v", err)
	}
	if counted != 98425344 || expanded != 98425344 {
		t.Fatalf("ParameterCountsFromConfig=(%d,%d) want 98425344/98425344", counted, expanded)
	}
}

func TestGPT2CompatValidationRejectsExplicitRoPEWithLearnedAbsolute(t *testing.T) {
	_, err := ParseArchConfig([]byte(`{
		"name":"bad_rope",
		"model_dim":16,
		"vocab_size":32,
		"seq_len":4,
		"positional_embedding":"learned_absolute",
		"blocks":[{"type":"plain","heads":4,"rope_dims":8}],
		"training":{"batch_tokens":8}
	}`), "bad_rope")
	if err == nil || !strings.Contains(err.Error(), "rope_dims") {
		t.Fatalf("ParseArchConfig error=%v want rope_dims rejection", err)
	}
}

func mustParseGPT2CompatConfig(t *testing.T, dim, vocab, seqLen, maxPositions, heads int, mlpMult float64, layers int) *ArchConfig {
	t.Helper()
	var blocks strings.Builder
	for i := 0; i < layers; i++ {
		if i > 0 {
			blocks.WriteByte(',')
		}
		fmt.Fprintf(&blocks, `{"type":"plain","heads":%d,"attention_mask":"causal","attn_bias":true,"ffn_activation":"gelu_new","ffn_pre_norm":true,"ffn_bias":true}`, heads)
	}
	cfg, err := ParseArchConfig([]byte(fmt.Sprintf(`{
		"name":"gpt2_compat",
		"model_dim":%d,
		"vocab_size":%d,
		"seq_len":%d,
		"mlp_mult":%g,
		"tie_embeddings":true,
		"norm_type":"layernorm",
		"norm_affine":true,
		"positional_embedding":"learned_absolute",
		"max_positions":%d,
		"embedding_dropout":0.1,
		"hf_export_format":"gpt2",
		"blocks":[%s],
		"training":{"batch_tokens":%d,"objective":"causal"}
	}`, dim, vocab, seqLen, mlpMult, maxPositions, blocks.String(), seqLen)), "gpt2_compat")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	return cfg
}

func weightMetaIndex(metas []WeightMeta, name string) int {
	for i, meta := range metas {
		if meta.Name == name {
			return i
		}
	}
	return -1
}

func assertWeightShape(t *testing.T, metas []WeightMeta, name string, shape []int) {
	t.Helper()
	idx := weightMetaIndex(metas, name)
	if idx < 0 {
		t.Fatalf("missing weight %q", name)
	}
	if !sameInts(metas[idx].Shape, shape) {
		t.Fatalf("%s shape=%v want %v", name, metas[idx].Shape, shape)
	}
}

func totalParams(metas []WeightMeta) int {
	total := 0
	for _, meta := range metas {
		n := 1
		for _, dim := range meta.Shape {
			n *= dim
		}
		total += n
	}
	return total
}
