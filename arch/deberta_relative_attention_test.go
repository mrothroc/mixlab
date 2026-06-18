package arch

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestParseArchConfig_DebertaRelativeAttention(t *testing.T) {
	cfg := ArchConfig{
		Name:      "deberta_relative",
		ModelDim:  48,
		VocabSize: 256,
		SeqLen:    16,
		Blocks: []BlockSpec{{
			Type:              "plain",
			Heads:             6,
			RelativeAttention: RelativeAttentionDebertaP2CC2P,
		}},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got, err := ParseArchConfig(data, "deberta_relative")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if got.Blocks[0].RelativeAttention != RelativeAttentionDebertaP2CC2P {
		t.Fatalf("relative_attention=%q", got.Blocks[0].RelativeAttention)
	}
	if win := effectiveRelativeAttentionWindow(got.Blocks[0]); win != defaultRelativeAttentionWindow {
		t.Fatalf("effective window=%d want %d", win, defaultRelativeAttentionWindow)
	}
}

func TestParseArchConfig_RejectsInvalidDebertaRelativeAttention(t *testing.T) {
	tests := []struct {
		name  string
		block BlockSpec
		want  string
	}{
		{
			name:  "invalid mode",
			block: BlockSpec{Type: "plain", Heads: 4, RelativeAttention: "bogus"},
			want:  "relative_attention",
		},
		{
			name:  "negative window",
			block: BlockSpec{Type: "plain", Heads: 4, RelativeAttention: RelativeAttentionDebertaP2CC2P, RelativeAttentionWindow: -1},
			want:  "relative_attention_window",
		},
		{
			name:  "rope conflict",
			block: BlockSpec{Type: "plain", Heads: 4, RelativeAttention: RelativeAttentionDebertaP2CC2P, RopeDims: 8},
			want:  "rope_dims",
		},
		{
			name:  "kv source conflict",
			block: BlockSpec{Type: "plain", Heads: 4, RelativeAttention: RelativeAttentionDebertaP2CC2P, KVSource: 1},
			want:  "kv_source",
		},
		{
			name:  "dimension mismatch",
			block: BlockSpec{Type: "plain", Heads: 5, RelativeAttention: RelativeAttentionDebertaP2CC2P},
			want:  "divisible",
		},
		{
			name:  "invalid parameterization",
			block: BlockSpec{Type: "plain", Heads: 4, RelativeAttention: RelativeAttentionDebertaP2CC2P, RelativeAttentionParameterization: "bogus"},
			want:  "relative_attention_parameterization",
		},
		{
			name:  "shared parameterization without relative attention",
			block: BlockSpec{Type: "plain", Heads: 4, RelativeAttentionParameterization: RelativeAttentionParamSharedQKReuse},
			want:  "shared_qk_reuse",
		},
		{
			name:  "shared relative embedding norm without shared parameterization",
			block: BlockSpec{Type: "plain", Heads: 4, RelativeAttention: RelativeAttentionDebertaP2CC2P, RelativeAttentionEmbeddingNorm: RelativeAttentionEmbeddingNormLayerNorm},
			want:  "relative_attention_embedding_norm",
		},
		{
			name:  "invalid attention post norm",
			block: BlockSpec{Type: "plain", Heads: 4, AttnPostNorm: "middle"},
			want:  "attn_post_norm",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := ArchConfig{
				ModelDim:  48,
				VocabSize: 256,
				SeqLen:    16,
				Blocks:    []BlockSpec{tc.block},
			}
			data, _ := json.Marshal(cfg)
			_, err := ParseArchConfig(data, tc.name)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q does not contain %q", err, tc.want)
			}
		})
	}
}

func TestParseArchConfig_RejectsMixedSharedRelativeWindows(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  48,
		VocabSize: 256,
		SeqLen:    16,
		Blocks: []BlockSpec{
			{Type: "plain", Heads: 4, RelativeAttention: RelativeAttentionDebertaP2CC2P, RelativeAttentionWindow: 16, RelativeAttentionParameterization: RelativeAttentionParamSharedQKReuse},
			{Type: "plain", Heads: 4, RelativeAttention: RelativeAttentionDebertaP2CC2P, RelativeAttentionWindow: 32, RelativeAttentionParameterization: RelativeAttentionParamSharedQKReuse},
		},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "mixed_shared_relative_windows")
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "same effective relative_attention_window") {
		t.Fatalf("error %q does not mention shared window", err)
	}
}

func TestParseArchConfig_RejectsMixedSharedRelativeEmbeddingNorms(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  48,
		VocabSize: 256,
		SeqLen:    16,
		Blocks: []BlockSpec{
			{Type: "plain", Heads: 4, RelativeAttention: RelativeAttentionDebertaP2CC2P, RelativeAttentionWindow: 16, RelativeAttentionParameterization: RelativeAttentionParamSharedQKReuse, RelativeAttentionEmbeddingNorm: RelativeAttentionEmbeddingNormLayerNorm},
			{Type: "plain", Heads: 4, RelativeAttention: RelativeAttentionDebertaP2CC2P, RelativeAttentionWindow: 16, RelativeAttentionParameterization: RelativeAttentionParamSharedQKReuse},
		},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "mixed_shared_relative_embedding_norms")
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "same relative_attention_embedding_norm") {
		t.Fatalf("error %q does not mention shared embedding norm", err)
	}
}

func TestParseArchConfig_RejectsKVSourceFromDebertaRelativeAttention(t *testing.T) {
	cfg := ArchConfig{
		ModelDim:  48,
		VocabSize: 256,
		SeqLen:    16,
		Blocks: []BlockSpec{
			{Type: "plain", Heads: 4, RelativeAttention: RelativeAttentionDebertaP2CC2P},
			{Type: "plain", Heads: 4, KVSource: 1},
		},
	}
	data, _ := json.Marshal(cfg)
	_, err := ParseArchConfig(data, "relative_kv_source")
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "source block uses relative_attention") {
		t.Fatalf("error %q does not mention relative_attention source", err)
	}
}

func TestDebertaSharedRelativeAttentionWeightLayout(t *testing.T) {
	sharedSpec := BlockSpec{
		Type:                              "plain",
		Heads:                             4,
		RelativeAttention:                 RelativeAttentionDebertaP2CC2P,
		RelativeAttentionWindow:           3,
		RelativeAttentionParameterization: RelativeAttentionParamSharedQKReuse,
	}
	n, err := BlockWeightCount(sharedSpec, false, false)
	if err != nil {
		t.Fatalf("BlockWeightCount: %v", err)
	}
	if n != 7 {
		t.Fatalf("shared block weight count=%d want 7", n)
	}
	blockMetas, err := blockWeightShapes(sharedSpec, 64, 16, 1, 256, DefaultFFNMultiplier, false, false)
	if err != nil {
		t.Fatalf("blockWeightShapes: %v", err)
	}
	for _, meta := range blockMetas {
		switch meta.Name {
		case "relative_embeddings", "w_pos_key", "w_pos_query":
			t.Fatalf("shared block unexpectedly has per-block relative weight %q", meta.Name)
		}
	}
	cfg := &ArchConfig{
		ModelDim:  64,
		VocabSize: 256,
		SeqLen:    16,
		Blocks:    []BlockSpec{sharedSpec, sharedSpec},
	}
	metas, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatalf("CollectWeightShapesFromConfig: %v", err)
	}
	countShared := 0
	for _, meta := range metas {
		if meta.Name == SharedRelativeEmbeddingsWeightName {
			countShared++
			if got, want := meta.Shape, []int{5, 64}; !sameInts(got, want) {
				t.Fatalf("shared relative shape=%v want %v", got, want)
			}
		}
	}
	if countShared != 1 {
		t.Fatalf("shared relative embedding count=%d want 1", countShared)
	}
}

func TestDebertaSharedRelativeAttentionEmbeddingNormWeightLayout(t *testing.T) {
	sharedSpec := BlockSpec{
		Type:                              "plain",
		Heads:                             4,
		RelativeAttention:                 RelativeAttentionDebertaP2CC2P,
		RelativeAttentionWindow:           3,
		RelativeAttentionParameterization: RelativeAttentionParamSharedQKReuse,
		RelativeAttentionEmbeddingNorm:    RelativeAttentionEmbeddingNormLayerNorm,
	}
	cfg := &ArchConfig{
		ModelDim:  64,
		VocabSize: 256,
		SeqLen:    16,
		Blocks:    []BlockSpec{sharedSpec, sharedSpec},
	}
	metas, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatalf("CollectWeightShapesFromConfig: %v", err)
	}
	count, err := CountWeights(DefaultFFNMultiplier, false, false, false, false, []BlockSpec{sharedSpec, sharedSpec})
	if err != nil {
		t.Fatalf("CountWeights: %v", err)
	}
	if count != len(metas) {
		t.Fatalf("CountWeights=%d len(metas)=%d", count, len(metas))
	}
	want := []struct {
		name  string
		shape []int
	}{
		{SharedRelativeEmbeddingsWeightName, []int{5, 64}},
		{SharedRelativeNormScaleWeightName, []int{64}},
		{SharedRelativeNormBiasWeightName, []int{64}},
	}
	found := 0
	for _, meta := range metas {
		if found >= len(want) || meta.Name != want[found].name {
			continue
		}
		if !sameInts(meta.Shape, want[found].shape) {
			t.Fatalf("%s shape=%v want %v", meta.Name, meta.Shape, want[found].shape)
		}
		found++
	}
	if found != len(want) {
		t.Fatalf("did not find shared relative embedding norm weight sequence %v in %+v", want, metas)
	}
}

func TestDebertaSharedRelativeAttentionGPTBERTParamDelta(t *testing.T) {
	const (
		D       = 384
		window  = 32
		layers  = 12
		wantCut = 3805056
	)
	perBlock := make([]BlockSpec, layers)
	shared := make([]BlockSpec, layers)
	for i := 0; i < layers; i++ {
		perBlock[i] = BlockSpec{Type: "plain", Heads: 6, RelativeAttention: RelativeAttentionDebertaP2CC2P, RelativeAttentionWindow: window}
		shared[i] = BlockSpec{Type: "plain", Heads: 6, RelativeAttention: RelativeAttentionDebertaP2CC2P, RelativeAttentionWindow: window, RelativeAttentionParameterization: RelativeAttentionParamSharedQKReuse}
	}
	perBlockMetas, err := CollectWeightShapesFromConfig(&ArchConfig{ModelDim: D, VocabSize: 1024, SeqLen: 64, Blocks: perBlock})
	if err != nil {
		t.Fatalf("CollectWeightShapesFromConfig per-block: %v", err)
	}
	sharedMetas, err := CollectWeightShapesFromConfig(&ArchConfig{ModelDim: D, VocabSize: 1024, SeqLen: 64, Blocks: shared})
	if err != nil {
		t.Fatalf("CollectWeightShapesFromConfig shared: %v", err)
	}
	gotCut := paramCount(perBlockMetas) - paramCount(sharedMetas)
	if gotCut != wantCut {
		t.Fatalf("param delta=%d want %d", gotCut, wantCut)
	}
}

func TestDebertaRelativeAttentionWeightLayout(t *testing.T) {
	spec := BlockSpec{Type: "plain", Heads: 4, RelativeAttention: RelativeAttentionDebertaP2CC2P, RelativeAttentionWindow: 3}
	n, err := BlockWeightCount(spec, false, false)
	if err != nil {
		t.Fatalf("BlockWeightCount: %v", err)
	}
	if n != 10 {
		t.Fatalf("weight count=%d want 10", n)
	}

	metas, err := blockWeightShapes(spec, 64, 16, 1, 256, DefaultFFNMultiplier, false, false)
	if err != nil {
		t.Fatalf("blockWeightShapes: %v", err)
	}
	wantNames := []string{"norm_scale", "wq", "wk", "wv", "relative_embeddings", "w_pos_key", "w_pos_query", "wo", "ff1", "ff2"}
	if len(metas) != len(wantNames) {
		t.Fatalf("len(metas)=%d want %d", len(metas), len(wantNames))
	}
	for i, want := range wantNames {
		if metas[i].Name != want {
			t.Fatalf("metas[%d].Name=%q want %q", i, metas[i].Name, want)
		}
	}
	if got, want := metas[4].Shape, []int{5, 64}; !sameInts(got, want) {
		t.Fatalf("relative_embeddings shape=%v want %v", got, want)
	}

	rich := BlockSpec{Type: "plain", Heads: 4, RelativeAttention: RelativeAttentionDebertaP2CC2P, QKGain: 1.25, SparseAttnGate: true}
	richCount, err := BlockWeightCount(rich, true, true)
	if err != nil {
		t.Fatalf("BlockWeightCount rich: %v", err)
	}
	if richCount != 15 {
		t.Fatalf("rich count=%d want 15", richCount)
	}
}

func TestDebertaRelativeAttentionParallelResidualWeightCount(t *testing.T) {
	blocks := []BlockSpec{
		{Type: "plain", Heads: 4, RelativeAttention: RelativeAttentionDebertaP2CC2P, ParallelResidual: boolPtr(true)},
		{Type: "swiglu"},
	}
	got, err := CountWeightsWithNgramsRecurrenceAndParallel(64, DefaultFFNMultiplier, false, false, false, false, false, 0, 0, 0, 0, blocks, nil)
	if err != nil {
		t.Fatalf("CountWeightsWithNgramsRecurrenceAndParallel: %v", err)
	}
	if got != 16 {
		t.Fatalf("weight count=%d want 16", got)
	}
}

func TestEmitPlainAttentionIR_DebertaRelativeAttention(t *testing.T) {
	p := NewProgram(10)
	wi, err := emitBlockIR(p, BlockSpec{Type: "plain", Heads: 4, RelativeAttention: RelativeAttentionDebertaP2CC2P, RelativeAttentionWindow: 4}, "x", 0, 64, 8, 2, 256, 0, nil, DefaultFFNMultiplier, false)
	if err != nil {
		t.Fatalf("emitBlockIR: %v", err)
	}
	if wi != 10 {
		t.Fatalf("wi=%d want 10", wi)
	}
	if got := countOps(p, OpDebertaRelativeBias); got != 1 {
		t.Fatalf("DebertaRelativeBias ops=%d want 1", got)
	}
	if got := countOps(p, OpRoPE); got != 0 {
		t.Fatalf("RoPE ops=%d want 0", got)
	}
	if got := countOps(p, OpCausalMask); got != 1 {
		t.Fatalf("CausalMask ops=%d want 1", got)
	}
}

func TestEmitPlainAttentionIR_DebertaSharedQKReuse(t *testing.T) {
	p := NewProgram(8)
	spec := BlockSpec{
		Type:                              "plain",
		Heads:                             4,
		KVHeads:                           2,
		RelativeAttention:                 RelativeAttentionDebertaP2CC2P,
		RelativeAttentionWindow:           4,
		RelativeAttentionParameterization: RelativeAttentionParamSharedQKReuse,
	}
	wi, err := EmitBlock(p, spec, "x", 1, 64, 8, 2, 256, 0, EmitOptions{
		MLPMult:        DefaultFFNMultiplier,
		sharedRelative: sharedRelativeAttentionPlan{Enabled: true, Window: 4, WeightIndex: 0},
	})
	if err != nil {
		t.Fatalf("EmitBlock: %v", err)
	}
	if wi != 8 {
		t.Fatalf("wi=%d want 8", wi)
	}
	if got := countOps(p, OpDebertaRelativeBias); got != 1 {
		t.Fatalf("DebertaRelativeBias ops=%d want 1", got)
	}
	assertMatMulInputsForOutputSuffix(t, p, "_rel_key_flat", []string{"w0", "w3"})
	assertMatMulInputsForOutputSuffix(t, p, "_rel_query_flat", []string{"w0", "w2"})
	for _, op := range p.Ops {
		if op.Code == OpMatMul && len(op.Inputs) == 2 && (op.Inputs[1] == "w4" || op.Inputs[1] == "w5") && len(op.Outputs) > 0 && strings.Contains(op.Outputs[0], "_rel_") {
			t.Fatalf("shared relative path used unexpected per-block projection input %v for output %v", op.Inputs, op.Outputs)
		}
	}
}

func TestEmitPlainAttentionIR_DebertaSharedQKReuseEmbeddingNorm(t *testing.T) {
	p := NewProgram(12)
	spec := BlockSpec{
		Type:                              "plain",
		Heads:                             4,
		KVHeads:                           2,
		RelativeAttention:                 RelativeAttentionDebertaP2CC2P,
		RelativeAttentionWindow:           4,
		RelativeAttentionParameterization: RelativeAttentionParamSharedQKReuse,
	}
	wi, err := EmitBlock(p, spec, "x", 3, 64, 8, 2, 256, 0, EmitOptions{
		MLPMult: DefaultFFNMultiplier,
		sharedRelative: sharedRelativeAttentionPlan{
			Enabled:     true,
			Window:      4,
			WeightIndex: 0,
			Norm:        RelativeAttentionEmbeddingNormLayerNorm,
			NormIndex:   1,
			NormEps:     1e-6,
		},
	})
	if err != nil {
		t.Fatalf("EmitBlock: %v", err)
	}
	if wi != 10 {
		t.Fatalf("wi=%d want 10", wi)
	}
	var normOut string
	for _, op := range p.Ops {
		if op.Code != OpLayerNorm {
			continue
		}
		if len(op.Inputs) == 3 && op.Inputs[0] == "w0" && op.Inputs[1] == "w1" && op.Inputs[2] == "w2" {
			if len(op.FloatParams) != 1 || op.FloatParams[0] != 1e-6 {
				t.Fatalf("shared relative LayerNorm params=%v want eps 1e-6", op.FloatParams)
			}
			normOut = op.Outputs[0]
			break
		}
	}
	if normOut == "" {
		t.Fatalf("missing shared relative LayerNorm op: %+v", p.Ops)
	}
	assertMatMulInputsForOutputSuffix(t, p, "_rel_key_flat", []string{normOut, "w5"})
	assertMatMulInputsForOutputSuffix(t, p, "_rel_query_flat", []string{normOut, "w4"})
}

func TestPlainAttentionPostNormBeforeOutProjWeightAndIROrder(t *testing.T) {
	spec := BlockSpec{Type: "plain", Heads: 4, AttnPostNorm: PlainAttnPostNormBeforeOutProj}
	metas, err := blockWeightShapes(spec, 64, 8, 1, 128, DefaultFFNMultiplier, false, false)
	if err != nil {
		t.Fatalf("blockWeightShapes: %v", err)
	}
	postNormAt := -1
	woAt := -1
	for i, meta := range metas {
		switch meta.Name {
		case "post_attn_norm_scale":
			postNormAt = i
		case "wo":
			woAt = i
		}
	}
	if postNormAt < 0 || woAt < 0 || postNormAt >= woAt {
		t.Fatalf("post_attn_norm_scale index=%d wo index=%d in metas=%+v", postNormAt, woAt, metas)
	}

	p := NewProgram(len(metas))
	_, err = emitBlockIR(p, spec, "x", 0, 64, 8, 1, 128, 0, nil, DefaultFFNMultiplier, false)
	if err != nil {
		t.Fatalf("emitBlockIR: %v", err)
	}
	postNormOut := ""
	postNormOpIndex := -1
	woOpIndex := -1
	for i, op := range p.Ops {
		if op.Code == OpRMSNorm && len(op.Outputs) > 0 && strings.HasSuffix(op.Outputs[0], "_flat_post_norm") {
			postNormOut = op.Outputs[0]
			postNormOpIndex = i
		}
		if op.Code == OpMatMul && len(op.Inputs) == 2 && op.Inputs[1] == weightName(5) {
			woOpIndex = i
			if op.Inputs[0] != postNormOut {
				t.Fatalf("wo MatMul inputs=%v want first input %q", op.Inputs, postNormOut)
			}
		}
	}
	if postNormOpIndex < 0 || woOpIndex < 0 {
		t.Fatalf("missing post norm or wo op in %+v", p.Ops)
	}
	if postNormOpIndex >= woOpIndex {
		t.Fatalf("post norm op index=%d should precede wo index=%d", postNormOpIndex, woOpIndex)
	}
}

func TestEmitPlainAttentionIR_DebertaRelativeBidirectionalNoCausalMask(t *testing.T) {
	p := NewProgram(10)
	_, err := emitBlockIR(p, BlockSpec{
		Type:              "plain",
		Heads:             4,
		AttentionMask:     AttentionMaskBidirectional,
		RelativeAttention: RelativeAttentionDebertaP2CC2P,
	}, "x", 0, 64, 8, 2, 256, 0, nil, DefaultFFNMultiplier, false)
	if err != nil {
		t.Fatalf("emitBlockIR: %v", err)
	}
	if got := countOps(p, OpDebertaRelativeBias); got != 1 {
		t.Fatalf("DebertaRelativeBias ops=%d want 1", got)
	}
	if got := countOps(p, OpCausalMask); got != 0 {
		t.Fatalf("CausalMask ops=%d want 0", got)
	}
}

func paramCount(metas []WeightMeta) int {
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

func assertMatMulInputsForOutputSuffix(t *testing.T, p *Program, suffix string, want []string) {
	t.Helper()
	for _, op := range p.Ops {
		if op.Code != OpMatMul || len(op.Outputs) == 0 || !strings.HasSuffix(op.Outputs[0], suffix) {
			continue
		}
		if len(op.Inputs) != len(want) {
			t.Fatalf("%s inputs=%v want %v", suffix, op.Inputs, want)
		}
		for i := range want {
			if op.Inputs[i] != want[i] {
				t.Fatalf("%s inputs=%v want %v", suffix, op.Inputs, want)
			}
		}
		return
	}
	t.Fatalf("missing OpMatMul output suffix %q", suffix)
}

func sameInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
