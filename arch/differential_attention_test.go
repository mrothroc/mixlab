package arch

import (
	"math"
	"strings"
	"testing"
)

func differentialAttentionTestConfig(blocks []BlockSpec) *ArchConfig {
	return &ArchConfig{
		Name:       "diff-attn-test",
		ModelDim:   12,
		VocabSize:  32,
		SeqLen:     4,
		MLPMult:    1,
		Blocks:     blocks,
		Training:   TrainingSpec{BatchTokens: 8},
		NormType:   NormTypeRMSNorm,
		NormAffine: boolPtr(true),
	}
}

func TestDifferentialAttentionValidConfigAndWeightDelta(t *testing.T) {
	base := BlockSpec{Type: "plain", Heads: 6}
	diff := BlockSpec{Type: "plain", Heads: 3, DifferentialAttention: true}
	if _, err := validateConfig(differentialAttentionTestConfig([]BlockSpec{diff}), "valid"); err != nil {
		t.Fatalf("validate differential config: %v", err)
	}
	baseMetas, err := blockWeightShapes(base, 12, 4, 2, 32, 1, false, false)
	if err != nil {
		t.Fatalf("base weight shapes: %v", err)
	}
	diffMetas, err := blockWeightShapes(diff, 12, 4, 2, 32, 1, false, false)
	if err != nil {
		t.Fatalf("diff weight shapes: %v", err)
	}
	gotDelta := countWeightMetaElements(diffMetas) - countWeightMetaElements(baseMetas)
	if wantDelta := int64(4*2 + 4); gotDelta != wantDelta {
		t.Fatalf("differential param delta=%d want %d", gotDelta, wantDelta)
	}
	wantShapes := map[string][]int{
		"diff_lambda_q1":   {2},
		"diff_lambda_k1":   {2},
		"diff_lambda_q2":   {2},
		"diff_lambda_k2":   {2},
		"diff_subln_scale": {4},
	}
	for name, want := range wantShapes {
		meta, ok := findWeightMeta(diffMetas, name)
		if !ok {
			t.Fatalf("missing weight %s in %+v", name, diffMetas)
		}
		if !intSlicesEqual(meta.Shape, want) {
			t.Fatalf("%s shape=%v want %v", name, meta.Shape, want)
		}
		if strings.HasPrefix(name, "diff_lambda_") && meta.InitMode != diffLambdaInitMode {
			t.Fatalf("%s InitMode=%q want %q", name, meta.InitMode, diffLambdaInitMode)
		}
		if name == "diff_subln_scale" && (!meta.IsNormScale || !meta.InitOne) {
			t.Fatalf("diff_subln_scale should be norm scale init-one: %+v", meta)
		}
	}
}

func TestDifferentialAttentionValidationRejectsUnsupportedCombinations(t *testing.T) {
	tests := []struct {
		name string
		cfg  *ArchConfig
		want string
	}{
		{
			name: "odd head width",
			cfg: func() *ArchConfig {
				cfg := differentialAttentionTestConfig([]BlockSpec{{Type: "plain", Heads: 3, DifferentialAttention: true}})
				cfg.ModelDim = 15
				return cfg
			}(),
			want: "model_dim/heads=5 to be even",
		},
		{
			name: "rope too wide",
			cfg:  differentialAttentionTestConfig([]BlockSpec{{Type: "plain", Heads: 3, DifferentialAttention: true, RopeDims: 4}}),
			want: "differential sub-head dim=2",
		},
		{
			name: "qk norm",
			cfg:  differentialAttentionTestConfig([]BlockSpec{{Type: "plain", Heads: 3, DifferentialAttention: true, QKNorm: true}}),
			want: "qk_norm",
		},
		{
			name: "relative",
			cfg:  differentialAttentionTestConfig([]BlockSpec{{Type: "plain", Heads: 3, DifferentialAttention: true, RelativeAttention: RelativeAttentionDebertaP2CC2P}}),
			want: "relative_attention",
		},
		{
			name: "kv heads",
			cfg:  differentialAttentionTestConfig([]BlockSpec{{Type: "plain", Heads: 3, DifferentialAttention: true, KVHeads: 1}}),
			want: "kv_heads",
		},
		{
			name: "post norm",
			cfg:  differentialAttentionTestConfig([]BlockSpec{{Type: "plain", Heads: 3, DifferentialAttention: true, AttnPostNorm: PlainAttnPostNormBeforeOutProj}}),
			want: "attn_post_norm",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := validateConfig(tt.cfg, tt.name); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("validate error=%v want substring %q", err, tt.want)
			}
		})
	}
}

func TestDifferentialAttentionIRUsesTwoSoftmaxesMasksAndSubmapRoPE(t *testing.T) {
	cfg := differentialAttentionTestConfig([]BlockSpec{{
		Type:                  "plain",
		Heads:                 3,
		DifferentialAttention: true,
		WindowSize:            2,
	}})
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	if got := countOps(prog, OpSoftmax); got != 2 {
		t.Fatalf("softmax ops=%d want 2", got)
	}
	if got := countOps(prog, OpCausalMask); got != 2 {
		t.Fatalf("causal mask ops=%d want 2", got)
	}
	if got := countOps(prog, OpClamp); got != 2 {
		t.Fatalf("lambda clamp ops=%d want 2", got)
	}
	ropeOps := 0
	for _, op := range prog.Ops {
		if op.Code != OpRoPE {
			continue
		}
		ropeOps++
		if len(op.IntParams) < 3 || op.IntParams[1] != 2 || op.IntParams[2] != 0 {
			t.Fatalf("RoPE params=%v want sub-head dim 2 with full-submap rope_dims=0", op.IntParams)
		}
	}
	if ropeOps != 2 {
		t.Fatalf("RoPE ops=%d want 2", ropeOps)
	}
	if !hasRMSNormInput(prog, "diff_ctx") {
		t.Fatalf("missing differential per-head SubLN RMSNorm")
	}
}

func TestDifferentialAttentionBidirectionalIRDoesNotEmitCausalMask(t *testing.T) {
	cfg := differentialAttentionTestConfig([]BlockSpec{{
		Type:                  "plain",
		Heads:                 3,
		DifferentialAttention: true,
		AttentionMask:         AttentionMaskBidirectional,
	}})
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	if got := countOps(prog, OpSoftmax); got != 2 {
		t.Fatalf("softmax ops=%d want 2", got)
	}
	if got := countOps(prog, OpCausalMask); got != 0 {
		t.Fatalf("causal mask ops=%d want 0", got)
	}
}

func TestDifferentialLambdaInitScheduleAndOverride(t *testing.T) {
	first := EffectiveDifferentialLambdaInit(BlockSpec{Type: "plain", Heads: 3, DifferentialAttention: true}, 0)
	wantFirst := 0.8 - 0.6*math.Exp(-0.3)
	if math.Abs(first-wantFirst) > 1e-6 {
		t.Fatalf("first lambda init=%g want %g", first, wantFirst)
	}
	second := EffectiveDifferentialLambdaInit(BlockSpec{Type: "plain", Heads: 3, DifferentialAttention: true}, 1)
	wantSecond := 0.8 - 0.6*math.Exp(-0.6)
	if math.Abs(second-wantSecond) > 1e-6 {
		t.Fatalf("second lambda init=%g want %g", second, wantSecond)
	}
	override := 0.0
	if got := EffectiveDifferentialLambdaInit(BlockSpec{Type: "plain", Heads: 3, DifferentialAttention: true, DifferentialLambdaInit: &override}, 3); got != 0 {
		t.Fatalf("override lambda init=%g want 0", got)
	}
}

func findWeightMeta(metas []WeightMeta, name string) (WeightMeta, bool) {
	for _, meta := range metas {
		if meta.Name == name {
			return meta, true
		}
	}
	return WeightMeta{}, false
}

func hasRMSNormInput(prog *Program, contains string) bool {
	for _, op := range prog.Ops {
		if op.Code != OpRMSNorm || len(op.Inputs) < 2 {
			continue
		}
		if strings.Contains(op.Inputs[0], contains) {
			return true
		}
	}
	return false
}
