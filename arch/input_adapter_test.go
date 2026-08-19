package arch

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func linearFramesTestConfig() ArchConfig {
	bias := true
	return ArchConfig{
		Name:                "continuous_classifier",
		ModelDim:            8,
		SeqLen:              4,
		PositionalEmbedding: PositionalEmbeddingNone,
		InputAdapter: &InputAdapterSpec{
			Kind: InputAdapterLinearFrames, FeatureDim: 1, Bias: &bias, Norm: InputAdapterNormLayerNorm,
		},
		Blocks: []BlockSpec{{Type: "plain", Heads: 2, AttentionMask: AttentionMaskBidirectional}},
		Training: TrainingSpec{
			Objective: ObjectiveClassification, BatchTokens: 8, Steps: 2, LR: 1e-3,
			Classification: &ClassificationSpec{NumLabels: 3, Pooling: ClassificationPoolingMean},
		},
	}
}

func parseInputAdapterTestConfig(t *testing.T, cfg ArchConfig) *ArchConfig {
	t.Helper()
	raw, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseArchConfig(raw, cfg.Name)
	if err != nil {
		t.Fatal(err)
	}
	return parsed
}

func TestLinearFramesConfigAndValidation(t *testing.T) {
	cfg := parseInputAdapterTestConfig(t, linearFramesTestConfig())
	if !cfg.LinearFramesEnabled() || cfg.InputFeatureDim() != 1 {
		t.Fatalf("adapter kind=%q feature_dim=%d", cfg.EffectiveInputAdapterKind(), cfg.InputFeatureDim())
	}
	if !cfg.EffectiveInputAdapterBias() || cfg.EffectiveInputAdapterNorm() != InputAdapterNormLayerNorm {
		t.Fatalf("adapter bias=%t norm=%q", cfg.EffectiveInputAdapterBias(), cfg.EffectiveInputAdapterNorm())
	}

	tests := []struct {
		name string
		edit func(*ArchConfig)
		want string
	}{
		{"missing feature dim", func(c *ArchConfig) { c.InputAdapter.FeatureDim = 0 }, "feature_dim"},
		{"bad norm", func(c *ArchConfig) { c.InputAdapter.Norm = "rmsnorm" }, "input_adapter.norm"},
		{"vocabulary", func(c *ArchConfig) { c.VocabSize = 16 }, "vocab_size"},
		{"causal objective", func(c *ArchConfig) {
			c.Training.Objective = ObjectiveCausal
			c.Training.Classification = nil
		}, "classification"},
		{"tied embeddings", func(c *ArchConfig) { c.TieEmbeddings = true }, "tie_embeddings"},
		{"token features", func(c *ArchConfig) { c.BigramVocabSize = 8 }, "feature channels"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			candidate := linearFramesTestConfig()
			tt.edit(&candidate)
			raw, _ := json.Marshal(candidate)
			if _, err := ParseArchConfig(raw, tt.name); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error=%v, want substring %q", err, tt.want)
			}
		})
	}
}

func TestLinearFramesLowDimLayerNormWarning(t *testing.T) {
	tests := []struct {
		name       string
		featureDim int
		norm       string
		wantWarn   bool
	}{
		{name: "one feature", featureDim: 1, norm: InputAdapterNormLayerNorm, wantWarn: true},
		{name: "warning boundary", featureDim: 4, norm: "layer_norm", wantWarn: true},
		{name: "above warning boundary", featureDim: 5, norm: InputAdapterNormLayerNorm},
		{name: "explicit none", featureDim: 1, norm: InputAdapterNormNone},
		{name: "omitted defaults to none", featureDim: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			candidate := linearFramesTestConfig()
			candidate.InputAdapter.FeatureDim = tt.featureDim
			candidate.InputAdapter.Norm = tt.norm
			raw, err := json.Marshal(candidate)
			if err != nil {
				t.Fatal(err)
			}
			cfg, stderr := parseConfigCapturingStderr(t, raw, tt.name)
			if got := strings.Contains(stderr, "post-projection LayerNorm can discard input magnitude"); got != tt.wantWarn {
				t.Fatalf("warning=%t want=%t stderr=%q", got, tt.wantWarn, stderr)
			}
			if tt.wantWarn {
				if !strings.Contains(stderr, fmt.Sprintf("feature_dim=%d", tt.featureDim)) ||
					!strings.Contains(stderr, `Prefer norm="none"`) {
					t.Fatalf("warning lacks actionable config details: %q", stderr)
				}
			}
			if cfg.EffectiveInputAdapterNorm() != normalizeInputAdapterNorm(tt.norm) {
				t.Fatalf("norm=%q want=%q", cfg.EffectiveInputAdapterNorm(), normalizeInputAdapterNorm(tt.norm))
			}
		})
	}
}

// TestLoadArchConfigQuietSuppressesInputAdapterWarning locks in the fix for the
// CLI double-parse: the internal preflight (LoadArchConfigQuiet) must stay
// silent while the primary load (LoadArchConfig) surfaces the warning exactly
// once, so a footgun config warns once per invocation, not twice.
func TestLoadArchConfigQuietSuppressesInputAdapterWarning(t *testing.T) {
	candidate := linearFramesTestConfig()
	candidate.InputAdapter.FeatureDim = 1
	candidate.InputAdapter.Norm = InputAdapterNormLayerNorm
	raw, err := json.Marshal(candidate)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "footgun.json")
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}

	const marker = "discard input magnitude"
	quiet := captureStderr(t, func() {
		if _, err := LoadArchConfigQuiet(path); err != nil {
			t.Fatalf("LoadArchConfigQuiet: %v", err)
		}
	})
	if strings.Contains(quiet, marker) {
		t.Fatalf("LoadArchConfigQuiet emitted a warning: %q", quiet)
	}
	loud := captureStderr(t, func() {
		if _, err := LoadArchConfig(path); err != nil {
			t.Fatalf("LoadArchConfig: %v", err)
		}
	})
	if got := strings.Count(loud, marker); got != 1 {
		t.Fatalf("LoadArchConfig warning count=%d want=1 stderr=%q", got, loud)
	}
}

func TestTokenEmbeddingAdapterOmissionPreservesIRAndWeights(t *testing.T) {
	base := classificationTestConfig(BlockSpec{Type: "plain", Heads: 2})
	omitted := parseClassificationTestConfig(t, base)
	explicitConfig := base
	explicitConfig.InputAdapter = &InputAdapterSpec{Kind: InputAdapterTokenEmbedding}
	explicit := parseClassificationTestConfig(t, explicitConfig)

	omittedWeights, err := CollectWeightShapesFromConfig(omitted)
	if err != nil {
		t.Fatal(err)
	}
	explicitWeights, err := CollectWeightShapesFromConfig(explicit)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(omittedWeights, explicitWeights) {
		t.Fatalf("explicit token adapter changed weights:\nomitted=%+v\nexplicit=%+v", omittedWeights, explicitWeights)
	}
	omittedIR, err := BuildIRProgramFromConfig(omitted)
	if err != nil {
		t.Fatal(err)
	}
	explicitIR, err := BuildIRProgramFromConfig(explicit)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(omittedIR, explicitIR) {
		t.Fatal("explicit token_embedding changed the legacy IR")
	}
}

func TestLinearFramesWeightsIRAndCounts(t *testing.T) {
	cfg := parseInputAdapterTestConfig(t, linearFramesTestConfig())
	weights, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	wantPrefix := []WeightMeta{
		{Name: "input_adapter_proj", Shape: []int{1, 8}, PyTorchLinearFanIn: 1},
		{Name: "final_norm", Shape: []int{8}, IsNormScale: true, InitOne: true},
		{Name: "input_adapter_bias", Shape: []int{8}, InitZero: true, PyTorchLinearFanIn: 1},
		{Name: "input_adapter_norm_scale", Shape: []int{8}, IsNormScale: true, InitOne: true},
		{Name: "input_adapter_norm_bias", Shape: []int{8}, InitZero: true},
	}
	if len(weights) < len(wantPrefix)+2 {
		t.Fatalf("weights=%d, want at least %d", len(weights), len(wantPrefix)+2)
	}
	for i, want := range wantPrefix {
		if !reflect.DeepEqual(weights[i], want) {
			t.Fatalf("weight[%d]=%+v, want %+v", i, weights[i], want)
		}
	}
	if got := weights[len(weights)-2]; got.Name != "head_classifier_proj" || !reflect.DeepEqual(got.Shape, []int{8, 3}) || got.PyTorchLinearFanIn != 8 {
		t.Fatalf("classifier projection=%+v", got)
	}
	for _, weight := range weights {
		if weight.Name == "embed" || weight.Name == "head" {
			t.Fatalf("continuous config retained token/LM weight %+v", weight)
		}
	}

	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !programHasInput(prog, "continuous_frames") || programHasInput(prog, "tokens") || programHasInput(prog, "targets") {
		t.Fatalf("continuous IR inputs=%+v", prog.Inputs)
	}
	if prog.NumWeights != len(weights) {
		t.Fatalf("program weights=%d metadata=%d", prog.NumWeights, len(weights))
	}
	var sawProjection, sawAdapterNorm bool
	for _, op := range prog.Ops {
		if op.Code == OpMatMul && reflect.DeepEqual(op.Inputs, []string{"continuous_frames", weightName(0)}) {
			sawProjection = true
		}
		if op.Code == OpLayerNorm && len(op.Outputs) == 1 && op.Outputs[0] == "x_frame_norm" {
			sawAdapterNorm = true
		}
	}
	if !sawProjection || !sawAdapterNorm {
		t.Fatalf("projection=%t adapter_norm=%t ops=%+v", sawProjection, sawAdapterNorm, prog.Ops)
	}

	params, expanded, err := ParameterCountsFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	wantParams := countWeightMetaElements(weights)
	if params != wantParams || expanded != wantParams {
		t.Fatalf("params=%d expanded=%d want=%d", params, expanded, wantParams)
	}
	if flops := EstimateFLOPs(cfg); flops.ForwardFLOPs <= 0 || flops.ParamCount != wantParams {
		t.Fatalf("flops=%+v", flops)
	}
}

func TestLinearFramesNormalizesBeforeTopLevelPositions(t *testing.T) {
	candidate := linearFramesTestConfig()
	candidate.PositionalEmbedding = PositionalEmbeddingLearnedAbsolute
	candidate.MaxPositions = candidate.SeqLen
	cfg := parseInputAdapterTestConfig(t, candidate)
	weights, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	wantNames := []string{
		"input_adapter_proj", "final_norm", "input_adapter_bias",
		"input_adapter_norm_scale", "input_adapter_norm_bias", "position_embeddings",
	}
	for i, want := range wantNames {
		if i >= len(weights) || weights[i].Name != want {
			t.Fatalf("weight[%d]=%+v, want name %q; weights=%+v", i, weights[i], want, weights)
		}
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	normOp, positionAddOp := -1, -1
	for i, op := range prog.Ops {
		if len(op.Outputs) == 1 && op.Outputs[0] == "x_frame_norm" {
			normOp = i
		}
		if len(op.Outputs) == 1 && op.Outputs[0] == "x_embed_pos" {
			positionAddOp = i
		}
	}
	if normOp < 0 || positionAddOp < 0 || normOp >= positionAddOp {
		t.Fatalf("adapter norm op=%d position add op=%d; ops=%+v", normOp, positionAddOp, prog.Ops)
	}
}

func discreteCodebookTestConfig() ArchConfig {
	bias := false
	return ArchConfig{
		Name:                "codebook_classifier",
		ModelDim:            6,
		SeqLen:              4,
		PositionalEmbedding: PositionalEmbeddingNone,
		InputAdapter: &InputAdapterSpec{
			Kind: InputAdapterDiscreteCodebooks, NumCodebooks: 2, CodebookVocabSize: 8,
			Fusion: InputAdapterFusionAttentionMLP, FusionHiddenDim: 6, Norm: InputAdapterNormNone,
		},
		Blocks: []BlockSpec{{Type: "swiglu"}},
		Training: TrainingSpec{
			Objective: ObjectiveClassification, BatchTokens: 8, Steps: 2, LR: 1e-3,
			Classification: &ClassificationSpec{NumLabels: 3, Pooling: ClassificationPoolingMean, Bias: &bias},
		},
	}
}

func TestDiscreteCodebooksConfigValidation(t *testing.T) {
	cfg := parseInputAdapterTestConfig(t, discreteCodebookTestConfig())
	if !cfg.DiscreteCodebooksEnabled() || cfg.CodebookEmbeddingRows() != 16 ||
		cfg.EffectiveCodebookFusion() != InputAdapterFusionAttentionMLP || cfg.EffectiveCodebookFusionHiddenDim() != 6 {
		t.Fatalf("resolved adapter=%+v rows=%d fusion=%q hidden=%d", cfg.InputAdapter, cfg.CodebookEmbeddingRows(), cfg.EffectiveCodebookFusion(), cfg.EffectiveCodebookFusionHiddenDim())
	}

	tests := []struct {
		name string
		edit func(*ArchConfig)
		want string
	}{
		{"codebooks", func(c *ArchConfig) { c.InputAdapter.NumCodebooks = 0 }, "num_codebooks"},
		{"domain", func(c *ArchConfig) { c.InputAdapter.CodebookVocabSize = 1 }, "codebook_vocab_size"},
		{"overflow", func(c *ArchConfig) { c.InputAdapter.NumCodebooks = math.MaxInt32; c.InputAdapter.CodebookVocabSize = 2 }, "int32"},
		{"fusion", func(c *ArchConfig) { c.InputAdapter.Fusion = "concat" }, "fusion"},
		{"mean hidden", func(c *ArchConfig) { c.InputAdapter.Fusion = "mean"; c.InputAdapter.FusionHiddenDim = 4 }, "fusion_hidden_dim"},
		{"feature dim", func(c *ArchConfig) { c.InputAdapter.FeatureDim = 3 }, "feature_dim"},
		{"bias", func(c *ArchConfig) { value := true; c.InputAdapter.Bias = &value }, "bias"},
		{"vocabulary", func(c *ArchConfig) { c.VocabSize = 16 }, "vocab_size"},
		{"causal", func(c *ArchConfig) { c.Training.Objective = ObjectiveCausal; c.Training.Classification = nil }, "classification"},
		{"token features", func(c *ArchConfig) { c.CharVocabSize = 257 }, "feature channels"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			candidate := discreteCodebookTestConfig()
			tt.edit(&candidate)
			raw, _ := json.Marshal(candidate)
			if _, err := ParseArchConfig(raw, tt.name); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error=%v, want substring %q", err, tt.want)
			}
		})
	}
}

func TestDiscreteCodebooksWeightsIRAndCounts(t *testing.T) {
	cfg := parseInputAdapterTestConfig(t, discreteCodebookTestConfig())
	weights, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	wantAdapter := map[string]WeightMeta{
		"input_adapter_codebook_embedding": {Name: "input_adapter_codebook_embedding", Shape: []int{16, 6}, InitMode: "torch_embedding_normal_1"},
		"input_adapter_codebook_attn_w1":   {Name: "input_adapter_codebook_attn_w1", Shape: []int{6, 6}, InitMode: "torch_linear_uniform"},
		"input_adapter_codebook_attn_b1":   {Name: "input_adapter_codebook_attn_b1", Shape: []int{6}, InitMode: "torch_linear_bias_uniform", PyTorchLinearFanIn: 6},
		"input_adapter_codebook_attn_w2":   {Name: "input_adapter_codebook_attn_w2", Shape: []int{6, 1}, InitMode: "torch_linear_uniform"},
	}
	for name, want := range wantAdapter {
		found := false
		for _, weight := range weights {
			if weight.Name == name {
				found = true
				if !reflect.DeepEqual(weight, want) {
					t.Fatalf("weight %q=%+v want=%+v", name, weight, want)
				}
			}
		}
		if !found {
			t.Fatalf("missing %q in %+v", name, weights)
		}
	}
	var adapterParams int64
	for _, weight := range weights {
		if strings.HasPrefix(weight.Name, "input_adapter_codebook_") {
			adapterParams += countWeightMetaElements([]WeightMeta{weight})
		}
	}
	// Q*V*D embedding + D*H + H first linear + H second linear.
	if want := int64(2*8*6 + 6*6 + 6 + 6); adapterParams != want {
		t.Fatalf("adapter params=%d want=%d", adapterParams, want)
	}
	if got := weights[len(weights)-1]; got.Name != "head_classifier_proj" || !reflect.DeepEqual(got.Shape, []int{6, 3}) {
		t.Fatalf("bias-free classifier tail=%+v", got)
	}
	for _, weight := range weights {
		if weight.Name == "head_classifier_bias" || weight.Name == "embed" || weight.Name == "head" {
			t.Fatalf("unexpected weight %+v", weight)
		}
	}

	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !programHasInput(prog, "codebook_tokens") || programHasInput(prog, "tokens") || programHasInput(prog, "continuous_frames") {
		t.Fatalf("codebook IR inputs=%+v", prog.Inputs)
	}
	var offset, codebookSoftmax, fusionMatMul bool
	for _, op := range prog.Ops {
		if op.Code == OpCodebookOffset && reflect.DeepEqual(op.IntParams, []int{2, 8}) {
			offset = true
		}
		if op.Code == OpSoftmax && len(op.Outputs) == 1 && op.Outputs[0] == "codebook_attn_weights" && reflect.DeepEqual(op.IntParams, []int{-1}) {
			codebookSoftmax = true
		}
		if op.Code == OpMatMul && reflect.DeepEqual(op.Inputs, []string{"codebook_attn_weights_b1q", "codebook_embeddings_bqd"}) {
			fusionMatMul = true
		}
	}
	if !offset || !codebookSoftmax || !fusionMatMul {
		t.Fatalf("offset=%t softmax=%t fusion_matmul=%t", offset, codebookSoftmax, fusionMatMul)
	}
	params, expanded, err := ParameterCountsFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	wantParams := countWeightMetaElements(weights)
	if params != wantParams || expanded != wantParams {
		t.Fatalf("params=%d expanded=%d want=%d", params, expanded, wantParams)
	}
}

func TestDiscreteCodebooksMeanFusionHasNoAttentionWeights(t *testing.T) {
	candidate := discreteCodebookTestConfig()
	candidate.InputAdapter.Fusion = InputAdapterFusionMean
	candidate.InputAdapter.FusionHiddenDim = 0
	cfg := parseInputAdapterTestConfig(t, candidate)
	weights, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, weight := range weights {
		if strings.Contains(weight.Name, "codebook_attn") {
			t.Fatalf("mean fusion retained attention weight %+v", weight)
		}
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, op := range prog.Ops {
		if op.Code == OpMeanAxis && reflect.DeepEqual(op.Inputs, []string{"codebook_embeddings"}) && reflect.DeepEqual(op.IntParams, []int{2}) {
			found = true
		}
	}
	if !found {
		t.Fatal("mean fusion did not reduce codebook axis 2")
	}
}
