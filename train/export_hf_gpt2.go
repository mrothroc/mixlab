package train

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
)

type hfGPT2ConfigJSON struct {
	ModelType                  string   `json:"model_type"`
	Architectures              []string `json:"architectures"`
	VocabSize                  int      `json:"vocab_size"`
	NPositions                 int      `json:"n_positions"`
	NCtx                       int      `json:"n_ctx"`
	NEmbd                      int      `json:"n_embd"`
	NLayer                     int      `json:"n_layer"`
	NHead                      int      `json:"n_head"`
	NInner                     int      `json:"n_inner"`
	ActivationFunction         string   `json:"activation_function"`
	ResidPDrop                 float32  `json:"resid_pdrop"`
	EmbPDrop                   float32  `json:"embd_pdrop"`
	AttnPDrop                  float32  `json:"attn_pdrop"`
	LayerNormEpsilon           float32  `json:"layer_norm_epsilon"`
	InitializerRange           float32  `json:"initializer_range"`
	ScaleAttnWeights           bool     `json:"scale_attn_weights"`
	ScaleAttnByInverseLayerIdx bool     `json:"scale_attn_by_inverse_layer_idx"`
	ReorderAndUpcastAttn       bool     `json:"reorder_and_upcast_attn"`
	TieWordEmbeddings          bool     `json:"tie_word_embeddings"`
	PadTokenID                 *int     `json:"pad_token_id,omitempty"`
	EOSTokenID                 *int     `json:"eos_token_id,omitempty"`
	BOSTokenID                 *int     `json:"bos_token_id,omitempty"`
	UNKTokenID                 *int     `json:"unk_token_id,omitempty"`
}

type hfGPT2Tensor struct {
	Mixlab string
	HF     string
	Shape  []int
	Data   []float32
}

func runExportHFGPT2(opts ExportHFOptions, cfg *ArchConfig) error {
	if err := validateHFGPT2ExportConfig(cfg); err != nil {
		return err
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		return fmt.Errorf("compute GPT-2 weight shapes: %w", err)
	}
	weights, err := loadSafetensorsWeights(opts.SafetensorsLoad, shapes)
	if err != nil {
		return fmt.Errorf("load safetensors %q: %w", opts.SafetensorsLoad, err)
	}
	tensors, err := materializeHFGPT2Weights(cfg, shapes, weights)
	if err != nil {
		return err
	}
	tokenizer, err := resolveHFTokenizerSource(opts.TokenizerSource, opts.ConfigPath, opts.SafetensorsLoad)
	if err != nil {
		return err
	}
	specials, err := deriveHFTokenizerSpecials(tokenizer, cfg)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(opts.OutputDir, 0o755); err != nil {
		return fmt.Errorf("create HF output directory %q: %w", opts.OutputDir, err)
	}
	if err := writeHFGPT2Config(filepath.Join(opts.OutputDir, "config.json"), cfg, specials); err != nil {
		return err
	}
	if err := writeHFGPT2Safetensors(filepath.Join(opts.OutputDir, "model.safetensors"), cfg, tensors); err != nil {
		return err
	}
	if err := writeJSONFile(filepath.Join(opts.OutputDir, "weight_map.json"), hfGPT2WeightMap(tensors)); err != nil {
		return err
	}
	if err := writeHFTokenizerArtifacts(opts.OutputDir, tokenizer, specials); err != nil {
		return err
	}
	fmt.Printf("exported native GPT-2 Hugging Face model to %s (%d tensors)\n", opts.OutputDir, len(tensors))
	return nil
}

func validateHFGPT2ExportConfig(cfg *ArchConfig) error {
	if cfg == nil {
		return unsupportedHFExport("hf_export_format", "nil config")
	}
	if cfg.EffectiveHFExportFormat() != "gpt2" {
		return unsupportedHFExport("hf_export_format", "native GPT-2 export requires hf_export_format=\"gpt2\"")
	}
	if cfg.Training.EffectiveObjective() != "causal" {
		return unsupportedHFExport("training.objective", "native GPT-2 export requires causal objective")
	}
	if cfg.EffectivePositionalEmbedding() != "learned_absolute" {
		return unsupportedHFExport("positional_embedding", "native GPT-2 export requires learned_absolute position embeddings")
	}
	if !cfg.TieEmbeddings {
		return unsupportedHFExport("tie_embeddings", "native GPT-2 export requires tied embeddings")
	}
	norm := cfg.EffectiveNormSpec()
	if strings.ToLower(strings.TrimSpace(norm.Type)) != "layernorm" || !norm.Affine {
		return unsupportedHFExport("norm", "native GPT-2 export requires affine layernorm")
	}
	if cfg.EffectiveNormPlacement() != "pre" {
		return unsupportedHFExport("norm_placement", "native GPT-2 export requires pre-norm blocks")
	}
	if cfg.FFNInternalNorm {
		return unsupportedHFExport("ffn_internal_norm", "native GPT-2 export requires no internal FFN norm")
	}
	if cfg.BlockScales || cfg.ResidMix || cfg.ParallelResidual || cfg.UNet {
		return unsupportedHFExport("architecture", "native GPT-2 export does not support Mixlab residual/layout extensions")
	}
	if cfg.CharVocabSize > 0 || cfg.BigramVocabSize > 0 || cfg.TrigramVocabSize > 0 || cfg.SmearEmbeddings {
		return unsupportedHFExport("feature_embeddings", "native GPT-2 export requires only token plus learned position embeddings")
	}
	if cfg.MTP != nil || cfg.Backout != nil || len(cfg.Recurrence) > 0 || len(cfg.RecurrencePhases) > 0 {
		return unsupportedHFExport("training_graph", "native GPT-2 export requires a plain sequential graph")
	}
	if cfg.Training.FirstByteMask || cfg.Training.Distillation != nil || cfg.Training.Data2VecActive() || cfg.Training.UsesBlockDiffusionObjective() {
		return unsupportedHFExport("training", "native GPT-2 export does not include training-only objectives or losses")
	}
	if cfg.EffectiveLayerAggregation() != "none" {
		return unsupportedHFExport("layer_aggregation", "native GPT-2 export requires no layer aggregation")
	}
	if cfg.EffectiveMLMHead() != "linear" {
		return unsupportedHFExport("mlm_head", "native GPT-2 export requires the standard tied LM head")
	}
	if cfg.LogitSoftcap != 0 {
		return unsupportedHFExport("logit_softcap", "native GPT-2 export does not support logit softcap")
	}
	if len(cfg.Blocks) == 0 {
		return unsupportedHFExport("blocks", "native GPT-2 export requires at least one plain block")
	}
	for i, block := range cfg.Blocks {
		field := fmt.Sprintf("blocks[%d]", i)
		if strings.ToLower(strings.TrimSpace(block.Type)) != "plain" {
			return unsupportedHFExport(field+".type", "native GPT-2 export supports only plain blocks")
		}
		if block.Heads <= 0 {
			return unsupportedHFExport(field+".heads", "heads must be > 0")
		}
		if block.KVHeads > 0 && block.KVHeads != block.Heads {
			return unsupportedHFExport(field+".kv_heads", "native GPT-2 export requires full multi-head attention")
		}
		mask := strings.ToLower(strings.TrimSpace(block.AttentionMask))
		if mask != "" && mask != "causal" {
			return unsupportedHFExport(field+".attention_mask", "native GPT-2 export requires causal attention")
		}
		if !block.AttnBias {
			return unsupportedHFExport(field+".attn_bias", "native GPT-2 export requires attention projection biases")
		}
		if !block.FFNBias {
			return unsupportedHFExport(field+".ffn_bias", "native GPT-2 export requires FFN biases")
		}
		if !block.FFNPreNorm {
			return unsupportedHFExport(field+".ffn_pre_norm", "native GPT-2 export requires the GPT-2 second pre-FFN norm")
		}
		switch hfPlainFFNActivation(block) {
		case "gelu_new", "gelu":
		default:
			return unsupportedHFExport(field+".ffn_activation", "native GPT-2 export requires gelu_new or gelu")
		}
		if block.RopeDims != 0 || strings.TrimSpace(block.RopeConvention) != "" {
			return unsupportedHFExport(field+".rope", "native GPT-2 export requires no RoPE fields")
		}
		if relativeAttentionEnabledForHF(block) {
			return unsupportedHFExport(field+".relative_attention", "native GPT-2 export requires learned absolute positions only")
		}
		if block.WindowSize > 0 || block.KVSource > 0 || block.SkipAttention || block.QKNorm || block.QKGain != 0 || block.XSA || block.SparseAttnGate || block.AttnValueGate {
			return unsupportedHFExport(field, "native GPT-2 export does not support Mixlab attention extras")
		}
		if hfNormalizePlainAttnPostNorm(block.AttnPostNorm) != "inherit" {
			return unsupportedHFExport(field+".attn_post_norm", "native GPT-2 export requires no attention post norm")
		}
		if block.ParallelResidual != nil || strings.TrimSpace(block.WeightGroup) != "" {
			return unsupportedHFExport(field, "native GPT-2 export requires a simple sequential block")
		}
	}
	return nil
}

func writeHFGPT2Config(path string, cfg *ArchConfig, specials hfTokenizerSpecials) error {
	first := cfg.Blocks[0]
	doc := hfGPT2ConfigJSON{
		ModelType:                  "gpt2",
		Architectures:              []string{"GPT2LMHeadModel"},
		VocabSize:                  cfg.VocabSize,
		NPositions:                 cfg.EffectiveMaxPositions(),
		NCtx:                       cfg.EffectiveMaxPositions(),
		NEmbd:                      cfg.ModelDim,
		NLayer:                     len(cfg.Blocks),
		NHead:                      first.Heads,
		NInner:                     gpt2FFNDim(cfg),
		ActivationFunction:         hfPlainFFNActivation(first),
		ResidPDrop:                 cfg.EffectiveHiddenDropout(),
		EmbPDrop:                   cfg.EffectiveEmbeddingDropout(),
		AttnPDrop:                  cfg.EffectiveAttnDropout(),
		LayerNormEpsilon:           cfg.EffectiveNormSpec().Eps,
		InitializerRange:           cfg.Training.WeightInitStd,
		ScaleAttnWeights:           true,
		ScaleAttnByInverseLayerIdx: false,
		ReorderAndUpcastAttn:       false,
		TieWordEmbeddings:          true,
		PadTokenID:                 specialTokenIDPtr(specials.Pad),
		EOSTokenID:                 specialTokenIDPtr(specials.EOS),
		BOSTokenID:                 specialTokenIDPtr(specials.BOS),
		UNKTokenID:                 specialTokenIDPtr(specials.UNK),
	}
	if doc.InitializerRange == 0 {
		doc.InitializerRange = 0.02
	}
	return writeJSONFile(path, doc)
}

func gpt2FFNDim(cfg *ArchConfig) int {
	ffn := int(math.Round(float64(cfg.ModelDim) * cfg.EffectiveMLPMult()))
	if ffn < cfg.ModelDim {
		return cfg.ModelDim
	}
	return ffn
}

func materializeHFGPT2Weights(cfg *ArchConfig, shapes []WeightShape, weights [][]float32) ([]hfGPT2Tensor, error) {
	if len(shapes) != len(weights) {
		return nil, fmt.Errorf("weight shape/data count mismatch: shapes=%d weights=%d", len(shapes), len(weights))
	}
	var out []hfGPT2Tensor
	cur := 0
	expect := func(name string, shape []int) ([]float32, error) {
		if cur >= len(shapes) {
			return nil, fmt.Errorf("GPT-2 export exhausted weights while expecting %q", name)
		}
		got := shapes[cur]
		if got.Name != name {
			return nil, fmt.Errorf("GPT-2 export expected weight %q at index %d, got %q", name, cur, got.Name)
		}
		if !sameIntSlice(got.Shape, shape) {
			return nil, fmt.Errorf("GPT-2 export weight %q shape=%v want %v", name, got.Shape, shape)
		}
		data := weights[cur]
		cur++
		return data, nil
	}
	add := func(mixlab, hf string, shape []int, data []float32) {
		out = append(out, hfGPT2Tensor{
			Mixlab: mixlab,
			HF:     hf,
			Shape:  append([]int(nil), shape...),
			Data:   data,
		})
	}
	D := cfg.ModelDim
	V := cfg.VocabSize
	ffn := gpt2FFNDim(cfg)
	embed, err := expect("embed", []int{V, D})
	if err != nil {
		return nil, err
	}
	finalScale, err := expect("final_norm_scale", []int{D})
	if err != nil {
		return nil, err
	}
	finalBias, err := expect("final_norm_bias", []int{D})
	if err != nil {
		return nil, err
	}
	wpe, err := expect("position_embeddings", []int{cfg.EffectiveMaxPositions(), D})
	if err != nil {
		return nil, err
	}
	add("embed", "transformer.wte.weight", []int{V, D}, embed)
	add("position_embeddings", "transformer.wpe.weight", []int{cfg.EffectiveMaxPositions(), D}, wpe)
	for blockIdx := range cfg.Blocks {
		prefix := fmt.Sprintf("transformer.h.%d.", blockIdx)
		ln1Scale, err := expect("norm_scale", []int{D})
		if err != nil {
			return nil, err
		}
		ln1Bias, err := expect("norm_bias", []int{D})
		if err != nil {
			return nil, err
		}
		wq, err := expect("wq", []int{D, D})
		if err != nil {
			return nil, err
		}
		wqBias, err := expect("wq_bias", []int{D})
		if err != nil {
			return nil, err
		}
		wk, err := expect("wk", []int{D, D})
		if err != nil {
			return nil, err
		}
		wkBias, err := expect("wk_bias", []int{D})
		if err != nil {
			return nil, err
		}
		wv, err := expect("wv", []int{D, D})
		if err != nil {
			return nil, err
		}
		wvBias, err := expect("wv_bias", []int{D})
		if err != nil {
			return nil, err
		}
		wo, err := expect("wo", []int{D, D})
		if err != nil {
			return nil, err
		}
		woBias, err := expect("wo_bias", []int{D})
		if err != nil {
			return nil, err
		}
		ln2Scale, err := expect("ffn_norm_scale", []int{D})
		if err != nil {
			return nil, err
		}
		ln2Bias, err := expect("ffn_norm_bias", []int{D})
		if err != nil {
			return nil, err
		}
		ff1, err := expect("ff1", []int{D, ffn})
		if err != nil {
			return nil, err
		}
		ff1Bias, err := expect("ff1_bias", []int{ffn})
		if err != nil {
			return nil, err
		}
		ff2, err := expect("ff2", []int{ffn, D})
		if err != nil {
			return nil, err
		}
		ff2Bias, err := expect("ff2_bias", []int{D})
		if err != nil {
			return nil, err
		}
		add("norm_scale", prefix+"ln_1.weight", []int{D}, ln1Scale)
		add("norm_bias", prefix+"ln_1.bias", []int{D}, ln1Bias)
		add("wq,wk,wv", prefix+"attn.c_attn.weight", []int{D, 3 * D}, concatMatrixColumns(D, D, wq, wk, wv))
		add("wq_bias,wk_bias,wv_bias", prefix+"attn.c_attn.bias", []int{3 * D}, concatVectors(wqBias, wkBias, wvBias))
		add("wo", prefix+"attn.c_proj.weight", []int{D, D}, wo)
		add("wo_bias", prefix+"attn.c_proj.bias", []int{D}, woBias)
		add("ffn_norm_scale", prefix+"ln_2.weight", []int{D}, ln2Scale)
		add("ffn_norm_bias", prefix+"ln_2.bias", []int{D}, ln2Bias)
		add("ff1", prefix+"mlp.c_fc.weight", []int{D, ffn}, ff1)
		add("ff1_bias", prefix+"mlp.c_fc.bias", []int{ffn}, ff1Bias)
		add("ff2", prefix+"mlp.c_proj.weight", []int{ffn, D}, ff2)
		add("ff2_bias", prefix+"mlp.c_proj.bias", []int{D}, ff2Bias)
	}
	add("final_norm_scale", "transformer.ln_f.weight", []int{D}, finalScale)
	add("final_norm_bias", "transformer.ln_f.bias", []int{D}, finalBias)
	add("embed", "lm_head.weight", []int{V, D}, embed)
	if cur != len(shapes) {
		return nil, fmt.Errorf("GPT-2 export did not consume all weights: next=%d total=%d", cur, len(shapes))
	}
	return out, nil
}

func writeHFGPT2Safetensors(path string, cfg *ArchConfig, tensors []hfGPT2Tensor) error {
	mapping := make([]hfWeightMapping, len(tensors))
	weights := make([][]float32, len(tensors))
	for i, t := range tensors {
		mapping[i] = hfWeightMapping{
			Mixlab: t.Mixlab,
			HF:     t.HF,
			Shape:  append([]int(nil), t.Shape...),
		}
		weights[i] = t.Data
	}
	return writeHFSafetensors(path, cfg, mapping, weights)
}

func hfGPT2WeightMap(tensors []hfGPT2Tensor) []hfWeightMapping {
	out := make([]hfWeightMapping, len(tensors))
	for i, t := range tensors {
		out[i] = hfWeightMapping{
			Mixlab: t.Mixlab,
			HF:     t.HF,
			Shape:  append([]int(nil), t.Shape...),
		}
	}
	return out
}

func concatMatrixColumns(rows, cols int, mats ...[]float32) []float32 {
	out := make([]float32, rows*cols*len(mats))
	for r := 0; r < rows; r++ {
		dst := r * cols * len(mats)
		for m, mat := range mats {
			copy(out[dst+m*cols:dst+(m+1)*cols], mat[r*cols:(r+1)*cols])
		}
	}
	return out
}

func concatVectors(vectors ...[]float32) []float32 {
	n := 0
	for _, v := range vectors {
		n += len(v)
	}
	out := make([]float32, 0, n)
	for _, v := range vectors {
		out = append(out, v...)
	}
	return out
}

func sameIntSlice(a, b []int) bool {
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
