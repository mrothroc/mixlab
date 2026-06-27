package train

import (
	"embed"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
)

//go:embed hf_templates/configuration_mixlab.py hf_templates/modeling_mixlab.py
var hfTemplateFS embed.FS

// ExportHFOptions describes a Hugging Face export operation.
type ExportHFOptions struct {
	ConfigPath      string
	SafetensorsLoad string
	OutputDir       string
	TokenizerSource string
}

type hfConfigJSON struct {
	ModelType             string            `json:"model_type"`
	Architectures         []string          `json:"architectures"`
	AutoMap               map[string]string `json:"auto_map"`
	Name                  string            `json:"name,omitempty"`
	ModelDim              int               `json:"model_dim"`
	HiddenSize            int               `json:"hidden_size"`
	VocabSize             int               `json:"vocab_size"`
	SeqLen                int               `json:"seq_len"`
	MaxPositionEmbeddings int               `json:"max_position_embeddings"`
	MLPMult               float64           `json:"mlp_mult"`
	NormType              string            `json:"norm_type,omitempty"`
	NormEps               float32           `json:"norm_eps,omitempty"`
	NormAffine            bool              `json:"norm_affine"`
	NormPlacement         string            `json:"norm_placement,omitempty"`
	FFNInternalNorm       bool              `json:"ffn_internal_norm,omitempty"`
	LogitSoftcap          float32           `json:"logit_softcap,omitempty"`
	MLMHead               string            `json:"mlm_head,omitempty"`
	LayerAggregation      string            `json:"layer_aggregation,omitempty"`
	HiddenDropout         float32           `json:"hidden_dropout,omitempty"`
	EmbeddingDropout      float32           `json:"embedding_dropout,omitempty"`
	PositionalEmbedding   string            `json:"positional_embedding,omitempty"`
	CharVocabSize         int               `json:"char_vocab_size,omitempty"`
	CharDim               int               `json:"char_dim,omitempty"`
	CharMaxPerToken       int               `json:"char_max_per_token,omitempty"`
	CharFeaturesFile      string            `json:"char_features_file,omitempty"`
	BigramVocabSize       int               `json:"bigram_vocab_size,omitempty"`
	BigramDim             int               `json:"bigram_dim,omitempty"`
	TrigramVocabSize      int               `json:"trigram_vocab_size,omitempty"`
	TrigramDim            int               `json:"trigram_dim,omitempty"`
	PadTokenID            *int              `json:"pad_token_id,omitempty"`
	EOSTokenID            *int              `json:"eos_token_id,omitempty"`
	BOSTokenID            *int              `json:"bos_token_id,omitempty"`
	UNKTokenID            *int              `json:"unk_token_id,omitempty"`
	Blocks                []map[string]any  `json:"blocks"`
	MaskedBlocks          []map[string]any  `json:"masked_blocks,omitempty"`
	Mixlab                map[string]any    `json:"mixlab"`
}

type hfWeightMapping struct {
	Mixlab string `json:"mixlab"`
	HF     string `json:"hf"`
	Shape  []int  `json:"shape"`
}

type hfBlockWeightName struct {
	mixlab string
	hf     string
}

type hfTokenizerSource struct {
	Dir           string
	TokenizerJSON string
}

// RunExportHF exports a supported Mixlab checkpoint as a minimal Hugging Face
// custom-code CausalLM directory.
func RunExportHF(opts ExportHFOptions) error {
	return runExportHF(opts)
}

func runExportHF(opts ExportHFOptions) error {
	if strings.TrimSpace(opts.ConfigPath) == "" {
		return fmt.Errorf("-config is required for export-hf mode")
	}
	if strings.TrimSpace(opts.SafetensorsLoad) == "" {
		return fmt.Errorf("-safetensors-load is required for export-hf mode")
	}
	if strings.TrimSpace(opts.OutputDir) == "" {
		return fmt.Errorf("-export-dir (or legacy -output) is required for export-hf mode")
	}

	cfg, err := LoadArchConfig(opts.ConfigPath)
	if err != nil {
		return err
	}
	if exportCfg := hfExportInferenceConfig(cfg); exportCfg != nil && exportCfg.EffectiveHFExportFormat() == "gpt2" {
		return runExportHFGPT2(opts, exportCfg)
	}
	if err := configureCharFeaturesForHFExport(cfg, opts.ConfigPath, opts.SafetensorsLoad, opts.TokenizerSource); err != nil {
		return err
	}
	exportCfg := hfExportInferenceConfig(cfg)
	if err := validateHFExportConfig(exportCfg); err != nil {
		return err
	}

	shapes, err := computeWeightShapes(exportCfg)
	if err != nil {
		return fmt.Errorf("compute weight shapes: %w", err)
	}
	weights, err := loadSafetensorsWeights(opts.SafetensorsLoad, shapes)
	if err != nil {
		return fmt.Errorf("load safetensors %q: %w", opts.SafetensorsLoad, err)
	}
	exportShapes, exportWeights, err := materializeHFExportWeights(exportCfg, shapes, weights)
	if err != nil {
		return err
	}
	mapping, err := buildHFWeightMap(exportCfg, exportShapes)
	if err != nil {
		return err
	}
	if len(mapping) != len(exportWeights) {
		return fmt.Errorf("HF weight map count mismatch: mapping=%d weights=%d", len(mapping), len(exportWeights))
	}

	tokenizer, err := resolveHFTokenizerSource(opts.TokenizerSource, opts.ConfigPath, opts.SafetensorsLoad)
	if err != nil {
		return err
	}
	specials, err := deriveHFTokenizerSpecials(tokenizer, exportCfg)
	if err != nil {
		return err
	}

	if err := os.MkdirAll(opts.OutputDir, 0o755); err != nil {
		return fmt.Errorf("create HF output directory %q: %w", opts.OutputDir, err)
	}
	if err := writeHFConfig(filepath.Join(opts.OutputDir, "config.json"), exportCfg, specials); err != nil {
		return err
	}
	if err := writeHFTemplates(opts.OutputDir); err != nil {
		return err
	}
	if err := writeHFSafetensors(filepath.Join(opts.OutputDir, "model.safetensors"), exportCfg, mapping, exportWeights); err != nil {
		return err
	}
	if err := writeJSONFile(filepath.Join(opts.OutputDir, "weight_map.json"), mapping); err != nil {
		return err
	}
	if err := writeHFTokenizerArtifacts(opts.OutputDir, tokenizer, specials); err != nil {
		return err
	}
	if err := writeHFCharFeatureArtifact(opts.OutputDir, exportCfg); err != nil {
		return err
	}

	fmt.Printf("exported Hugging Face model to %s (%d tensors)\n", opts.OutputDir, len(weights))
	return nil
}

func hfExportInferenceConfig(cfg *ArchConfig) *ArchConfig {
	if cfg == nil {
		return nil
	}
	out := *cfg
	out.Training = cfg.Training
	out.Training.Data2Vec = nil
	return &out
}

func materializeHFExportWeights(cfg *ArchConfig, shapes []WeightShape, weights [][]float32) ([]WeightShape, [][]float32, error) {
	if cfg == nil {
		return nil, nil, fmt.Errorf("nil config")
	}
	if len(shapes) != len(weights) {
		return nil, nil, fmt.Errorf("weight shape/data count mismatch: shapes=%d weights=%d", len(shapes), len(weights))
	}
	if hasWeightShapeName(shapes, "head") {
		return shapes, weights, nil
	}
	if !cfg.TieEmbeddings {
		return nil, nil, fmt.Errorf("HF export requires head weight for untied embeddings")
	}
	embedIdx := weightShapeIndex(shapes, "embed")
	if embedIdx < 0 {
		return nil, nil, fmt.Errorf("HF export requires base weight %q", "embed")
	}
	embedShape := shapes[embedIdx].Shape
	if len(embedShape) != 2 || embedShape[0] != cfg.VocabSize || embedShape[1] != cfg.ModelDim {
		return nil, nil, fmt.Errorf("embed shape=%v does not match vocab/model dims [%d,%d]", embedShape, cfg.VocabSize, cfg.ModelDim)
	}
	head := transposeEmbeddingToHead(weights[embedIdx], cfg.VocabSize, cfg.ModelDim)
	outShapes := append([]WeightShape(nil), shapes...)
	outWeights := append([][]float32(nil), weights...)
	outShapes = append(outShapes, WeightShape{Name: "head", Shape: []int{cfg.ModelDim, cfg.VocabSize}})
	outWeights = append(outWeights, head)
	return outShapes, outWeights, nil
}

func transposeEmbeddingToHead(embed []float32, vocab, dim int) []float32 {
	head := make([]float32, dim*vocab)
	for v := 0; v < vocab; v++ {
		for d := 0; d < dim; d++ {
			head[d*vocab+v] = embed[v*dim+d]
		}
	}
	return head
}

func hasWeightShapeName(shapes []WeightShape, name string) bool {
	return weightShapeIndex(shapes, name) >= 0
}

func weightShapeIndex(shapes []WeightShape, name string) int {
	for i, shape := range shapes {
		if shape.Name == name {
			return i
		}
	}
	return -1
}

func writeHFTemplates(outputDir string) error {
	for _, name := range []string{"configuration_mixlab.py", "modeling_mixlab.py"} {
		data, err := hfTemplateFS.ReadFile(filepath.Join("hf_templates", name))
		if err != nil {
			return fmt.Errorf("read HF template %s: %w", name, err)
		}
		if err := os.WriteFile(filepath.Join(outputDir, name), data, 0o644); err != nil {
			return fmt.Errorf("write HF template %s: %w", name, err)
		}
	}
	return nil
}

func buildHFWeightMap(cfg *ArchConfig, shapes []WeightShape) ([]hfWeightMapping, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
	}
	out := make([]hfWeightMapping, len(shapes))
	used := make([]bool, len(shapes))
	add := func(idx int, hf string) error {
		if idx < 0 || idx >= len(shapes) {
			return fmt.Errorf("weight index %d out of range", idx)
		}
		if used[idx] {
			return fmt.Errorf("weight index %d (%s) mapped more than once", idx, shapes[idx].Name)
		}
		out[idx] = hfWeightMapping{
			Mixlab: fmt.Sprintf("w%d_%s", idx, shapes[idx].Name),
			HF:     hf,
			Shape:  append([]int(nil), shapes[idx].Shape...),
		}
		used[idx] = true
		return nil
	}
	addExpected := func(idx int, wantName, hf string) error {
		if idx < 0 || idx >= len(shapes) {
			return fmt.Errorf("weight index %d out of range while expecting %q", idx, wantName)
		}
		if shapes[idx].Name != wantName {
			return fmt.Errorf("HF export weight map expected Mixlab weight %q at index %d, got %q", wantName, idx, shapes[idx].Name)
		}
		return add(idx, hf)
	}
	addByName := func(name, hf string) error {
		for i, shape := range shapes {
			if !used[i] && shape.Name == name {
				return add(i, hf)
			}
		}
		return fmt.Errorf("HF export requires base weight %q", name)
	}
	if err := addByName("embed", "embed_tokens.weight"); err != nil {
		return nil, err
	}
	if err := addByName("head", "lm_head_weight"); err != nil {
		return nil, err
	}
	normSpec := cfg.EffectiveNormSpec()
	normPlacement := cfg.EffectiveNormPlacement()
	addNormByName := func(base, hfPrefix string) error {
		switch strings.ToLower(strings.TrimSpace(normSpec.Type)) {
		case "", "rmsnorm":
			return addByName(rmsNormWeightName(base), hfPrefix+".weight")
		case "layernorm":
			if !normSpec.Affine {
				return nil
			}
			if err := addByName(base+"_scale", hfPrefix+".weight"); err != nil {
				return err
			}
			return addByName(base+"_bias", hfPrefix+".bias")
		default:
			return fmt.Errorf("unsupported HF export norm_type %q", normSpec.Type)
		}
	}
	appendNormNames := func(names []hfBlockWeightName, base, hfPrefix string) []hfBlockWeightName {
		switch strings.ToLower(strings.TrimSpace(normSpec.Type)) {
		case "", "rmsnorm":
			return append(names, hfBlockWeightName{mixlab: rmsNormWeightName(base), hf: hfPrefix + ".weight"})
		case "layernorm":
			if !normSpec.Affine {
				return names
			}
			return append(names,
				hfBlockWeightName{mixlab: base + "_scale", hf: hfPrefix + ".weight"},
				hfBlockWeightName{mixlab: base + "_bias", hf: hfPrefix + ".bias"},
			)
		default:
			return names
		}
	}
	if err := addNormByName("final_norm", "final_norm"); err != nil {
		return nil, err
	}
	if cfg.EffectivePositionalEmbedding() == "learned_absolute" {
		if err := addByName("position_embeddings", "position_embeddings.weight"); err != nil {
			return nil, err
		}
	}

	wi := firstUnmappedWeight(used, 0)
	featureNames := []struct {
		enabled bool
		items   []struct {
			mixlab string
			hf     string
		}
	}{
		{enabled: cfg.CharVocabSize > 0, items: []struct {
			mixlab string
			hf     string
		}{
			{mixlab: "char_table", hf: "char_table.weight"},
			{mixlab: "char_proj", hf: "char_proj.weight"},
			{mixlab: "char_scale", hf: "char_scale"},
		}},
		{enabled: cfg.BigramVocabSize > 0, items: []struct {
			mixlab string
			hf     string
		}{
			{mixlab: "bigram_table", hf: "bigram_table.weight"},
			{mixlab: "bigram_proj", hf: "bigram_proj.weight"},
			{mixlab: "bigram_scale", hf: "bigram_scale"},
		}},
		{enabled: cfg.TrigramVocabSize > 0, items: []struct {
			mixlab string
			hf     string
		}{
			{mixlab: "trigram_table", hf: "trigram_table.weight"},
			{mixlab: "trigram_proj", hf: "trigram_proj.weight"},
			{mixlab: "trigram_scale", hf: "trigram_scale"},
		}},
	}
	for _, group := range featureNames {
		if !group.enabled {
			continue
		}
		for _, name := range group.items {
			if wi >= len(shapes) || shapes[wi].Name != name.mixlab {
				if name.mixlab == "char_proj" || name.mixlab == "bigram_proj" || name.mixlab == "trigram_proj" {
					continue
				}
			}
			if err := addExpected(wi, name.mixlab, name.hf); err != nil {
				return nil, err
			}
			wi = firstUnmappedWeight(used, wi+1)
		}
	}
	if hfConfigUsesSharedRelativeAttention(cfg) {
		if wi >= len(shapes) || shapes[wi].Name != "shared_relative_embeddings" {
			return nil, fmt.Errorf("weight map expected shared_relative_embeddings at index %d", wi)
		}
		if err := addExpected(wi, "shared_relative_embeddings", "relative_embeddings"); err != nil {
			return nil, err
		}
		wi = firstUnmappedWeight(used, wi+1)
		if hfConfigUsesSharedRelativeEmbeddingNorm(cfg) {
			if wi >= len(shapes) || shapes[wi].Name != "shared_relative_norm_scale" {
				return nil, fmt.Errorf("weight map expected shared_relative_norm_scale at index %d", wi)
			}
			if err := addExpected(wi, "shared_relative_norm_scale", "relative_layer_norm.weight"); err != nil {
				return nil, err
			}
			wi = firstUnmappedWeight(used, wi+1)
			if wi >= len(shapes) || shapes[wi].Name != "shared_relative_norm_bias" {
				return nil, fmt.Errorf("weight map expected shared_relative_norm_bias at index %d", wi)
			}
			if err := addExpected(wi, "shared_relative_norm_bias", "relative_layer_norm.bias"); err != nil {
				return nil, err
			}
			wi = firstUnmappedWeight(used, wi+1)
		}
	}
	for blockIdx, block := range cfg.Blocks {
		prefix := fmt.Sprintf("blocks.%d", blockIdx)
		switch strings.ToLower(strings.TrimSpace(block.Type)) {
		case "plain":
			var names []hfBlockWeightName
			if normPlacement == "pre" || normPlacement == "sandwich" {
				names = appendNormNames(names, "norm", "norm")
			}
			names = append(names, []hfBlockWeightName{
				{mixlab: "wq", hf: "wq.weight"},
				{mixlab: "wq_bias", hf: "wq.bias"},
				{mixlab: "wk", hf: "wk.weight"},
				{mixlab: "wk_bias", hf: "wk.bias"},
				{mixlab: "wv", hf: "wv.weight"},
				{mixlab: "wv_bias", hf: "wv.bias"},
				{mixlab: "q_norm_scale", hf: "q_norm.weight"},
				{mixlab: "k_norm_scale", hf: "k_norm.weight"},
				{mixlab: "relative_embeddings", hf: "relative_embeddings"},
				{mixlab: "w_pos_key", hf: "w_pos_key.weight"},
				{mixlab: "w_pos_query", hf: "w_pos_query.weight"},
				{mixlab: "qk_gain", hf: "qk_gain"},
				{mixlab: "attn_gate_w", hf: "attn_gate_w"},
			}...)
			attnPostNorm := hfEffectivePlainAttnPostNorm(block, normPlacement)
			if attnPostNorm == "before_outproj" {
				names = appendNormNames(names, "post_attn_norm", "post_attn_norm")
			}
			names = append(names,
				hfBlockWeightName{mixlab: "wo", hf: "wo.weight"},
				hfBlockWeightName{mixlab: "wo_bias", hf: "wo.bias"},
			)
			if attnPostNorm == "after_outproj" {
				names = appendNormNames(names, "post_attn_norm", "post_attn_norm")
			}
			if normPlacement == "sandwich" || block.FFNPreNorm {
				names = appendNormNames(names, "ffn_norm", "ffn_norm")
			}
			if hfPlainFFNActivationUsesGate(block) {
				names = append(names, hfBlockWeightName{mixlab: "ff_gate", hf: "ff_gate.weight"})
			}
			names = append(names, hfBlockWeightName{mixlab: "ff1", hf: "ff1.weight"})
			if block.FFNBias {
				names = append(names, hfBlockWeightName{mixlab: "ff1_bias", hf: "ff1.bias"})
			}
			if cfg.FFNInternalNorm {
				names = appendNormNames(names, "ffn_internal_norm", "ffn_internal_norm")
			}
			names = append(names, hfBlockWeightName{mixlab: "ff2", hf: "ff2.weight"})
			if block.FFNBias {
				names = append(names, hfBlockWeightName{mixlab: "ff2_bias", hf: "ff2.bias"})
			}
			if normPlacement == "post" || normPlacement == "sandwich" {
				names = appendNormNames(names, "post_ffn_norm", "post_ffn_norm")
			}
			for _, name := range names {
				if wi >= len(shapes) {
					return nil, fmt.Errorf("weight map exhausted while mapping plain block %d", blockIdx)
				}
				if (name.mixlab == "relative_embeddings" || name.mixlab == "w_pos_key" || name.mixlab == "w_pos_query") && (!relativeAttentionEnabledForHF(block) || hfRelativeAttentionUsesSharedQKReuse(block)) {
					continue
				}
				if (name.mixlab == "q_norm_scale" || name.mixlab == "k_norm_scale") && !block.QKNorm {
					continue
				}
				switch name.mixlab {
				case "wq_bias", "wk_bias", "wv_bias", "wo_bias":
					if !block.AttnBias {
						continue
					}
				}
				if name.mixlab == "qk_gain" && shapes[wi].Name != "qk_gain" {
					continue
				}
				if name.mixlab == "attn_gate_w" && !block.SparseAttnGate {
					continue
				}
				if err := addExpected(wi, name.mixlab, prefix+"."+name.hf); err != nil {
					return nil, err
				}
				wi = firstUnmappedWeight(used, wi+1)
			}
		case "swiglu", "geglu":
			var names []hfBlockWeightName
			if normPlacement == "pre" || normPlacement == "sandwich" {
				names = appendNormNames(names, "ffn_norm", "norm")
			}
			names = append(names, []hfBlockWeightName{
				{mixlab: "w_gate", hf: "w_gate.weight"},
				{mixlab: "w_up", hf: "w_up.weight"},
			}...)
			if cfg.FFNInternalNorm {
				names = appendNormNames(names, "ffn_internal_norm", "internal_norm")
			}
			names = append(names, hfBlockWeightName{mixlab: "w_down", hf: "w_down.weight"})
			if normPlacement == "post" || normPlacement == "sandwich" {
				names = appendNormNames(names, "post_ffn_norm", "post_norm")
			}
			for _, name := range names {
				if wi >= len(shapes) {
					return nil, fmt.Errorf("weight map exhausted while mapping swiglu block %d", blockIdx)
				}
				if err := addExpected(wi, name.mixlab, prefix+"."+name.hf); err != nil {
					return nil, err
				}
				wi = firstUnmappedWeight(used, wi+1)
			}
		case "mlp":
			var names []hfBlockWeightName
			if normPlacement == "pre" || normPlacement == "sandwich" {
				names = appendNormNames(names, "ffn_norm", "norm")
			}
			names = append(names, hfBlockWeightName{mixlab: "w_up", hf: "w_up.weight"})
			if cfg.FFNInternalNorm {
				names = appendNormNames(names, "ffn_internal_norm", "internal_norm")
			}
			names = append(names, hfBlockWeightName{mixlab: "w_down", hf: "w_down.weight"})
			if normPlacement == "post" || normPlacement == "sandwich" {
				names = appendNormNames(names, "post_ffn_norm", "post_norm")
			}
			for _, name := range names {
				if wi >= len(shapes) {
					return nil, fmt.Errorf("weight map exhausted while mapping mlp block %d", blockIdx)
				}
				if err := addExpected(wi, name.mixlab, prefix+"."+name.hf); err != nil {
					return nil, err
				}
				wi = firstUnmappedWeight(used, wi+1)
			}
		case "moe":
			names := []struct {
				mixlab string
				hf     string
			}{
				{mixlab: "moe_norm_scale", hf: "norm.weight"},
				{mixlab: "router_w", hf: "router_w"},
			}
			for _, name := range names {
				if wi >= len(shapes) {
					return nil, fmt.Errorf("weight map exhausted while mapping moe block %d", blockIdx)
				}
				if err := addExpected(wi, name.mixlab, prefix+"."+name.hf); err != nil {
					return nil, err
				}
				wi = firstUnmappedWeight(used, wi+1)
			}
			expert := BlockSpec{Type: "swiglu"}
			if block.ExpertBlock != nil {
				expert = *block.ExpertBlock
			}
			expertType := strings.ToLower(strings.TrimSpace(expert.Type))
			if expertType == "" {
				expertType = "swiglu"
			}
			for e := 0; e < block.NumExperts; e++ {
				var expertNames []struct {
					mixlab string
					hf     string
				}
				switch expertType {
				case "swiglu", "geglu":
					expertNames = []struct {
						mixlab string
						hf     string
					}{
						{mixlab: fmt.Sprintf("expert_%d_w_gate", e), hf: fmt.Sprintf("experts.%d.w_gate.weight", e)},
						{mixlab: fmt.Sprintf("expert_%d_w_up", e), hf: fmt.Sprintf("experts.%d.w_up.weight", e)},
						{mixlab: fmt.Sprintf("expert_%d_w_down", e), hf: fmt.Sprintf("experts.%d.w_down.weight", e)},
					}
				case "mlp":
					expertNames = []struct {
						mixlab string
						hf     string
					}{
						{mixlab: fmt.Sprintf("expert_%d_w_up", e), hf: fmt.Sprintf("experts.%d.w_up.weight", e)},
						{mixlab: fmt.Sprintf("expert_%d_w_down", e), hf: fmt.Sprintf("experts.%d.w_down.weight", e)},
					}
				default:
					return nil, fmt.Errorf("unsupported HF export moe expert type %q", expert.Type)
				}
				for _, name := range expertNames {
					if wi >= len(shapes) {
						return nil, fmt.Errorf("weight map exhausted while mapping moe block %d expert %d", blockIdx, e)
					}
					if err := addExpected(wi, name.mixlab, prefix+"."+name.hf); err != nil {
						return nil, err
					}
					wi = firstUnmappedWeight(used, wi+1)
				}
			}
		default:
			return nil, fmt.Errorf("unsupported HF export block type %q", block.Type)
		}
	}
	if cfg.EffectiveMLMHead() == "bert" {
		for _, name := range []hfBlockWeightName{
			{mixlab: "mlm_head_dense", hf: "mlm_head_dense.weight"},
			{mixlab: "mlm_head_dense_bias", hf: "mlm_head_dense.bias"},
			{mixlab: "mlm_head_output_bias", hf: "mlm_head_output_bias"},
		} {
			if wi >= len(shapes) {
				return nil, fmt.Errorf("weight map exhausted while mapping BERT MLM head")
			}
			if err := addExpected(wi, name.mixlab, name.hf); err != nil {
				return nil, err
			}
			wi = firstUnmappedWeight(used, wi+1)
		}
	}
	if cfg.EffectiveLayerAggregation() == "dwa" {
		count, err := hfLayerAggregationPointCount(cfg)
		if err != nil {
			return nil, err
		}
		for i := 0; i < count; i++ {
			name := fmt.Sprintf("dwa_alpha_%d", i)
			if wi >= len(shapes) {
				return nil, fmt.Errorf("weight map exhausted while mapping DWA alpha %d", i)
			}
			if err := addExpected(wi, name, fmt.Sprintf("dwa_alphas.%d", i)); err != nil {
				return nil, err
			}
			wi = firstUnmappedWeight(used, wi+1)
		}
	}
	if wi != len(shapes) {
		return nil, fmt.Errorf("HF weight map did not consume all weights: next_unmapped=%d total=%d", wi, len(shapes))
	}
	return out, nil
}

func hfLayerAggregationPointCount(cfg *ArchConfig) (int, error) {
	if cfg == nil || cfg.EffectiveLayerAggregation() != "dwa" {
		return 0, nil
	}
	count := 0
	for i, block := range cfg.Blocks {
		switch strings.ToLower(strings.TrimSpace(block.Type)) {
		case "plain":
			count += 2
		case "swiglu", "geglu", "mlp", "moe":
			count++
		default:
			return 0, fmt.Errorf("HF export DWA does not support blocks[%d].type=%q", i, block.Type)
		}
	}
	return count, nil
}

func rmsNormWeightName(base string) string {
	if base == "final_norm" || strings.HasSuffix(base, "_scale") {
		return base
	}
	return base + "_scale"
}

func firstUnmappedWeight(used []bool, start int) int {
	for start < len(used) && used[start] {
		start++
	}
	return start
}

func writeHFSafetensors(path string, cfg *ArchConfig, mapping []hfWeightMapping, weights [][]float32) error {
	header := make(map[string]safetensorHeaderEntry, len(weights))
	var offset uint64
	for i, m := range mapping {
		start := offset
		offset += uint64(len(weights[i]) * 4)
		header[m.HF] = safetensorHeaderEntry{
			DType:       "F32",
			Shape:       append([]int(nil), m.Shape...),
			DataOffsets: []uint64{start, offset},
		}
	}
	meta := map[string]string{
		"format":     "pt",
		"name":       cfg.Name,
		"model_dim":  fmt.Sprintf("%d", cfg.ModelDim),
		"vocab_size": fmt.Sprintf("%d", cfg.VocabSize),
		"seq_len":    fmt.Sprintf("%d", cfg.SeqLen),
	}

	headerMap := make(map[string]json.RawMessage, len(header)+1)
	for k, v := range header {
		b, err := json.Marshal(v)
		if err != nil {
			return fmt.Errorf("marshal safetensors header entry %q: %w", k, err)
		}
		headerMap[k] = b
	}
	metaBytes, err := json.Marshal(meta)
	if err != nil {
		return err
	}
	headerMap["__metadata__"] = metaBytes
	headerBytes, err := json.Marshal(headerMap)
	if err != nil {
		return err
	}
	if pad := (8 - ((8 + len(headerBytes)) % 8)) % 8; pad > 0 {
		headerBytes = append(headerBytes, []byte(strings.Repeat(" ", pad))...)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %q: %w", path, err)
	}
	defer func() { _ = f.Close() }()
	if err := binary.Write(f, binary.LittleEndian, uint64(len(headerBytes))); err != nil {
		return err
	}
	if _, err := f.Write(headerBytes); err != nil {
		return err
	}
	for i := range mapping {
		if err := writeFloat32Data(f, weights[i]); err != nil {
			return err
		}
	}
	return nil
}

func writeFloat32Data(w io.Writer, data []float32) error {
	const chunkValues = 16 * 1024
	buf := make([]byte, chunkValues*4)
	for len(data) > 0 {
		n := len(data)
		if n > chunkValues {
			n = chunkValues
		}
		for i, v := range data[:n] {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
		}
		if _, err := w.Write(buf[:n*4]); err != nil {
			return err
		}
		data = data[n:]
	}
	return nil
}

func configureCharFeaturesForHFExport(cfg *ArchConfig, configPath, weightsPath, tokenizerSource string) error {
	if cfg == nil || cfg.CharVocabSize <= 0 {
		return nil
	}
	if len(cfg.CharFeatureIDs) == cfg.VocabSize*cfg.CharMaxPerToken {
		if strings.TrimSpace(cfg.CharFeatureSource) == "" {
			return fmt.Errorf("char feature IDs are preloaded but CharFeatureSource is empty; export-hf needs %s source path to copy", charFeaturesFilename)
		}
		return nil
	}
	candidates := make([]string, 0, 4)
	if configPath != "" {
		candidates = append(candidates, filepath.Join(filepath.Dir(configPath), charFeaturesFilename))
	}
	if weightsPath != "" {
		candidates = append(candidates, filepath.Join(filepath.Dir(weightsPath), charFeaturesFilename))
	}
	if tokenizerSource != "" {
		if info, err := os.Stat(tokenizerSource); err == nil && info.IsDir() {
			candidates = append(candidates, filepath.Join(tokenizerSource, charFeaturesFilename))
		} else {
			candidates = append(candidates, filepath.Join(filepath.Dir(tokenizerSource), charFeaturesFilename))
		}
	}
	for _, path := range uniqueStrings(candidates) {
		if _, err := os.Stat(path); err == nil {
			return loadCharFeaturesIntoConfig(cfg, path)
		}
	}
	return fmt.Errorf("char_vocab_size=%d requires %s next to config, weights, or tokenizer source for export-hf", cfg.CharVocabSize, charFeaturesFilename)
}

func writeHFCharFeatureArtifact(outputDir string, cfg *ArchConfig) error {
	if cfg == nil || cfg.CharVocabSize <= 0 {
		return nil
	}
	if strings.TrimSpace(cfg.CharFeatureSource) == "" {
		return fmt.Errorf("char features enabled but no char feature source was loaded")
	}
	return copyFile(cfg.CharFeatureSource, filepath.Join(outputDir, charFeaturesFilename))
}

func resolveHFTokenizerSource(explicit, configPath, weightsPath string) (hfTokenizerSource, error) {
	candidates := []string{}
	if strings.TrimSpace(explicit) != "" {
		candidates = append(candidates, explicit)
	} else {
		seen := map[string]bool{}
		for _, base := range []string{filepath.Dir(configPath), filepath.Dir(weightsPath)} {
			if base == "" {
				base = "."
			}
			candidate := filepath.Join(base, "tokenizer.json")
			if !seen[candidate] {
				candidates = append(candidates, candidate)
				seen[candidate] = true
			}
		}
		if !seen["tokenizer.json"] {
			candidates = append(candidates, "tokenizer.json")
		}
	}
	for _, candidate := range candidates {
		src, err := inspectHFTokenizerSource(candidate)
		if err == nil {
			return src, nil
		}
		if strings.TrimSpace(explicit) != "" {
			return hfTokenizerSource{}, err
		}
	}
	return hfTokenizerSource{}, fmt.Errorf("tokenizer source is required for export-hf: pass -tokenizer-path pointing at tokenizer.json or a directory containing tokenizer.json")
}

func inspectHFTokenizerSource(path string) (hfTokenizerSource, error) {
	if strings.TrimSpace(path) == "" {
		return hfTokenizerSource{}, fmt.Errorf("empty tokenizer source")
	}
	info, err := os.Stat(path)
	if err != nil {
		return hfTokenizerSource{}, fmt.Errorf("verify tokenizer source %q: %w", path, err)
	}
	if info.IsDir() {
		tokenizerJSON := filepath.Join(path, "tokenizer.json")
		if _, err := os.Stat(tokenizerJSON); err != nil {
			return hfTokenizerSource{}, fmt.Errorf("verify tokenizer source %q: missing tokenizer.json: %w", path, err)
		}
		return hfTokenizerSource{Dir: path, TokenizerJSON: tokenizerJSON}, nil
	}
	if filepath.Base(path) != "tokenizer.json" {
		return hfTokenizerSource{}, fmt.Errorf("verify tokenizer source %q: expected tokenizer.json or a directory containing tokenizer.json", path)
	}
	return hfTokenizerSource{Dir: filepath.Dir(path), TokenizerJSON: path}, nil
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("open %q: %w", src, err)
	}
	defer func() { _ = in.Close() }()
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	out, err := os.Create(dst)
	if err != nil {
		return fmt.Errorf("create %q: %w", dst, err)
	}
	defer func() { _ = out.Close() }()
	if _, err := io.Copy(out, in); err != nil {
		return fmt.Errorf("copy %q to %q: %w", src, dst, err)
	}
	return nil
}

func writeJSONFile(path string, v any) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal %q: %w", path, err)
	}
	data = append(data, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write %q: %w", path, err)
	}
	return nil
}
