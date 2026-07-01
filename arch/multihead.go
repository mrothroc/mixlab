package arch

import (
	"fmt"
	"math"
	"strings"
)

const (
	MultiheadOutputLinear  = "linear"
	MultiheadOutputBERTMLM = "bert_mlm"
	MultiheadOutputBinary  = "binary"
)

func normalizeMultiheadOutputHead(raw string, objective string) string {
	mode := strings.ToLower(strings.TrimSpace(raw))
	switch mode {
	case "", "default":
		switch normalizeTrainingObjective(objective) {
		case ObjectiveMLM, ObjectiveMNTP:
			return MultiheadOutputBERTMLM
		case ObjectiveRTD:
			return MultiheadOutputBinary
		default:
			return MultiheadOutputLinear
		}
	case "linear", "lm", "lm_head":
		return MultiheadOutputLinear
	case "bert", "bert_mlm", "mlm_bert":
		return MultiheadOutputBERTMLM
	case "binary", "rtd":
		return MultiheadOutputBinary
	default:
		return mode
	}
}

func validateTrainingMultihead(cfg *ArchConfig, source string) error {
	t := &cfg.Training
	if len(t.Heads) < 2 {
		return fmt.Errorf("config %q training.objective=\"multihead\" requires at least two training.heads entries", source)
	}
	if t.Diffusion != nil {
		return fmt.Errorf("config %q training.objective=\"multihead\" requires per-head diffusion configs; remove top-level training.diffusion", source)
	}
	if cfg.EffectiveLayerAggregation() != LayerAggregationNone {
		return fmt.Errorf("config %q training.objective=\"multihead\" requires top-level layer_aggregation to be omitted or \"none\"; set per-head layer_aggregation instead", source)
	}
	if cfg.MTP != nil {
		return fmt.Errorf("config %q training.objective=\"multihead\" cannot be combined with top-level mtp in v1", source)
	}
	if t.FirstByteMask {
		return fmt.Errorf("config %q training.objective=\"multihead\" cannot be combined with training.first_byte_mask in v1", source)
	}
	if t.Distillation != nil {
		return fmt.Errorf("config %q training.objective=\"multihead\" cannot be combined with training.distillation in v1", source)
	}
	if t.Data2VecActive() {
		return fmt.Errorf("config %q training.objective=\"multihead\" cannot be combined with training.data2vec in v1", source)
	}
	if t.ExampleFramingEnabled() {
		return fmt.Errorf("config %q training.objective=\"multihead\" cannot be combined with training.example_framing in v1", source)
	}
	if t.AttentionSegmentMaskEnabled() {
		return fmt.Errorf("config %q training.objective=\"multihead\" cannot be combined with training.attention_segment_mask in v1", source)
	}
	if cfg.UNet || cfg.ParallelResidual || cfg.Backout != nil || len(cfg.Recurrence) > 0 || len(cfg.RecurrencePhases) > 0 || cfg.executionOrderSet {
		return fmt.Errorf("config %q training.objective=\"multihead\" does not support unet, parallel_residual, backout, recurrence, or custom execution order in v1", source)
	}
	if t.MLMMaskProb < 0 || t.MLMMaskProb > 1 || math.IsNaN(t.MLMMaskProb) {
		return fmt.Errorf("config %q has invalid training.mlm_mask_prob=%g (must be in [0,1])", source, t.MLMMaskProb)
	}
	for name, value := range map[string]float64{
		"mlm_mask_token_prob":     t.MLMMaskTokenProb,
		"mlm_random_token_prob":   t.MLMRandomTokenProb,
		"mlm_kept_unchanged_prob": t.MLMKeptUnchangedProb,
	} {
		if value < 0 || value > 1 || math.IsNaN(value) {
			return fmt.Errorf("config %q has invalid training.%s=%g (must be in [0,1])", source, name, value)
		}
	}
	if probSum := t.MLMMaskTokenProb + t.MLMRandomTokenProb + t.MLMKeptUnchangedProb; math.Abs(probSum-1.0) > 1e-6 {
		return fmt.Errorf("config %q has invalid MLM replacement probabilities: mlm_mask_token_prob + mlm_random_token_prob + mlm_kept_unchanged_prob = %g (must sum to 1.0)", source, probSum)
	}

	seen := make(map[string]int, len(t.Heads))
	totalWeight := 0.0
	diffusionHeads := 0
	exportableHeads := 0
	rtdHeads := 0
	usesAdaLN := false
	for i := range t.Heads {
		h := &t.Heads[i]
		h.Name = strings.TrimSpace(h.Name)
		if h.Name == "" {
			return fmt.Errorf("config %q training.heads[%d].name is required", source, i)
		}
		if prev, ok := seen[h.Name]; ok {
			return fmt.Errorf("config %q training.heads[%d].name=%q duplicates training.heads[%d]", source, i, h.Name, prev)
		}
		seen[h.Name] = i
		h.Objective = normalizeTrainingObjective(h.Objective)
		switch h.Objective {
		case ObjectiveCausal, ObjectiveMLM, ObjectiveMNTP, ObjectiveBlockDiffusion, ObjectiveRTD:
		default:
			return fmt.Errorf("config %q training.heads[%d].objective=%q is invalid for multihead (must be causal, mlm, mntp, block_diffusion, or rtd)", source, i, h.Objective)
		}
		if !h.lossWeightSet && h.LossWeight == 0 {
			h.LossWeight = 1
		}
		if h.LossWeight < 0 || math.IsNaN(h.LossWeight) || math.IsInf(h.LossWeight, 0) {
			return fmt.Errorf("config %q training.heads[%d].loss_weight=%g must be finite and >= 0", source, i, h.LossWeight)
		}
		totalWeight += h.LossWeight
		h.LayerAggregation = normalizeLayerAggregation(h.LayerAggregation)
		switch h.LayerAggregation {
		case LayerAggregationNone, LayerAggregationDWA:
		default:
			return fmt.Errorf("config %q training.heads[%d].layer_aggregation=%q must be \"none\" or \"dwa\"", source, i, h.LayerAggregation)
		}
		h.OutputHead = normalizeMultiheadOutputHead(h.OutputHead, h.Objective)
		switch h.OutputHead {
		case MultiheadOutputLinear, MultiheadOutputBERTMLM, MultiheadOutputBinary:
		default:
			return fmt.Errorf("config %q training.heads[%d].output_head=%q must be \"linear\", \"bert_mlm\", or \"binary\"", source, i, h.OutputHead)
		}
		if !h.finalNormSet {
			h.FinalNorm = true
		}
		switch {
		case h.OutputHead == MultiheadOutputBERTMLM:
			if h.Objective != ObjectiveMLM && h.Objective != ObjectiveMNTP {
				return fmt.Errorf("config %q training.heads[%d].output_head=\"bert_mlm\" requires objective mlm or mntp", source, i)
			}
			if h.tieEmbeddingsSet && !h.TieEmbeddings {
				return fmt.Errorf("config %q training.heads[%d].output_head=\"bert_mlm\" requires tie_embeddings=true", source, i)
			}
			h.TieEmbeddings = true
		case h.OutputHead == MultiheadOutputBinary:
			if h.Objective != ObjectiveRTD {
				return fmt.Errorf("config %q training.heads[%d].output_head=\"binary\" requires objective rtd", source, i)
			}
			h.TieEmbeddings = false
		case !h.tieEmbeddingsSet:
			h.TieEmbeddings = h.Objective != ObjectiveBlockDiffusion && cfg.TieEmbeddings
		}
		if h.Objective == ObjectiveBlockDiffusion {
			diffusionHeads++
			if h.Diffusion == nil {
				h.Diffusion = &DiffusionSpec{}
			}
			h.Diffusion.applyDefaults(cfg.SeqLen)
			if err := validateMultiheadDiffusionSpec(cfg, h.Diffusion, source, i); err != nil {
				return err
			}
			if h.Diffusion.TimestepConditioning == DiffusionTimestepConditioningAdaLN {
				usesAdaLN = true
			}
		} else {
			if h.Objective == ObjectiveRTD {
				rtdHeads++
			} else {
				exportableHeads++
			}
			if h.Diffusion != nil {
				return fmt.Errorf("config %q training.heads[%d].diffusion is only valid with objective=\"block_diffusion\"", source, i)
			}
		}
	}
	if t.RTD != nil {
		if rtdHeads != 1 {
			return fmt.Errorf("config %q training.rtd requires exactly one multihead objective=\"rtd\" head, got %d", source, rtdHeads)
		}
		if !finiteInClosedRange(t.RTD.MaskProb, 0, 1) {
			return fmt.Errorf("config %q training.rtd.mask_prob=%g must be in [0,1]", source, t.RTD.MaskProb)
		}
		if t.RTD.SampleTemperature <= 0 || math.IsNaN(t.RTD.SampleTemperature) || math.IsInf(t.RTD.SampleTemperature, 0) {
			return fmt.Errorf("config %q training.rtd.sample_temperature=%g must be finite and > 0", source, t.RTD.SampleTemperature)
		}
		if t.RTD.DiscriminatorLossWeight < 0 || math.IsNaN(t.RTD.DiscriminatorLossWeight) || math.IsInf(t.RTD.DiscriminatorLossWeight, 0) {
			return fmt.Errorf("config %q training.rtd.discriminator_loss_weight=%g must be finite and >= 0", source, t.RTD.DiscriminatorLossWeight)
		}
		switch t.RTD.Generator {
		case "tied":
			if t.RTD.DedicatedGenerator != nil {
				return fmt.Errorf("config %q training.rtd.generator has dedicated generator fields but type=\"tied\"", source)
			}
		case "dedicated":
			if err := validateRTDDedicatedGenerator(cfg, source); err != nil {
				return err
			}
		default:
			return fmt.Errorf("config %q training.rtd.generator=%q must be \"tied\" or a dedicated generator object", source, t.RTD.Generator)
		}
	} else if rtdHeads > 0 {
		return fmt.Errorf("config %q multihead objective=\"rtd\" requires training.rtd", source)
	}
	if totalWeight <= 0 {
		return fmt.Errorf("config %q training.objective=\"multihead\" requires positive total head loss_weight", source)
	}
	if exportableHeads == 0 {
		return fmt.Errorf("config %q training.objective=\"multihead\" requires at least one non-block_diffusion export/scorer head", source)
	}
	if strings.TrimSpace(t.ExportHead) == "" {
		for _, h := range t.Heads {
			if h.Objective != ObjectiveBlockDiffusion && h.Objective != ObjectiveRTD {
				t.ExportHead = h.Name
				break
			}
		}
	} else if idx, ok := seen[strings.TrimSpace(t.ExportHead)]; !ok {
		return fmt.Errorf("config %q training.export_head=%q does not match any training.heads[].name", source, t.ExportHead)
	} else if t.Heads[idx].Objective == ObjectiveBlockDiffusion || t.Heads[idx].Objective == ObjectiveRTD {
		return fmt.Errorf("config %q training.export_head=%q cannot select a %s head for HF export in v1", source, t.ExportHead, t.Heads[idx].Objective)
	}
	if strings.TrimSpace(t.DiffusionHead) == "" {
		if diffusionHeads > 0 {
			for _, h := range t.Heads {
				if h.Objective == ObjectiveBlockDiffusion {
					t.DiffusionHead = h.Name
					break
				}
			}
		}
	} else if idx, ok := seen[strings.TrimSpace(t.DiffusionHead)]; !ok {
		return fmt.Errorf("config %q training.diffusion_head=%q does not match any training.heads[].name", source, t.DiffusionHead)
	} else if t.Heads[idx].Objective != ObjectiveBlockDiffusion {
		return fmt.Errorf("config %q training.diffusion_head=%q must select a block_diffusion head", source, t.DiffusionHead)
	}
	if t.RTD != nil {
		if t.RTD.Generator == "tied" {
			if strings.TrimSpace(t.RTD.GeneratorHead) == "" {
				t.RTD.GeneratorHead = t.ExportHead
			}
			idx, ok := seen[strings.TrimSpace(t.RTD.GeneratorHead)]
			if !ok {
				return fmt.Errorf("config %q training.rtd.generator_head=%q does not match any training.heads[].name", source, t.RTD.GeneratorHead)
			}
			h := &t.Heads[idx]
			if h.Objective != ObjectiveMLM && h.Objective != ObjectiveMNTP {
				return fmt.Errorf("config %q training.rtd.generator_head=%q must select an mlm or mntp head", source, t.RTD.GeneratorHead)
			}
			if h.OutputHead != MultiheadOutputBERTMLM && h.OutputHead != MultiheadOutputLinear {
				return fmt.Errorf("config %q training.rtd.generator_head=%q must emit vocab logits", source, t.RTD.GeneratorHead)
			}
		} else if strings.TrimSpace(t.RTD.GeneratorHead) != "" {
			return fmt.Errorf("config %q training.rtd.generator_head is only valid with generator=\"tied\"", source)
		}
	}
	if usesAdaLN && cfg.EffectiveNormPlacement() == NormPlacementPost {
		return fmt.Errorf("config %q multihead diffusion timestep_conditioning=\"adaln\" requires pre or sandwich norm placement in v1", source)
	}
	if !t.mlmMaskTokenIDSet || t.MLMMaskTokenID < 0 || t.MLMMaskTokenID >= cfg.VocabSize {
		return fmt.Errorf("config %q training.mlm_mask_token_id is required and must be in [0,%d) for multihead masked/diffusion heads", source, cfg.VocabSize)
	}
	for i, block := range cfg.Blocks {
		if block.WindowSize > 0 {
			return fmt.Errorf("config %q blocks[%d].window_size cannot be combined with training.objective=\"multihead\" in v1", source, i)
		}
		if blockTypeKey(block) == "custom" {
			return fmt.Errorf("config %q blocks[%d].type=\"custom\" cannot be combined with training.objective=\"multihead\" in v1", source, i)
		}
		switch blockTypeKey(block) {
		case "plain", "swiglu", "geglu", "mlp", "moe":
		default:
			return fmt.Errorf("config %q blocks[%d].type=%q cannot be combined with training.objective=\"multihead\" in v1; supported blocks are plain self-attention plus position-wise FFN/MoE blocks", source, i, block.Type)
		}
	}
	return nil
}

func validateRTDDedicatedGenerator(cfg *ArchConfig, source string) error {
	if cfg == nil || cfg.Training.RTD == nil {
		return fmt.Errorf("config %q training.rtd.generator is missing", source)
	}
	d := cfg.Training.RTD.DedicatedGenerator
	if d == nil {
		return fmt.Errorf("config %q training.rtd.generator=\"dedicated\" requires a generator object with model_dim, layers, and heads", source)
	}
	if strings.ToLower(strings.TrimSpace(d.Type)) != "dedicated" {
		return fmt.Errorf("config %q training.rtd.generator.type=%q must be \"dedicated\"", source, d.Type)
	}
	if d.ModelDim <= 0 {
		return fmt.Errorf("config %q training.rtd.generator.model_dim=%d must be > 0", source, d.ModelDim)
	}
	if d.Layers <= 0 {
		return fmt.Errorf("config %q training.rtd.generator.layers=%d must be > 0", source, d.Layers)
	}
	if d.Heads <= 0 {
		return fmt.Errorf("config %q training.rtd.generator.heads=%d must be > 0", source, d.Heads)
	}
	if d.ModelDim%d.Heads != 0 {
		return fmt.Errorf("config %q training.rtd.generator.model_dim=%d must be divisible by heads=%d", source, d.ModelDim, d.Heads)
	}
	if d.MLPMult <= 0 || math.IsNaN(d.MLPMult) || math.IsInf(d.MLPMult, 0) {
		return fmt.Errorf("config %q training.rtd.generator.mlp_mult=%g must be finite and > 0", source, d.MLPMult)
	}
	if d.GeneratorLossWeight < 0 || math.IsNaN(d.GeneratorLossWeight) || math.IsInf(d.GeneratorLossWeight, 0) {
		return fmt.Errorf("config %q training.rtd.generator.generator_loss_weight=%g must be finite and >= 0", source, d.GeneratorLossWeight)
	}
	return nil
}

func validateMultiheadDiffusionSpec(cfg *ArchConfig, d *DiffusionSpec, source string, headIdx int) error {
	if d.BlockSize <= 0 {
		return fmt.Errorf("config %q training.heads[%d].diffusion.block_size=%d must be > 0", source, headIdx, d.BlockSize)
	}
	if d.BlockSize > cfg.SeqLen || cfg.SeqLen%d.BlockSize != 0 {
		return fmt.Errorf("config %q training.heads[%d].diffusion.block_size=%d must divide seq_len=%d", source, headIdx, d.BlockSize, cfg.SeqLen)
	}
	if d.StepsPerBlock <= 0 {
		return fmt.Errorf("config %q training.heads[%d].diffusion.steps_per_block=%d must be > 0", source, headIdx, d.StepsPerBlock)
	}
	if !finiteInClosedRange(d.MinMaskFraction, 0, 1) || !finiteInClosedRange(d.MaxMaskFraction, 0, 1) || d.MaxMaskFraction <= 0 || d.MinMaskFraction > d.MaxMaskFraction {
		return fmt.Errorf("config %q training.heads[%d].diffusion has invalid mask fraction range [%g,%g]", source, headIdx, d.MinMaskFraction, d.MaxMaskFraction)
	}
	if !finiteInClosedRange(d.ConfidenceThreshold, 0, 1) {
		return fmt.Errorf("config %q training.heads[%d].diffusion.confidence_threshold=%g must be in [0,1]", source, headIdx, d.ConfidenceThreshold)
	}
	if d.CommitFloor <= 0 || d.CommitFloor > d.BlockSize {
		return fmt.Errorf("config %q training.heads[%d].diffusion.commit_floor=%d must be in [1,block_size=%d]", source, headIdx, d.CommitFloor, d.BlockSize)
	}
	switch d.TimestepConditioning {
	case DiffusionTimestepConditioningNone, DiffusionTimestepConditioningAdaLN:
	default:
		return fmt.Errorf("config %q training.heads[%d].diffusion.timestep_conditioning=%q must be \"none\" or \"adaln\"", source, headIdx, d.TimestepConditioning)
	}
	if d.TimestepConditioning == DiffusionTimestepConditioningAdaLN && d.TimestepConditionDim <= 0 {
		return fmt.Errorf("config %q training.heads[%d].diffusion.timestep_conditioning_dim=%d must be > 0", source, headIdx, d.TimestepConditionDim)
	}
	return nil
}

func multiheadUsesAdaLN(t TrainingSpec) bool {
	if !t.MultiheadEnabled() {
		return false
	}
	for _, h := range t.Heads {
		if h.Objective == ObjectiveBlockDiffusion && h.Diffusion != nil && h.Diffusion.TimestepConditioning == DiffusionTimestepConditioningAdaLN {
			return true
		}
	}
	return false
}

func multiheadAdaLNDim(t TrainingSpec) int {
	for _, h := range t.Heads {
		if h.Objective == ObjectiveBlockDiffusion && h.Diffusion != nil && h.Diffusion.TimestepConditioning == DiffusionTimestepConditioningAdaLN {
			if h.Diffusion.TimestepConditionDim > 0 {
				return h.Diffusion.TimestepConditionDim
			}
		}
	}
	return 0
}

func collectMultiheadWeightShapesFromConfig(cfg *ArchConfig) ([]WeightMeta, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
	}
	shapes := []WeightMeta{{Name: "embed", Shape: []int{cfg.VocabSize, cfg.ModelDim}}}
	shapes = append(shapes, positionalEmbeddingWeightShapes(cfg.ModelDim, cfg.EffectiveMaxPositions(), cfg.EffectivePositionalEmbedding())...)
	shapes = append(shapes, charWeightShapes(cfg.ModelDim, cfg.CharVocabSize, cfg.EffectiveCharDim())...)
	shapes = append(shapes, bigramWeightShapes(cfg.ModelDim, cfg.BigramVocabSize, cfg.EffectiveBigramDim())...)
	shapes = append(shapes, trigramWeightShapes(cfg.ModelDim, cfg.TrigramVocabSize, cfg.EffectiveTrigramDim())...)
	sharedRel, err := sharedRelativeAttentionWeightShapes(cfg.ModelDim, cfg.Blocks)
	if err != nil {
		return nil, err
	}
	shapes = append(shapes, sharedRel...)
	blockShapes, err := multiheadTrunkWeightShapes(cfg)
	if err != nil {
		return nil, fmt.Errorf("blocks: %w", err)
	}
	shapes = append(shapes, blockShapes...)
	if multiheadUsesAdaLN(cfg.Training) {
		condDim := multiheadAdaLNDim(cfg.Training)
		for i := range cfg.Blocks {
			shapes = append(shapes,
				WeightMeta{Name: fmt.Sprintf("adaln_%d_w1", i), Shape: []int{1, condDim}},
				WeightMeta{Name: fmt.Sprintf("adaln_%d_w2", i), Shape: []int{condDim, 2 * cfg.ModelDim}, InitZero: true},
			)
		}
	}
	for _, h := range cfg.Training.Heads {
		shapes = append(shapes, multiheadHeadWeightShapes(cfg.ModelDim, cfg.VocabSize, cfg.Blocks, cfg.EffectiveNormSpec(), h)...)
	}
	rtdGeneratorShapes, err := rtdDedicatedGeneratorWeightShapes(cfg)
	if err != nil {
		return nil, err
	}
	shapes = append(shapes, rtdGeneratorShapes...)
	return shapes, nil
}

func multiheadTrunkWeightShapes(cfg *ArchConfig) ([]WeightMeta, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
	}
	blocks := multiheadResolvedBlocks(cfg.Blocks)
	sharedRel, err := newSharedRelativeAttentionPlan(cfg.Blocks)
	if err != nil {
		return nil, err
	}
	return streamWeightShapesWithRefsAndParallelOptions(blocks, identityWeightRefs(blocks), cfg.ModelDim, cfg.SeqLen, 1, cfg.VocabSize, multiheadTrunkWeightShapeOptions(cfg, sharedRel), false)
}

func multiheadTrunkWeightShapeOptions(cfg *ArchConfig, sharedRel sharedRelativeAttentionPlan) EmitOptions {
	return EmitOptions{
		MLPMult:             cfg.EffectiveMLPMult(),
		BlockScales:         cfg.BlockScales,
		ResidMix:            cfg.ResidMix,
		Norm:                cfg.EffectiveNormSpec(),
		NormPlacement:       cfg.EffectiveNormPlacement(),
		FFNInternalNorm:     cfg.FFNInternalNorm,
		PositionalEmbedding: cfg.EffectivePositionalEmbedding(),
		sharedRelative:      sharedRel,
	}
}

func multiheadHeadWeightShapes(modelDim, vocabSize int, blocks []BlockSpec, norm NormSpec, h MultiheadHeadSpec) []WeightMeta {
	prefix := "head_" + h.Name
	var shapes []WeightMeta
	if h.LayerAggregation == LayerAggregationDWA {
		if n, err := dwaSublayerCount(blocks); err == nil {
			shapes = append(shapes, WeightMeta{Name: prefix + "_dwa_alpha", Shape: []int{n + 1}, InitMode: dwaAlphaInitMode})
		}
	}
	if h.FinalNorm {
		shapes = append(shapes, normWeights(prefix+"_final_norm", modelDim, norm)...)
	}
	switch h.OutputHead {
	case MultiheadOutputBERTMLM:
		shapes = append(shapes,
			WeightMeta{Name: prefix + "_mlm_dense", Shape: []int{modelDim, modelDim}},
			WeightMeta{Name: prefix + "_mlm_dense_bias", Shape: []int{modelDim}, InitZero: true},
			WeightMeta{Name: prefix + "_mlm_output_bias", Shape: []int{vocabSize}, InitZero: true},
		)
	case MultiheadOutputLinear:
		if !h.TieEmbeddings {
			shapes = append(shapes, WeightMeta{Name: prefix + "_proj", Shape: []int{modelDim, vocabSize}})
		}
	case MultiheadOutputBinary:
		shapes = append(shapes,
			WeightMeta{Name: prefix + "_binary_proj", Shape: []int{modelDim, 1}},
			WeightMeta{Name: prefix + "_binary_bias", Shape: []int{1}, InitZero: true},
		)
	}
	return shapes
}
