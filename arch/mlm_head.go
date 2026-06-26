package arch

import (
	"fmt"
	"strings"
)

const (
	MLMHeadLinear = "linear"
	MLMHeadBERT   = "bert"

	MLMHeadDenseWeightName = "mlm_head_dense"
	MLMHeadDenseBiasName   = "mlm_head_dense_bias"
	MLMHeadOutputBiasName  = "mlm_head_output_bias"
)

func normalizeMLMHead(raw string) string {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "linear", "none", "default":
		return MLMHeadLinear
	case "bert", "bert_mlm", "bert-style", "bert_style":
		return MLMHeadBERT
	default:
		return strings.ToLower(strings.TrimSpace(raw))
	}
}

func (cfg *ArchConfig) EffectiveMLMHead() string {
	if cfg == nil {
		return MLMHeadLinear
	}
	return normalizeMLMHead(cfg.MLMHead)
}

func mlmHeadWeightShapes(modelDim, vocabSize int, mlmHead string) []WeightMeta {
	if normalizeMLMHead(mlmHead) != MLMHeadBERT {
		return nil
	}
	return []WeightMeta{
		{Name: MLMHeadDenseWeightName, Shape: []int{modelDim, modelDim}},
		{Name: MLMHeadDenseBiasName, Shape: []int{modelDim}, InitZero: true},
		{Name: MLMHeadOutputBiasName, Shape: []int{vocabSize}, InitZero: true},
	}
}

func mlmHeadWeightCount(modelDim, vocabSize int, mlmHead string) int {
	return len(mlmHeadWeightShapes(modelDim, vocabSize, mlmHead))
}

func validateMLMHead(cfg *ArchConfig, source string) error {
	cfg.MLMHead = normalizeMLMHead(cfg.MLMHead)
	switch cfg.MLMHead {
	case MLMHeadLinear, MLMHeadBERT:
	default:
		return fmt.Errorf("config %q has invalid mlm_head=%q (must be \"linear\" or \"bert\")", source, cfg.MLMHead)
	}
	if cfg.MLMHead != MLMHeadBERT {
		return nil
	}
	obj := cfg.Training.EffectiveObjective()
	maskedCapable := obj == ObjectiveMLM || obj == ObjectiveMNTP || obj == ObjectiveBlockDiffusion || (obj == ObjectiveHybrid && cfg.Training.HybridCLMFraction < 1)
	if !maskedCapable {
		return fmt.Errorf("config %q sets mlm_head=\"bert\" but training.objective=%q has no masked objective path", source, obj)
	}
	if !cfg.TieEmbeddings {
		return fmt.Errorf("config %q sets mlm_head=\"bert\" but tie_embeddings is false; BERT MLM output weight is tied to the input embedding", source)
	}
	return nil
}

func emitBERTMLMHeadIR(prog *Program, hidden string, wi int, modelDim, vocabSize int, eps, dropout float32) (string, int, error) {
	if modelDim <= 0 || vocabSize <= 0 {
		return "", wi, fmt.Errorf("invalid BERT MLM head shape D=%d V=%d", modelDim, vocabSize)
	}
	prog.LayerNormNoAffine(hidden, "mlm_head_ln1", eps)
	prog.MatMul("mlm_head_ln1", weightName(wi), "mlm_head_dense_mm")
	wi++
	prog.Add("mlm_head_dense_mm", weightName(wi), "mlm_head_dense_out")
	wi++
	prog.GELU("mlm_head_dense_out", "mlm_head_gelu")
	prog.LayerNormNoAffine("mlm_head_gelu", "mlm_head_ln2", eps)
	headInput := "mlm_head_ln2"
	if dropout > 0 {
		prog.Dropout(headInput, dropout, "mlm_head_dropout")
		headInput = "mlm_head_dropout"
	}
	prog.Transpose(weightName(0), []int{1, 0}, "mlm_head_tied_weight")
	prog.MatMul(headInput, "mlm_head_tied_weight", "mlm_head_logits_mm")
	prog.Add("mlm_head_logits_mm", weightName(wi), "mlm_head_logits")
	wi++
	return "mlm_head_logits", wi, nil
}
