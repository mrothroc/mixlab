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
	maskedCapable := obj == ObjectiveMLM || obj == ObjectiveMNTP || obj == ObjectiveBlockDiffusion ||
		obj == ObjectiveClassification || cfg.Training.HybridHasMaskedSteps()
	if !maskedCapable {
		return fmt.Errorf("config %q sets mlm_head=\"bert\" but training.objective=%q has no masked objective path", source, obj)
	}
	if !cfg.TieEmbeddings {
		return fmt.Errorf("config %q sets mlm_head=\"bert\" but tie_embeddings is false; BERT MLM output weight is tied to the input embedding", source)
	}
	return nil
}

func emitBERTMLMHeadIR(prog *Program, hidden string, wi int, modelDim, vocabSize int, eps, dropout float32) (string, int, error) {
	return emitBERTMLMHeadIRTiedTo(prog, "mlm_head", hidden, wi, weightName(0), modelDim, vocabSize, eps, dropout)
}

func emitBERTMLMHeadIRTiedTo(prog *Program, prefix, hidden string, wi int, tiedEmbeddingWeight string, modelDim, vocabSize int, eps, dropout float32) (string, int, error) {
	if modelDim <= 0 || vocabSize <= 0 {
		return "", wi, fmt.Errorf("invalid BERT MLM head shape D=%d V=%d", modelDim, vocabSize)
	}
	ln1 := prefix + "_ln1"
	denseMM := prefix + "_dense_mm"
	denseOut := prefix + "_dense_out"
	gelu := prefix + "_gelu"
	ln2 := prefix + "_ln2"
	drop := prefix + "_dropout"
	tied := prefix + "_tied_weight"
	logitsMM := prefix + "_logits_mm"
	logits := prefix + "_logits"
	prog.LayerNormNoAffine(hidden, ln1, eps)
	prog.MatMul(ln1, weightName(wi), denseMM)
	wi++
	prog.Add(denseMM, weightName(wi), denseOut)
	wi++
	prog.GELU(denseOut, gelu)
	prog.LayerNormNoAffine(gelu, ln2, eps)
	headInput := ln2
	if dropout > 0 {
		prog.Dropout(headInput, dropout, drop)
		headInput = drop
	}
	prog.Transpose(tiedEmbeddingWeight, []int{1, 0}, tied)
	prog.MatMul(headInput, tied, logitsMM)
	prog.Add(logitsMM, weightName(wi), logits)
	wi++
	return logits, wi, nil
}
