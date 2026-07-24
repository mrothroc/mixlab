package arch

import (
	"fmt"
	"strings"
)

// RCEquivarianceEnabled reports whether the model uses shared-weight
// reverse-complement branches.
func (c *ArchConfig) RCEquivarianceEnabled() bool {
	return c != nil && c.RCEquivariant
}

func validateRCEquivariance(cfg *ArchConfig, source string) error {
	if cfg == nil || !cfg.RCEquivariant {
		return nil
	}
	objective := cfg.Training.EffectiveObjective()
	if objective != ObjectiveMLM && objective != ObjectiveClassification {
		return fmt.Errorf("config %q rc_equivariant=true supports training.objective=%q or %q in v1; got %q",
			source, ObjectiveMLM, ObjectiveClassification, objective)
	}
	if cfg.Training.ReverseComplementProb != 0 {
		return fmt.Errorf("config %q rc_equivariant=true cannot be combined with training.reverse_complement_prob; RCPS already evaluates both orientations", source)
	}
	if objective == ObjectiveClassification && cfg.EffectiveClassificationPooling() != ClassificationPoolingMean {
		return fmt.Errorf("config %q rc_equivariant classification requires training.classification.pooling=%q", source, ClassificationPoolingMean)
	}
	if cfg.Dropout != 0 || cfg.AttnDropout != 0 || cfg.HiddenDropout != 0 ||
		cfg.EmbeddingDropout != 0 || cfg.EffectiveClassifierDropout() != 0 {
		return fmt.Errorf("config %q rc_equivariant=true requires dropout, attention dropout, embedding dropout, and classifier dropout to be zero in v1", source)
	}
	if cfg.CharVocabSize > 0 || cfg.BigramVocabSize > 0 || cfg.TrigramVocabSize > 0 || cfg.SmearEmbeddings {
		return fmt.Errorf("config %q rc_equivariant=true does not support char, bigram, trigram, or smear embedding features in v1", source)
	}
	if cfg.EffectivePositionalEmbedding() == PositionalEmbeddingLearnedAbsolute {
		return fmt.Errorf("config %q rc_equivariant=true does not support positional_embedding=%q in v1", source, PositionalEmbeddingLearnedAbsolute)
	}
	if cfg.BlockScales || cfg.ResidMix || cfg.ParallelResidual || cfg.UNet || cfg.Backout != nil ||
		len(cfg.Recurrence) > 0 || len(cfg.RecurrencePhases) > 0 || len(cfg.ExecutionOrder) > 0 {
		return fmt.Errorf("config %q rc_equivariant=true does not support block_scales, resid_mix, parallel groups, U-Net, backout, or recurrence in v1", source)
	}
	if cfg.EffectiveLayerAggregation() != LayerAggregationNone {
		return fmt.Errorf("config %q rc_equivariant=true requires layer_aggregation=%q in v1", source, LayerAggregationNone)
	}
	if cfg.MTP != nil || cfg.Training.FirstByteMask || cfg.Training.Distillation != nil ||
		cfg.Training.Data2Vec != nil || cfg.Training.ExampleFraming != nil ||
		cfg.Training.RTD != nil || cfg.Training.MinimalPair != nil ||
		cfg.Training.Invariance != nil || cfg.Training.PLLMargin != nil ||
		cfg.Training.WordStructuralObjective != nil || cfg.Training.Diffusion != nil ||
		len(cfg.Training.Heads) > 0 || len(cfg.Training.SeqLenSchedule) > 0 {
		return fmt.Errorf("config %q rc_equivariant=true cannot be combined with auxiliary, multihead, diffusion, framing, or sequence-schedule training features in v1", source)
	}

	mixers := 0
	for i, block := range cfg.Blocks {
		if strings.TrimSpace(block.WeightGroup) != "" || block.ParallelResidual != nil || block.ParallelGroup > 0 {
			return fmt.Errorf("config %q rc_equivariant=true does not support weight groups or parallel block composition (blocks[%d]) in v1", source, i)
		}
		switch blockTypeKey(block) {
		case "plain":
			mask := resolvedPlainAttentionMask(block, objective)
			if mask != AttentionMaskBidirectional && mask != AttentionMaskNone {
				return fmt.Errorf("config %q rc_equivariant plain blocks require bidirectional or none attention; blocks[%d] resolves to %q", source, i, mask)
			}
			if block.KVSource > 0 {
				return fmt.Errorf("config %q rc_equivariant=true does not support blocks[%d].kv_source in v1", source, i)
			}
			mixers++
		case "gated_deltanet":
			if objective != ObjectiveClassification {
				return fmt.Errorf("config %q rc_equivariant gated_deltanet is supported only for one-record classification in v1", source)
			}
			mixers++
		case "swiglu", "geglu", "mlp":
			// Pointwise channel mixers preserve the paired representation.
		default:
			return fmt.Errorf("config %q rc_equivariant=true does not support blocks[%d].type=%q in v1", source, i, block.Type)
		}
	}
	if mixers == 0 {
		return fmt.Errorf("config %q rc_equivariant=true requires at least one plain or gated_deltanet token mixer", source)
	}
	return nil
}

func emitRCEquivariantInputIR(prog *Program, batchSize, seqLen, modelDim, nextWeightIndex int) (int, error) {
	if prog == nil || batchSize <= 0 || seqLen <= 0 || modelDim <= 0 {
		return nextWeightIndex, fmt.Errorf("invalid RC-equivariant input shape B=%d T=%d D=%d", batchSize, seqLen, modelDim)
	}
	prog.Embed(weightName(0), "tokens", "rc_forward_embed")
	prog.Reshape("rc_forward_embed", []int{batchSize * seqLen, modelDim}, "rc_forward_x")
	prog.Embed(weightName(0), "rc_tokens", "rc_reverse_embed")
	prog.Reshape("rc_reverse_embed", []int{batchSize * seqLen, modelDim}, "rc_reverse_x")
	prog.Concat("rc_forward_x", "rc_reverse_x", 0, "x")
	return nextWeightIndex, nil
}

// emitRCEquivariantHiddenIR splits the shared-backbone output, aligns the
// reverse-complement branch to forward positions, and exposes both the
// half-swapping representation and its invariant average.
func emitRCEquivariantHiddenIR(prog *Program, paired string, batchSize, seqLen, modelDim int) string {
	rows := batchSize * seqLen
	prog.Slice(paired, 0, rows, 1, 0, "rc_forward_hidden")
	prog.Slice(paired, rows, 2*rows, 1, 0, "rc_reverse_hidden_raw")
	prog.Embed("rc_reverse_hidden_raw", "rc_alignment_positions", "rc_reverse_hidden_aligned")
	prog.Concat("rc_forward_hidden", "rc_reverse_hidden_aligned", 1, "rc_equivariant_hidden_flat")
	prog.Reshape("rc_equivariant_hidden_flat", []int{batchSize, seqLen, 2 * modelDim}, "rc_equivariant_hidden")
	prog.Add("rc_forward_hidden", "rc_reverse_hidden_aligned", "rc_hidden_sum")
	prog.ScalarMul("rc_hidden_sum", 0.5, "x_final_norm")
	return "x_final_norm"
}

func emitRCEquivariantLMHeadIR(
	prog *Program,
	wi, modelDim, vocabSize int,
	maskedObjective bool,
	mlmHead string,
	useTiedHead bool,
	normEps, dropout float32,
) (string, int, error) {
	// The hidden split already exists for x_hidden; project each orientation
	// separately so vocabulary complementation occurs before averaging.
	forwardLogits := "rc_forward_logits"
	reverseLogits := "rc_reverse_logits_raw"
	nextWI := wi
	if maskedObjective && mlmHead == MLMHeadBERT {
		var err error
		forwardLogits, nextWI, err = emitBERTMLMHeadIRTiedTo(
			prog, "rc_forward_mlm_head", "rc_forward_hidden", wi, weightName(0),
			modelDim, vocabSize, normEps, dropout,
		)
		if err != nil {
			return "", wi, err
		}
		var reverseWI int
		reverseLogits, reverseWI, err = emitBERTMLMHeadIRTiedTo(
			prog, "rc_reverse_mlm_head", "rc_reverse_hidden_raw", wi, weightName(0),
			modelDim, vocabSize, normEps, dropout,
		)
		if err != nil {
			return "", wi, err
		}
		if reverseWI != nextWI {
			return "", wi, fmt.Errorf("RC-equivariant BERT MLM head weight mismatch: forward=%d reverse=%d", nextWI, reverseWI)
		}
	} else {
		headWeight := weightName(1)
		if useTiedHead {
			prog.Transpose(weightName(0), []int{1, 0}, "rc_tied_head")
			headWeight = "rc_tied_head"
		}
		prog.MatMul("rc_forward_hidden", headWeight, forwardLogits)
		prog.MatMul("rc_reverse_hidden_raw", headWeight, reverseLogits)
		if mlmHead == MLMHeadBERT {
			nextWI += mlmHeadWeightCount(modelDim, vocabSize, mlmHead)
		}
	}
	prog.Embed(reverseLogits, "rc_alignment_positions", "rc_reverse_logits_position_aligned")
	prog.Transpose("rc_reverse_logits_position_aligned", []int{1, 0}, "rc_reverse_logits_vbt")
	prog.Embed("rc_reverse_logits_vbt", "rc_complement_ids", "rc_reverse_logits_complemented_vbt")
	prog.Transpose("rc_reverse_logits_complemented_vbt", []int{1, 0}, "rc_reverse_logits_aligned")
	prog.Add(forwardLogits, "rc_reverse_logits_aligned", "rc_logits_sum")
	prog.ScalarMul("rc_logits_sum", 0.5, "rc_logits")
	return "rc_logits", nextWI, nil
}
