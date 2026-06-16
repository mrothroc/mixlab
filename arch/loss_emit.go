package arch

import (
	"fmt"
	"strings"
)

// emitLanguageModelLossIR emits the IR ops that materialise the next-token
// language-model loss. When mtp is enabled it produces the weighted multi-token
// prediction loss; otherwise it produces the standard cross-entropy loss.
//
// When firstByteMask is true, the training-time loss is computed against the
// first-byte masked softmax (only valid UTF-8 first-byte tokens contribute to
// the denominator at codepoint-boundary targets). The unmasked cross-entropy
// is still emitted as "eval_loss" so validation BPB and per-token NLL exports
// remain comparable to the unmasked baseline.
func emitLanguageModelLossIR(prog *Program, logits, targets string, B, T, V int, mtp *MTPSpec, firstByteMask bool) error {
	if mtp == nil || mtp.EffectiveN() <= 1 {
		if firstByteMask {
			prog.CrossEntropy(logits, targets, "eval_loss")
			prog.FirstByteMaskedCrossEntropy(logits, targets, "first_byte_valid", "loss")
		} else {
			prog.CrossEntropy(logits, targets, "loss")
		}
		prog.CrossEntropyPerToken(logits, targets, "per_token_nll")
		return nil
	}

	n := mtp.EffectiveN()
	if n > T {
		return fmt.Errorf("mtp.n=%d must be <= sequence length %d", n, T)
	}
	weights := mtp.EffectiveLossWeights()
	var weightSum float32
	for _, w := range weights {
		weightSum += w
	}
	if weightSum <= 0 {
		return fmt.Errorf("mtp loss weights must sum to > 0")
	}

	if firstByteMask {
		prog.CrossEntropy(logits, targets, "eval_loss")
		prog.FirstByteMaskedCrossEntropy(logits, targets, "first_byte_valid", "mtp_loss_0")
	} else {
		prog.CrossEntropy(logits, targets, "mtp_loss_0")
		prog.ScalarMul("mtp_loss_0", 1.0, "eval_loss")
	}
	prog.CrossEntropyPerToken(logits, targets, "per_token_nll")

	var accum string
	if weights[0] != 0 {
		accum = "mtp_weighted_0"
		prog.ScalarMul("mtp_loss_0", weights[0]/weightSum, accum)
	}

	prog.Reshape(logits, []int{B, T, V}, "mtp_logits_btv")
	prog.Reshape(targets, []int{B, T}, "mtp_targets_bt")
	for i := 1; i < n; i++ {
		validT := T - i
		logitSlice := fmt.Sprintf("mtp_logits_%d_slice", i)
		logitFlat := fmt.Sprintf("mtp_logits_%d_flat", i)
		targetSlice := fmt.Sprintf("mtp_targets_%d_slice", i)
		targetFlat := fmt.Sprintf("mtp_targets_%d_flat", i)
		loss := fmt.Sprintf("mtp_loss_%d", i)

		prog.Slice("mtp_logits_btv", 0, validT, 1, 1, logitSlice)
		prog.Reshape(logitSlice, []int{B * validT, V}, logitFlat)
		prog.Slice("mtp_targets_bt", i, T, 1, 1, targetSlice)
		prog.Reshape(targetSlice, []int{B * validT}, targetFlat)
		if firstByteMask {
			prog.FirstByteMaskedCrossEntropy(logitFlat, targetFlat, "first_byte_valid", loss)
		} else {
			prog.CrossEntropy(logitFlat, targetFlat, loss)
		}
		if weights[i] == 0 {
			continue
		}
		weighted := fmt.Sprintf("mtp_weighted_%d", i)
		prog.ScalarMul(loss, weights[i]/weightSum, weighted)
		if accum == "" {
			accum = weighted
			continue
		}
		next := fmt.Sprintf("mtp_accum_%d", i)
		prog.Add(accum, weighted, next)
		accum = next
	}
	if accum == "" {
		return fmt.Errorf("mtp loss weights must include at least one positive coefficient")
	}
	prog.ScalarMul(accum, 1.0, "loss")
	return nil
}

// emitMaskedLanguageModelLossIR emits the masked-objective loss stack used by
// MLM and MNTP. The training loss is averaged over rows where loss_mask > 0;
// eval_loss remains the dense cross-entropy for callers that need an unmasked
// scalar, and per_token_nll is zeroed for ignored rows.
func emitMaskedLanguageModelLossIR(prog *Program, logits, targets, lossMask string) {
	prog.CrossEntropy(logits, targets, "eval_loss")
	prog.MaskedCrossEntropy(logits, targets, lossMask, "loss")
	prog.MaskedCrossEntropyPerToken(logits, targets, lossMask, "per_token_nll")
}

func emitDistillationLanguageModelLossIR(prog *Program, logits, targets, teacherProbs string, ceWeight, klWeight float64) error {
	if ceWeight < 0 || klWeight < 0 || ceWeight+klWeight <= 0 {
		return fmt.Errorf("distillation loss weights must be non-negative and sum to > 0")
	}
	prog.CrossEntropy(logits, targets, "eval_loss")
	prog.CrossEntropyPerToken(logits, targets, "per_token_nll")
	prog.DistillationKL(logits, teacherProbs, "distill_kl_loss")

	accum := ""
	if ceWeight != 0 {
		prog.ScalarMul("eval_loss", float32(ceWeight), "distill_ce_weighted")
		accum = "distill_ce_weighted"
	}
	if klWeight != 0 {
		prog.ScalarMul("distill_kl_loss", float32(klWeight), "distill_kl_weighted")
		if accum == "" {
			accum = "distill_kl_weighted"
		} else {
			prog.Add(accum, "distill_kl_weighted", "distill_loss_sum")
			accum = "distill_loss_sum"
		}
	}
	prog.ScalarMul(accum, 1.0, "loss")
	return nil
}

func emitData2VecPredictorIR(prog *Program, spec *Data2VecSpec, wi int) int {
	if spec == nil || spec.LossWeight <= 0 {
		return wi
	}
	if spec.PredictorHidden > 0 {
		prog.MatMul("x_final_norm", weightName(wi), "data2vec_pred_hidden")
		wi++
		prog.GELU("data2vec_pred_hidden", "data2vec_pred_hidden_gelu")
		prog.MatMul("data2vec_pred_hidden_gelu", weightName(wi), "data2vec_pred")
		wi++
		return wi
	}
	prog.MatMul("x_final_norm", weightName(wi), "data2vec_pred")
	return wi + 1
}

func emitData2VecLossIR(prog *Program, spec *Data2VecSpec) {
	if spec == nil || spec.LossWeight <= 0 {
		return
	}
	prog.ScalarMul("loss", 1.0, "primary_loss")
	prog.MaskedSmoothL1("data2vec_pred", "data2vec_targets", "data2vec_loss_mask", float32(spec.SmoothL1Beta), "data2vec_loss")
	prog.ScalarMul("data2vec_loss", float32(spec.LossWeight), "data2vec_loss_weighted")
	prog.Add("primary_loss", "data2vec_loss_weighted", "loss")
}

func emitZLossIR(prog *Program, logits string, weight float64, taskLossHasEvalLoss bool) bool {
	if weight <= 0 {
		return taskLossHasEvalLoss
	}
	if !taskLossHasEvalLoss {
		prog.ScalarMul("loss", 1.0, "eval_loss")
		taskLossHasEvalLoss = true
	}
	prog.ZLoss(logits, "z_loss_raw")
	prog.ScalarMul("z_loss_raw", float32(weight), "z_loss_weighted")
	prog.Add("loss", "z_loss_weighted", "loss")
	return taskLossHasEvalLoss
}

func collectMoEAuxiliaryOutputs(prog *Program) (auxLosses []string, entropies []string) {
	for _, op := range prog.Ops {
		for _, out := range op.Outputs {
			switch {
			case strings.HasSuffix(out, "_moe_aux_loss") && out != "moe_aux_loss":
				auxLosses = append(auxLosses, out)
			case strings.HasSuffix(out, "_moe_router_entropy") && out != "moe_router_entropy":
				entropies = append(entropies, out)
			}
		}
	}
	return auxLosses, entropies
}

func emitMoEAuxiliaryLossIR(prog *Program, auxLosses []string) {
	if len(auxLosses) == 0 {
		return
	}
	accum := auxLosses[0]
	if len(auxLosses) == 1 {
		prog.ScalarMul(accum, 1.0, "moe_aux_loss")
		return
	}
	for i := 1; i < len(auxLosses); i++ {
		next := fmt.Sprintf("moe_aux_accum_%d", i)
		prog.Add(accum, auxLosses[i], next)
		accum = next
	}
	prog.ScalarMul(accum, 1.0, "moe_aux_loss")
}

func emitMoERouterEntropyIR(prog *Program, entropies []string) {
	if len(entropies) == 0 {
		return
	}
	accum := entropies[0]
	if len(entropies) > 1 {
		for i := 1; i < len(entropies); i++ {
			next := fmt.Sprintf("moe_entropy_accum_%d", i)
			prog.Add(accum, entropies[i], next)
			accum = next
		}
	}
	prog.ScalarMul(accum, 1.0/float32(len(entropies)), "moe_router_entropy")
}

func emitMoEAuxiliaryAggregatesIR(prog *Program, taskLossHasEvalLoss bool) bool {
	auxLosses, entropies := collectMoEAuxiliaryOutputs(prog)
	if len(auxLosses) == 0 {
		return false
	}
	if !taskLossHasEvalLoss {
		prog.ScalarMul("loss", 1.0, "eval_loss")
	}
	prog.ScalarMul("loss", 1.0, "task_loss")
	emitMoEAuxiliaryLossIR(prog, auxLosses)
	emitMoERouterEntropyIR(prog, entropies)
	prog.Add("task_loss", "moe_aux_loss", "loss")
	return true
}
