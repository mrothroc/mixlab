package arch

import "fmt"

func convertToClassificationIR(cfg *ArchConfig, state TrainingProgramState, prog *Program) (*Program, error) {
	if cfg == nil || cfg.Training.Classification == nil || prog == nil {
		return nil, fmt.Errorf("classification IR requires config, classification spec, and base program")
	}
	B := cfg.Training.BatchTokens / cfg.SeqLen
	if B <= 0 {
		return nil, fmt.Errorf("classification requires batch_tokens to be a positive multiple of seq_len")
	}
	D := cfg.ModelDim
	N := cfg.Training.Classification.NumLabels

	// Keep the complete backbone through its normalized [B,T,D] output and
	// discard the unused LM projection/loss tail.
	lastHiddenOp := -1
	for i, op := range prog.Ops {
		for _, output := range op.Outputs {
			if output == "x_hidden" {
				lastHiddenOp = i
			}
		}
	}
	if lastHiddenOp < 0 {
		return nil, fmt.Errorf("classification base program does not expose x_hidden")
	}
	prog.Ops = prog.Ops[:lastHiddenOp+1]
	filteredInputs := prog.Inputs[:0]
	for _, input := range prog.Inputs {
		switch input.Name {
		case "targets", "loss_mask", "teacher_probs", "distill_loss_mask",
			"word_struct_targets", "word_struct_loss_mask", "invariance_loss_mask",
			"pll_margin_loss_mask", "data2vec_targets", "data2vec_loss_mask",
			"first_byte_valid":
			continue
		default:
			filteredInputs = append(filteredInputs, input)
		}
	}
	prog.Inputs = filteredInputs
	prog.Outputs = nil

	baseWeights := prog.NumWeights
	prog.NumWeights += len(classificationWeightShapes(D, cfg.Training.Classification))
	prog.DeclareInput("classification_labels", TensorInt32, []int{B})

	pooled := "classification_pooled"
	switch cfg.EffectiveClassificationPooling() {
	case ClassificationPoolingLast:
		prog.DeclareInput("classification_positions", TensorInt32, []int{B})
		prog.Embed("x_final_norm", "classification_positions", pooled)
	case ClassificationPoolingMean:
		prog.DeclareInput("classification_valid_mask", TensorFloat32, []int{B, cfg.SeqLen})
		prog.Reshape("classification_valid_mask", []int{B, cfg.SeqLen, 1}, "classification_valid_mask_btd")
		prog.Mul("x_hidden", "classification_valid_mask_btd", "classification_masked_hidden")
		prog.MeanAxis("classification_masked_hidden", 1, "classification_masked_mean")
		prog.MeanAxis("classification_valid_mask_btd", 1, "classification_valid_mean")
		prog.DivSafe("classification_masked_mean", "classification_valid_mean", 1e-12, pooled)
	default:
		return nil, fmt.Errorf("unsupported classification pooling %q", cfg.EffectiveClassificationPooling())
	}

	classifierInput := pooled
	dropout := cfg.EffectiveClassifierDropout()
	if state.DropoutInactive {
		dropout = 0
	}
	if dropout > 0 {
		prog.Dropout(pooled, dropout, "classification_pooled_dropout")
		classifierInput = "classification_pooled_dropout"
	}
	prog.MatMul(classifierInput, weightName(baseWeights), "classification_logits_linear")
	prog.Add("classification_logits_linear", weightName(baseWeights+1), "classification_logits")
	prog.CrossEntropy("classification_logits", "classification_labels", "classification_task_loss")
	prog.ScalarMul("classification_task_loss", 1, "loss")
	prog.ScalarMul("classification_task_loss", 1, "eval_loss")
	moeEnabled := emitMoEAuxiliaryAggregatesIR(prog, true)

	prog.DeclareOutput("loss", TensorFloat32, []int{1})
	prog.DeclareOutput("eval_loss", TensorFloat32, []int{1})
	prog.DeclareOutput("classification_logits", TensorFloat32, []int{B, N})
	prog.DeclareOutput("x_hidden", TensorFloat32, []int{B, cfg.SeqLen, D})
	if moeEnabled {
		prog.DeclareOutput("moe_aux_loss", TensorFloat32, []int{1})
		prog.DeclareOutput("moe_router_entropy", TensorFloat32, []int{1})
	}
	return prog, nil
}
