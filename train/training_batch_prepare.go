package train

import "fmt"

func prepareTrainingBatch(
	cfg *ArchConfig,
	trainer GPUTrainer,
	raw trainBatch,
	step int,
	objective string,
	batchSize int,
	seqLen int,
	pairSampler *minimalPairSampler,
	invarianceSampler *invariancePairSampler,
	pllMarginSampler *pllMarginPairSampler,
	distiller *distillationEnsemble,
	data2vec *data2VecTeacher,
) (objectiveBatch, error) {
	prepared, err := prepareObjectiveBatchWithSeqLen(cfg, raw, step, objective, seqLen)
	if err != nil {
		return objectiveBatch{}, trainingStepError(step, "objective batch", err)
	}
	prepared, err = maybeAttachMinimalPairs(pairSampler, cfg, step, prepared, batchSize, seqLen)
	if err != nil {
		return objectiveBatch{}, trainingStepError(step, "minimal-pair batch", err)
	}
	prepared, err = maybeAttachRTDCorruption(trainer, cfg, raw, step, prepared, batchSize, seqLen, objective)
	if err != nil {
		return objectiveBatch{}, err
	}
	prepared, err = maybeAttachInvariancePairs(invarianceSampler, cfg, step, prepared, batchSize, seqLen, objective)
	if err != nil {
		return objectiveBatch{}, trainingStepError(step, "invariance batch", err)
	}
	prepared, err = maybeAttachPLLMarginPairs(pllMarginSampler, cfg, step, prepared, batchSize, seqLen, objective)
	if err != nil {
		return objectiveBatch{}, trainingStepError(step, "PLL margin batch", err)
	}
	prepared, err = attachDistillationTeacherProbs(distiller, prepared, batchSize, seqLen)
	if err != nil {
		return objectiveBatch{}, trainingStepError(step, "distillation batch", err)
	}
	prepared, err = attachData2VecTargets(data2vec, prepared, objective, batchSize, seqLen)
	if err != nil {
		return objectiveBatch{}, trainingStepError(step, "data2vec batch", err)
	}
	return prepared, nil
}

func trainingStepError(step int, part string, err error) error {
	return fmt.Errorf("prepare step %d %s: %w", step, part, err)
}
