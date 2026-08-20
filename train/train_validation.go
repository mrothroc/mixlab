package train

import (
	"fmt"
	"strings"
	"time"

	"github.com/mrothroc/mixlab/data"
)

func loadTrainingValidationSet(cfg *ArchConfig, trainPattern, name string, seed int64, batchTokens, seqLen int) (*data.ValSet, string, error) {
	valPattern := strings.Replace(trainPattern, "train", "val", 1)
	if cfg.Training.ConfiguredMetricValidation() {
		valSet, err := data.NewClassificationValSetWithExampleLimit(
			valPattern, cfg.Training.ValExamples, batchTokens, seqLen, effectiveLoaderOptions(cfg),
		)
		if err != nil {
			return nil, valPattern, fmt.Errorf("load configured classification validation set: %w", err)
		}
		return valSet, valPattern, nil
	}

	const defaultValBatchCount = 10
	valSet, err := data.NewValSetWithOptions(
		valPattern, seed, defaultValBatchCount, batchTokens, seqLen, effectiveLoaderOptions(cfg),
	)
	if err != nil {
		fmt.Printf("  [%s] no val set: %v\n", name, err)
	}
	return valSet, valPattern, nil
}

type trainingValidationRequest struct {
	cfg               *ArchConfig
	name              string
	valSet            *data.ValSet
	trainer           GPUTrainer
	causalEval        causalEvalSwitcher
	currentProgramKey trainingProgramCacheKey
	pairSampler       *minimalPairSampler
	invarianceSampler *invariancePairSampler
	pllMarginSampler  *pllMarginPairSampler
	metricScheduler   metricTrainingScheduler
	step              int
	totalSteps        int
	valEvery          int
	batchSize         int
	seqLen            int
}

type trainingValidationResult struct {
	duration time.Duration
	log      string
	ran      bool
	loss     float64
}

func runTrainingValidation(req trainingValidationRequest) (trainingValidationResult, error) {
	if req.valSet == nil || len(req.valSet.Batches) == 0 ||
		!shouldRunTrainingValidationStep(req.cfg.Training, req.step, req.totalSteps, req.valEvery) {
		return trainingValidationResult{}, nil
	}

	started := time.Now()
	stopSlowLog := startSlowTrainingPhaseLogger(req.name, req.step, "validation")
	var loss float64
	var classificationMetrics ClassificationMetrics
	classificationSummary := ""
	var validationErr error
	switch {
	case req.cfg.ClassificationEnabled():
		runShape := func(evalBatchSize, evalSeqLen int, fn func() error) error {
			evalKey := req.currentProgramKey
			evalKey.dropoutInactive = true
			evalKey.batchSize = evalBatchSize
			evalKey.seqLen = evalSeqLen
			return req.causalEval.withProgramKey(req.currentProgramKey, evalKey, fn)
		}
		classificationMetrics, validationErr = evaluateClassificationValidationWithTrainer(
			req.cfg, req.valSet, req.trainer, req.step, req.batchSize, req.seqLen, nil, runShape,
		)
		if validationErr == nil {
			loss = classificationMetrics.Loss
			classificationSummary = classificationMetrics.summary()
		}
	case req.cfg.Training.MultiheadEnabled():
		validationErr = req.causalEval.withCausalEvalProgram(req.currentProgramKey, func() error {
			var err error
			loss, err = meanMultiheadValidationLoss(
				req.cfg, req.valSet, req.trainer, req.pairSampler, req.invarianceSampler,
				req.pllMarginSampler, req.step, req.batchSize, req.seqLen,
			)
			return err
		})
	default:
		loss, validationErr = req.causalEval.meanValidationLossCausal(req.currentProgramKey, req.valSet)
	}
	stopSlowLog()
	result := trainingValidationResult{duration: time.Since(started)}
	if validationErr != nil {
		if req.cfg.Training.ConfiguredMetricValidation() {
			return result, fmt.Errorf("configured validation after %d completed steps: %w", req.step+1, validationErr)
		}
		return result, nil
	}

	result.ran = true
	result.loss = loss
	if classificationSummary != "" {
		result.log = " val_" + classificationSummary
	} else {
		result.log = fmt.Sprintf(" val=%.4f", loss)
	}
	if req.metricScheduler == nil {
		return result, nil
	}
	metric, err := newBobClassificationMetric(req.cfg.Training.NewBob, classificationMetrics)
	if err != nil {
		return result, err
	}
	observation, err := req.metricScheduler.Observe(metric)
	if err != nil {
		return result, fmt.Errorf("observe NewBob metric after %d completed steps: %w", req.step+1, err)
	}
	if observation.HavePrevious {
		fmt.Printf("  [%s] NewBob completed_steps=%d metric=%s value=%.6f improvement=%.6f patient=%d->%d lr=%.8g->%.8g annealed=%t\n",
			req.name, req.step+1, req.cfg.Training.NewBob.Metric, metric, observation.Improvement,
			observation.PatientBefore, observation.PatientAfter, observation.OldLR, observation.NewLR, observation.Annealed)
	} else {
		fmt.Printf("  [%s] NewBob completed_steps=%d metric=%s value=%.6f first_observation=true lr=%.8g\n",
			req.name, req.step+1, req.cfg.Training.NewBob.Metric, metric, observation.NewLR)
	}
	return result, nil
}
