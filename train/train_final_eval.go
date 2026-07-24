package train

import (
	"fmt"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
)

// computeFinalTrainingLoss re-prepares the last training batch under the run's
// objective and evaluates it once more (dropout-inactive for classification) to
// report a stable end-of-run training loss.
func computeFinalTrainingLoss(
	cfg *ArchConfig, lastTrainBatch trainBatch, steps, seqLen, batchSize int,
	trainer GPUTrainer, causalEval causalEvalSwitcher, currentProgramKey trainingProgramCacheKey,
	pairSampler *minimalPairSampler, invarianceSampler *invariancePairSampler, pllMarginSampler *pllMarginPairSampler,
) (float32, error) {
	finalEvalBatch := objectiveBatch{x: lastTrainBatch.x, y: lastTrainBatch.y}
	switch {
	case cfg.ClassificationEnabled():
		var err error
		finalEvalBatch, err = prepareObjectiveBatchWithSeqLen(cfg, lastTrainBatch, steps, arch.ObjectiveClassification, seqLen)
		if err != nil {
			return 0, fmt.Errorf("prepare final classification training loss batch: %w", err)
		}
	case cfg.Training.MultiheadEnabled():
		var err error
		finalEvalBatch, err = prepareObjectiveBatchWithSeqLen(cfg, lastTrainBatch, steps, arch.ObjectiveMultihead, seqLen)
		if err != nil {
			return 0, fmt.Errorf("prepare final multihead training loss batch: %w", err)
		}
		finalEvalBatch, err = maybeAttachMinimalPairs(pairSampler, cfg, steps, finalEvalBatch, batchSize, seqLen)
		if err != nil {
			return 0, fmt.Errorf("prepare final minimal-pair training loss batch: %w", err)
		}
		finalEvalBatch, err = maybeAttachInvariancePairs(invarianceSampler, cfg, steps, finalEvalBatch, batchSize, seqLen, arch.ObjectiveMultihead)
		if err != nil {
			return 0, fmt.Errorf("prepare final invariance training loss batch: %w", err)
		}
		finalEvalBatch, err = maybeAttachPLLMarginPairs(pllMarginSampler, cfg, steps, finalEvalBatch, batchSize, seqLen, arch.ObjectiveMultihead)
		if err != nil {
			return 0, fmt.Errorf("prepare final PLL margin training loss batch: %w", err)
		}
	case cfg.Training.ExampleFramingEnabled() || cfg.Training.DatasetSequencePacking || cfg.Training.RecordFramingEnabled():
		var err error
		finalEvalBatch, err = prepareObjectiveBatchWithSeqLen(cfg, lastTrainBatch, steps, arch.ObjectiveCausal, seqLen)
		if err != nil {
			return 0, fmt.Errorf("prepare final framed training loss batch: %w", err)
		}
	}
	var evalLoss float32
	var err error
	switch {
	case cfg.ClassificationEnabled():
		evalKey := currentProgramKey
		evalKey.dropoutInactive = true
		err = causalEval.withProgramKey(currentProgramKey, evalKey, func() error {
			var evalErr error
			evalLoss, evalErr = trainer.EvaluateObjectiveGPU(finalEvalBatch, batchSize, seqLen)
			return evalErr
		})
	case cfg.Training.MultiheadEnabled():
		err = causalEval.withCausalEvalProgram(currentProgramKey, func() error {
			var evalErr error
			evalLoss, evalErr = evaluateObjectiveTrainingLossGPU(trainer, finalEvalBatch, batchSize, seqLen)
			return evalErr
		})
	case cfg.Training.ExampleFramingEnabled() || cfg.Training.DatasetSequencePacking || cfg.Training.RecordFramingEnabled():
		evalLoss, err = causalEval.evaluateCausalObjectiveTrainingLossGPU(currentProgramKey, finalEvalBatch)
	default:
		evalLoss, err = causalEval.evaluateCausalObjectiveGPU(currentProgramKey, finalEvalBatch)
	}
	if err != nil {
		return 0, fmt.Errorf("evaluate final training loss: %w", err)
	}
	return evalLoss, nil
}

// runFullEvaluation runs the optional end-of-run full BPB / native validation
// pass, printing results. Objectives without continuous-stream full-eval support
// print an explanatory notice instead.
func runFullEvaluation(
	cfg *ArchConfig, name, valPattern string, valSet *data.ValSet,
	trainer GPUTrainer, causalEval causalEvalSwitcher, currentProgramKey trainingProgramCacheKey,
	opts TrainOptions, steps, batchSize, seqLen int,
) {
	switch {
	case cfg.ClassificationEnabled():
		if valSet == nil {
			fmt.Printf("  [%s] native classification evaluation failed: validation set is unavailable\n", name)
			return
		}
		evalKey := currentProgramKey
		evalKey.dropoutInactive = true
		if err := causalEval.withProgramKey(currentProgramKey, evalKey, func() error {
			metrics, evalErr := evaluateClassificationValidation(cfg, valSet, trainer, steps, batchSize, seqLen)
			if evalErr == nil {
				fmt.Printf("  [%s] classification validation: %s examples=%d\n", name, metrics.summary(), metrics.Examples)
			}
			return evalErr
		}); err != nil {
			fmt.Printf("  [%s] native classification evaluation failed: %v\n", name, err)
		}
	case cfg.Training.MultiheadEnabled():
		fmt.Printf("  [%s] full validation BPB failed: training.objective=multihead is not supported by continuous-stream full eval in v1\n", name)
	case cfg.Training.ExampleFramingEnabled():
		fmt.Printf("  [%s] full validation BPB failed: training.example_framing is not supported by continuous-stream full eval in v1\n", name)
	case cfg.Training.DatasetSequencePacking:
		fmt.Printf("  [%s] full validation BPB failed: record-oriented sequence datasets require packed evaluation; use validation loss or native scoring\n", name)
	case cfg.Training.RecordFramingEnabled():
		fmt.Printf("  [%s] full validation BPB failed: one-record-per-row datasets require framed evaluation; use validation loss\n", name)
	default:
		if cfg.Training.TTTSteps > 0 {
			if cfg.Training.TTTMode == "lora" {
				fmt.Printf("  [%s] computing full validation BPB with LoRA-TTT (steps=%d lr=%g rank=%d)...\n", name, cfg.Training.TTTSteps, cfg.Training.TTTLR, cfg.Training.TTTRank)
			} else {
				fmt.Printf("  [%s] computing full validation BPB with score-first TTT (steps=%d lr=%g)...\n", name, cfg.Training.TTTSteps, cfg.Training.TTTLR)
			}
		} else {
			fmt.Printf("  [%s] computing full validation BPB...\n", name)
		}
		lutDir := opts.LUTDir
		if lutDir == "" {
			lutDir = "data"
		}
		if err := causalEval.withCausalEvalProgram(currentProgramKey, func() error {
			return runFullEval(cfg, valPattern, trainer, lutDir)
		}); err != nil {
			fmt.Printf("  [%s] full validation BPB failed: %v\n", name, err)
		}
	}
}
