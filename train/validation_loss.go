package train

import (
	"fmt"

	"github.com/mrothroc/mixlab/data"
)

// meanValidationLoss computes the mean loss across validation batches.
func meanValidationLoss(valSet *data.ValSet, trainer GPUTrainer, batchSize, seqLen int) (float64, error) {
	return meanValidationLossWithTTT(valSet, trainer, batchSize, seqLen, "full", 0, 0, 0)
}

// meanValidationLossWithTTT computes score-first validation loss and, when
// tttSteps > 0, adapts weights after each scored batch.
func meanValidationLossWithTTT(
	valSet *data.ValSet,
	trainer GPUTrainer,
	batchSize, seqLen int,
	tttMode string,
	tttSteps int,
	tttLR float32,
	tttRank int,
) (float64, error) {
	if valSet == nil || len(valSet.Batches) == 0 {
		return 0, fmt.Errorf("no validation batches")
	}
	if tttSteps < 0 {
		return 0, fmt.Errorf("ttt_steps must be >= 0")
	}
	if tttMode == "" {
		tttMode = "full"
	}
	if tttMode != "full" && tttMode != "lora" {
		return 0, fmt.Errorf("ttt_mode must be \"full\" or \"lora\"")
	}
	if tttMode == "lora" && tttRank <= 0 {
		return 0, fmt.Errorf("ttt_rank must be > 0")
	}
	sum := 0.0
	count := 0
	failures := 0
	for _, vb := range valSet.Batches {
		var (
			loss float32
			err  error
		)
		if tttMode == "lora" && tttSteps > 0 {
			loss, err = trainer.EvaluateLoRATTTGPU(vb.X, vb.Y, batchSize, seqLen, tttSteps, tttLR, tttRank)
		} else {
			loss, err = trainer.EvaluateGPU(vb.X, vb.Y, batchSize, seqLen)
		}
		if err != nil {
			failures++
			continue
		}
		sum += float64(loss)
		count++
		if tttMode == "full" {
			for step := 0; step < tttSteps; step++ {
				if _, err := trainer.TrainStepGPU(vb.X, vb.Y, batchSize, seqLen, tttLR); err != nil {
					return 0, fmt.Errorf("ttt step %d after val batch %d: %w", step+1, count, err)
				}
			}
		}
	}
	if count == 0 {
		return 0, fmt.Errorf("validation evaluation failed for all %d batches", len(valSet.Batches))
	}
	if failures > 0 {
		fmt.Printf("  warning: %d/%d val batches failed, using %d successful\n", failures, len(valSet.Batches), count)
	}
	return sum / float64(count), nil
}
