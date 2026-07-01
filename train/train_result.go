package train

import (
	"fmt"
	"time"
)

// TrainResult holds the outcome of a training run.
type TrainResult struct {
	Name      string
	FirstLoss float64
	// LastLoss is the value of the IR "loss" output collected at the final
	// training step — the quantity the optimizer was actually minimising. This
	// is masked cross-entropy when training.first_byte_mask is enabled, the
	// MTP-weighted multi-token loss when mtp.n>1, otherwise the standard
	// next-token cross-entropy.
	LastLoss float64
	// LastUnmaskedLoss is the unmasked next-token cross-entropy of the trained
	// model on the final training batch, measured by an extra forward pass
	// after the last optimizer update. This is commensurate across runs
	// regardless of whether training.first_byte_mask or MTP is on, so it can
	// be compared directly between configurations. NaN when training did not
	// run (steps == 0).
	LastUnmaskedLoss float64
	LastValLoss      float64
	HasValLoss       bool
	Delta            float64
	Elapsed          time.Duration
	StepFLOPs        int64
	FLOPsPerTok      int64
}

// formatSummary returns a one-line summary of the training result.
func (r TrainResult) formatSummary() string {
	if r.HasValLoss {
		return fmt.Sprintf("%-12s first=%.4f last=%.4f val=%.4f delta=%.4f (%s)",
			r.Name, r.FirstLoss, r.LastLoss, r.LastValLoss, r.Delta, r.Elapsed.Round(time.Millisecond))
	}
	return fmt.Sprintf("%-12s first=%.4f last=%.4f delta=%.4f (%s)",
		r.Name, r.FirstLoss, r.LastLoss, r.Delta, r.Elapsed.Round(time.Millisecond))
}
