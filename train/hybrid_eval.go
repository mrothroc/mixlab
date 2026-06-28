package train

import (
	"fmt"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
)

// gpuProgramSwitcher is implemented by trainers that can swap their active
// compiled program in place — used for per-step objective/phase switching in
// the training loop and for causal-program evaluation under hybrid training.
type gpuProgramSwitcher interface {
	SetProgramGPU(*arch.Program) error
}

// causalEvalSwitcher runs evaluation against the causal program variant for
// hybrid-objective training. Hybrid training alternates the in-flight program
// between causal and masked objectives per batch, but validation, the final
// unmasked training-loss probe, and full eval must always score the generative
// (causal) task. For non-hybrid runs every method is a direct passthrough.
type causalEvalSwitcher struct {
	trainer       GPUTrainer
	hybrid        bool
	programForKey func(trainingProgramCacheKey) (*arch.Program, error)
	batchSize     int
	seqLen        int
}

// withCausalEvalProgram flushes the trainer, switches it to the cached causal
// program variant for currentKey, runs fn, then restores the in-flight program.
// When the run is not hybrid, or the current program is already causal, fn runs
// without any program switch.
func (c causalEvalSwitcher) withCausalEvalProgram(currentKey trainingProgramCacheKey, fn func() error) error {
	needsShapeSwitch := currentKey.seqLen > 0 && currentKey.seqLen != c.seqLen
	if !c.hybrid && !needsShapeSwitch {
		return fn()
	}
	switcher, ok := c.trainer.(gpuProgramSwitcher)
	if !ok {
		return fmt.Errorf("trainer does not support scheduled program switching")
	}
	if err := c.trainer.FlushGPU(); err != nil {
		return err
	}
	restoreKey := currentKey
	causalKey := currentKey
	if c.hybrid {
		causalKey.objective = arch.ObjectiveCausal
	}
	causalKey.seqLen = c.seqLen
	switched := causalKey != restoreKey
	if switched {
		causalProg, err := c.programForKey(causalKey)
		if err != nil {
			return fmt.Errorf("build causal eval IR program: %w", err)
		}
		if err := switcher.SetProgramGPU(causalProg); err != nil {
			return fmt.Errorf("switch causal eval IR program: %w", err)
		}
	}
	runErr := fn()
	if switched {
		restoreProg, err := c.programForKey(restoreKey)
		if err != nil {
			if runErr != nil {
				return fmt.Errorf("%v; build restore IR program: %w", runErr, err)
			}
			return fmt.Errorf("build restore IR program: %w", err)
		}
		if err := switcher.SetProgramGPU(restoreProg); err != nil {
			if runErr != nil {
				return fmt.Errorf("%v; restore IR program: %w", runErr, err)
			}
			return fmt.Errorf("restore IR program: %w", err)
		}
	}
	return runErr
}

func (c causalEvalSwitcher) evaluateCausalObjectiveGPU(currentKey trainingProgramCacheKey, batch objectiveBatch) (float32, error) {
	var loss float32
	err := c.withCausalEvalProgram(currentKey, func() error {
		var evalErr error
		loss, evalErr = c.trainer.EvaluateObjectiveGPU(batch, c.batchSize, c.seqLen)
		return evalErr
	})
	return loss, err
}

func (c causalEvalSwitcher) evaluateCausalObjectiveTrainingLossGPU(currentKey trainingProgramCacheKey, batch objectiveBatch) (float32, error) {
	var loss float32
	err := c.withCausalEvalProgram(currentKey, func() error {
		var evalErr error
		loss, evalErr = evaluateObjectiveTrainingLossGPU(c.trainer, batch, c.batchSize, c.seqLen)
		return evalErr
	})
	return loss, err
}

func (c causalEvalSwitcher) meanValidationLossCausal(currentKey trainingProgramCacheKey, valSet *data.ValSet) (float64, error) {
	var loss float64
	err := c.withCausalEvalProgram(currentKey, func() error {
		var evalErr error
		loss, evalErr = meanValidationLoss(valSet, c.trainer, c.batchSize, c.seqLen)
		return evalErr
	})
	return loss, err
}
