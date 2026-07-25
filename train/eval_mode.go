// Package main implements the mixlab command-line modes.
package train

import (
	"fmt"
	"os"
	"runtime"
	"strings"

	"github.com/mrothroc/mixlab/data"
)

func runEvalMode(configPath, trainPattern, safetensorsLoad, lutDir string) error {
	return runEvalModeWithOptions(configPath, trainPattern, safetensorsLoad, EvalModeOptions{LUTDir: lutDir})
}

// EvalModeOptions controls standalone evaluation. ValBatches applies only to
// classification: zero traverses the full finite split and a positive value
// explicitly caps the number of batches.
type EvalModeOptions struct {
	LUTDir            string
	ValBatches        int
	ClassificationOut string
}

func runEvalModeWithOptions(configPath, trainPattern, safetensorsLoad string, opts EvalModeOptions) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if opts.ValBatches < 0 {
		return fmt.Errorf("-val-batches must be >= 0")
	}
	if configPath == "" {
		return fmt.Errorf("-config is required for eval mode; pass a JSON config file, e.g.: mixlab -mode eval -config examples/plain_3L.json -safetensors-load weights.st -train 'data/train_*.bin'")
	}
	if safetensorsLoad == "" {
		return fmt.Errorf("-safetensors-load is required for eval mode")
	}
	if trainPattern == "" {
		return fmt.Errorf("-train is required for eval mode; pass a glob pattern for data shards, e.g.: -train 'data/train_*.bin'")
	}

	cfg, err := LoadArchConfig(configPath)
	if err != nil {
		return err
	}
	if err := configureDatasetForTraining(cfg, trainPattern, cfg.Name); err != nil {
		return err
	}
	if !cfg.ClassificationEnabled() {
		if opts.ValBatches != 0 {
			return fmt.Errorf("-val-batches is supported only for training.objective=%q", "classification")
		}
		if opts.ClassificationOut != "" {
			return fmt.Errorf("-classification-out is supported only for training.objective=%q", "classification")
		}
	}
	if cfg.Training.ExampleFramingEnabled() {
		return fmt.Errorf("training.example_framing is not supported by standalone eval mode in v1")
	}
	if _, err := configureCharFeaturesForTraining(cfg, trainPattern); err != nil {
		return err
	}
	var prog *Program
	switch {
	case cfg.ClassificationEnabled():
		prog, err = BuildEvalIRProgramFromConfig(cfg)
	case cfg.Training.DatasetSequencePacking || cfg.Training.RecordFramingEnabled():
		prog, err = BuildTrainingIRProgramFromConfig(cfg, TrainingProgramState{
			RecurrenceActive: true, HeadUntied: cfg.MTPUntieEnabled(), Objective: "causal",
			MTPAuxInactive: true, DistillationInactive: true, Data2VecInactive: true,
			InvarianceInactive: true, PLLMarginInactive: true, ZLossInactive: true, DropoutInactive: true,
		})
	default:
		prog, err = BuildEvalIRProgramFromConfig(cfg)
	}
	if err != nil {
		return fmt.Errorf("build IR program: %w", err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		return fmt.Errorf("compute weight shapes: %w", err)
	}
	loadedWeights, err := loadSafetensorsWeights(safetensorsLoad, shapes)
	if err != nil {
		return fmt.Errorf("load safetensors %q: %w", safetensorsLoad, err)
	}
	valPattern := strings.Replace(trainPattern, "train", "val", 1)
	if cfg.EffectiveEvalSpec().LegalChunkSGDEnabled() {
		if opts.ClassificationOut != "" {
			return fmt.Errorf("-classification-out is not supported with legal chunk-SGD eval")
		}
		fmt.Printf("loaded config %q: model_dim=%d vocab_size=%d seq_len=%d blocks=%d\n",
			cfg.Name, cfg.ModelDim, cfg.VocabSize, cfg.SeqLen, len(cfg.Blocks))
		fmt.Printf("  [%s] loaded %d weights from %s\n", cfg.Name, len(loadedWeights), safetensorsLoad)
		trainer, err := initLegalChunkSGDTrainer(prog, cfg, loadedWeights)
		if err != nil {
			return err
		}
		defer trainer.CloseTrainer()
		return runFullEvalLegalChunkSGDWithTrainer(cfg, valPattern, trainer, opts.LUTDir)
	}
	trainer, err := initGPUTrainer(prog, cfg, loadedWeights, nil)
	if err != nil {
		return fmt.Errorf("init GPU trainer: %w", err)
	}
	defer trainer.CloseTrainer()

	batchTokens := cfg.Training.BatchTokens
	seqLen := cfg.SeqLen
	if batchTokens%seqLen != 0 {
		return fmt.Errorf("batch_tokens (%d) must be divisible by seq_len (%d)", batchTokens, seqLen)
	}
	batchSize := batchTokens / seqLen

	var valSet *data.ValSet
	if cfg.ClassificationEnabled() {
		valSet, err = data.NewClassificationValSet(valPattern, opts.ValBatches, batchTokens, seqLen)
	} else {
		const defaultValBatchCount = 10
		valSet, err = data.NewValSetWithOptions(valPattern, cfg.Training.Seed, defaultValBatchCount, batchTokens, seqLen, effectiveLoaderOptions(cfg))
	}
	if err != nil {
		return fmt.Errorf("load val set %q: %w", valPattern, err)
	}
	if cfg.ClassificationEnabled() {
		var predictionsFile *os.File
		if opts.ClassificationOut != "" {
			predictionsFile, err = os.Create(opts.ClassificationOut)
			if err != nil {
				return fmt.Errorf("create classification output %q: %w", opts.ClassificationOut, err)
			}
			defer func() {
				if predictionsFile != nil {
					_ = predictionsFile.Close()
				}
			}()
		}
		metrics, err := evaluateClassificationValidationWithPredictions(
			cfg, valSet, trainer, 0, batchSize, seqLen, predictionsFile,
		)
		if err != nil {
			return fmt.Errorf("evaluate classification validation: %w", err)
		}
		if predictionsFile != nil {
			if err := predictionsFile.Close(); err != nil {
				return fmt.Errorf("close classification output %q: %w", opts.ClassificationOut, err)
			}
			predictionsFile = nil
		}
		fmt.Printf("loaded config %q: model_dim=%d vocab_size=%d seq_len=%d blocks=%d\n",
			cfg.Name, cfg.ModelDim, cfg.VocabSize, cfg.SeqLen, len(cfg.Blocks))
		fmt.Printf("  [%s] loaded %d weights from %s\n", cfg.Name, len(loadedWeights), safetensorsLoad)
		if valSet.EvaluatedExamples < valSet.TotalExamples {
			fmt.Printf(
				"  [%s] warning: classification validation capped at %d of %d examples by -val-batches=%d\n",
				cfg.Name, valSet.EvaluatedExamples, valSet.TotalExamples, opts.ValBatches,
			)
		}
		fmt.Printf("  [%s] classification validation: %s examples=%d\n", cfg.Name, metrics.summary(), metrics.Examples)
		if opts.ClassificationOut != "" {
			fmt.Printf("  [%s] classification predictions: %s\n", cfg.Name, opts.ClassificationOut)
		}
		return nil
	}

	tttSteps := cfg.Training.TTTSteps
	tttMode := cfg.Training.TTTMode
	tttLR := float32(cfg.Training.TTTLR)
	tttRank := cfg.Training.TTTRank
	valLoss, err := meanValidationLossWithTTT(valSet, trainer, batchSize, seqLen, tttMode, tttSteps, tttLR, tttRank)
	if err != nil {
		return fmt.Errorf("evaluate validation loss: %w", err)
	}

	fmt.Printf("loaded config %q: model_dim=%d vocab_size=%d seq_len=%d blocks=%d\n",
		cfg.Name, cfg.ModelDim, cfg.VocabSize, cfg.SeqLen, len(cfg.Blocks))
	fmt.Printf("  [%s] loaded %d weights from %s\n", cfg.Name, len(loadedWeights), safetensorsLoad)
	if tttSteps > 0 {
		if tttMode == "lora" {
			fmt.Printf("  [%s] validation loss=%.6f (LoRA-TTT steps=%d lr=%g rank=%d)\n", cfg.Name, valLoss, tttSteps, tttLR, tttRank)
		} else {
			fmt.Printf("  [%s] validation loss=%.6f (score-first TTT steps=%d lr=%g)\n", cfg.Name, valLoss, tttSteps, tttLR)
		}
	} else {
		fmt.Printf("  [%s] validation loss=%.6f\n", cfg.Name, valLoss)
	}
	return nil
}
