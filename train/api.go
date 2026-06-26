package train

import (
	"fmt"
)

func RunArch(configPath, trainPattern string, opts TrainOptions) error {
	if configPath == "" {
		return fmt.Errorf("-config is required for arch mode; pass a JSON config file, e.g.: mixlab -mode arch -config examples/plain_3L.json -train 'data/train_*.bin'")
	}
	cfg, err := LoadArchConfig(configPath)
	if err != nil {
		return err
	}
	fmt.Printf("loaded config %q: model_dim=%d vocab_size=%d seq_len=%d blocks=%d\n",
		cfg.Name, cfg.ModelDim, cfg.VocabSize, cfg.SeqLen, len(cfg.Blocks))
	if len(cfg.Training.Phases) > 0 {
		fmt.Printf("  training: phases=%d steps=%d grad_clip=%g weight_decay=%g seed=%d batch_tokens=%d\n",
			len(cfg.Training.Phases), cfg.Training.TotalSteps(), cfg.Training.GradClip, cfg.Training.WeightDecay,
			cfg.Training.Seed, cfg.Training.BatchTokens)
	} else {
		fmt.Printf("  training: steps=%d lr=%g grad_clip=%g weight_decay=%g seed=%d batch_tokens=%d\n",
			cfg.Training.Steps, cfg.Training.LR, cfg.Training.GradClip, cfg.Training.WeightDecay,
			cfg.Training.Seed, cfg.Training.BatchTokens)
	}

	if trainPattern == "" {
		return fmt.Errorf("-train is required for arch mode; pass a glob pattern for data shards, e.g.: -train 'data/train_*.bin'")
	}

	result, err := runTrain(cfg, trainPattern, opts)
	if err != nil {
		return err
	}

	fmt.Println("\n=== ARCH RESULT ===")
	fmt.Println(result.FormatSummary())
	return nil
}

func RunArchRace(configsDir, trainPattern string, opts TrainOptions) error {
	return runArchRace(configsDir, trainPattern, opts)
}

func RunCount(configPath string) error {
	return runCount(configPath)
}

func RunEvalMode(configPath, trainPattern, safetensorsLoad string) error {
	return runEvalMode(configPath, trainPattern, safetensorsLoad, "data")
}

func RunEvalModeWithLUT(configPath, trainPattern, safetensorsLoad, lutDir string) error {
	return runEvalMode(configPath, trainPattern, safetensorsLoad, lutDir)
}

func RunEvalLogprobs(configPath, trainPattern, safetensorsLoad, lutDir, logprobsOut string) error {
	return runEvalLogprobs(configPath, trainPattern, safetensorsLoad, lutDir, logprobsOut, "", "")
}

// RunEvalLogprobsAndRanks runs eval mode and writes per-token NLLs to
// logprobsOut (skipped if empty) and per-token target ranks to ranksOut
// (skipped if empty). At least one must be non-empty. When ranksOut is
// supplied, both outputs are derived from a single GPU pass over the
// validation set, so the position-aligned records share token IDs.
func RunEvalLogprobsAndRanks(configPath, trainPattern, safetensorsLoad, lutDir, logprobsOut, ranksOut string) error {
	return runEvalLogprobs(configPath, trainPattern, safetensorsLoad, lutDir, logprobsOut, ranksOut, "")
}

// RunEvalLogprobsRanksAndUncertainty runs eval mode and writes any requested
// per-token NLL, target-rank, and candidate uncertainty files. At least one
// output path must be non-empty. Rank and uncertainty outputs are derived from
// one logits pass per batch and align position-by-position with logprobs.
func RunEvalLogprobsRanksAndUncertainty(configPath, trainPattern, safetensorsLoad, lutDir, logprobsOut, ranksOut, uncertaintyOut string) error {
	return runEvalLogprobs(configPath, trainPattern, safetensorsLoad, lutDir, logprobsOut, ranksOut, uncertaintyOut)
}

// RunEvalExports is the general per-token export entry point. It runs eval
// mode and writes any combination of -logprobs-out, -ranks-out,
// -uncertainty-out, and -logits-out from a single GPU pass. At least one
// output path on exports must be non-empty. Records align position-by-position
// across files.
func RunEvalExports(configPath, trainPattern, safetensorsLoad, lutDir string, exports EvalExportOptions) error {
	return runEvalExports(configPath, trainPattern, safetensorsLoad, lutDir, exports)
}

func RunGenerate(configPath, safetensorsLoad string, maxTokens int, temperature float32, topK int, prompt string) error {
	return runGenerate(configPath, safetensorsLoad, maxTokens, temperature, topK, prompt)
}

func RunGenerateDiffusion(configPath, safetensorsLoad string, maxTokens int, prompt string) error {
	return runGenerateDiffusion(configPath, safetensorsLoad, maxTokens, prompt)
}

func RunGenerateDiffusionWithOptions(opts GenerateDiffusionOptions) error {
	return runGenerateDiffusionWithOptions(opts)
}

func RunHiddenstats(configPath, trainPattern, safetensorsLoad, outputPath string) error {
	return runHiddenstats(configPath, trainPattern, safetensorsLoad, outputPath)
}

func RunPrepare(opts PrepareOptions) error {
	return runPrepare(opts)
}

func RunSmoke() error {
	return runSmoke()
}

func MLXAvailable() bool {
	return mlxAvailable()
}

func (r TrainResult) FormatSummary() string {
	return r.formatSummary()
}
