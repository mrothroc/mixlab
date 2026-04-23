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
	return runEvalMode(configPath, trainPattern, safetensorsLoad)
}

func RunGenerate(configPath, safetensorsLoad string, maxTokens int, temperature float32, topK int, prompt string) error {
	return runGenerate(configPath, safetensorsLoad, maxTokens, temperature, topK, prompt)
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
