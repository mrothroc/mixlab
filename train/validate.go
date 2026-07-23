package train

import (
	"fmt"
)

func runValidate(configPath string) error {
	if configPath == "" {
		return fmt.Errorf("-config is required for validate mode; pass a JSON config file, e.g.: mixlab -mode validate -config examples/plain_3L.json")
	}

	cfg, err := LoadArchConfig(configPath)
	if err != nil {
		return err
	}
	if _, err := BuildIRProgramFromConfig(cfg); err != nil {
		return fmt.Errorf("build IR program: %w", err)
	}

	fmt.Printf("valid config %q: objective=%s model_dim=%d vocab_size=%d seq_len=%d blocks=%d\n",
		configPath, cfg.Training.Objective, cfg.ModelDim, cfg.VocabSize, cfg.SeqLen, len(cfg.Blocks))
	return nil
}
