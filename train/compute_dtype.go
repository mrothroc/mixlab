package train

import (
	"fmt"
	"strings"

	"github.com/mrothroc/mixlab/gpu"
)

func gpuComputeDTypeForTraining(cfg *ArchConfig) (gpu.ComputeDType, error) {
	if err := validateMLXComputeDTypeConfig(cfg); err != nil {
		return gpu.ComputeDTypeFloat32, err
	}
	switch cfg.Training.EffectiveComputeDType() {
	case "float32":
		return gpu.ComputeDTypeFloat32, nil
	case "bf16":
		return gpu.ComputeDTypeBF16, nil
	default:
		return gpu.ComputeDTypeFloat32, fmt.Errorf("unsupported training.compute_dtype=%q", cfg.Training.ComputeDType)
	}
}

func validateMLXComputeDTypeConfig(cfg *ArchConfig) error {
	if cfg == nil {
		return nil
	}
	if cfg.Training.EffectiveComputeDType() != "bf16" {
		return nil
	}
	if qat := strings.ToLower(strings.TrimSpace(cfg.Training.QAT)); qat != "" && qat != "none" {
		return fmt.Errorf("training.compute_dtype=\"bf16\" is not supported with training.qat=%q", cfg.Training.QAT)
	}
	for i, block := range cfg.Blocks {
		if !bf16SupportedBlockType(block.Type) {
			return fmt.Errorf("training.compute_dtype=\"bf16\" is not supported with blocks[%d].type=%q in v1", i, block.Type)
		}
	}
	return nil
}

func bf16SupportedBlockType(blockType string) bool {
	switch strings.ToLower(strings.TrimSpace(blockType)) {
	case "plain", "swiglu", "geglu", "mlp", "moe":
		return true
	default:
		return false
	}
}
