package arch

import (
	"fmt"
	"math"
)

func validateWeightInitialization(cfg *ArchConfig, source string) error {
	switch cfg.Training.WeightInit {
	case "", "xavier_uniform", "normal", "gptbert", "gpt2", "pytorch_linear", "pytorch_linear_all":
	default:
		return fmt.Errorf("config %q has invalid training.weight_init=%q (must be xavier_uniform, normal, gptbert, gpt2, pytorch_linear, or pytorch_linear_all)", source, cfg.Training.WeightInit)
	}
	for _, field := range []struct {
		name       string
		std        *float64
		applicable bool
	}{
		{"position_embedding_init_std", cfg.Training.PositionEmbeddingInitStd, cfg.EffectivePositionalEmbedding() == PositionalEmbeddingLearnedAbsolute},
		{"cls_token_init_std", cfg.Training.CLSTokenInitStd, cfg.ClassificationEnabled() && cfg.EffectiveClassificationPooling() == ClassificationPoolingCLS},
	} {
		if field.std == nil {
			continue
		}
		if !field.applicable {
			return fmt.Errorf("config %q training.%s requires the corresponding learned position/CLS tensor", source, field.name)
		}
		std := *field.std
		if std < 0 || math.IsNaN(std) || math.IsInf(std, 0) || math.IsInf(float64(float32(std)), 0) {
			return fmt.Errorf("config %q training.%s must be finite, non-negative, and representable in float32", source, field.name)
		}
	}
	return nil
}

func validateTrainingLearningRates(t TrainingSpec, source string) error {
	for _, field := range []struct {
		name  string
		value float64
	}{
		{"lr", t.LR}, {"embed_lr", float64(t.EmbedLR)},
		{"matrix_lr", float64(t.MatrixLR)}, {"scalar_lr", float64(t.ScalarLR)},
		{"head_lr", float64(t.HeadLR)},
	} {
		if field.value < 0 || math.IsNaN(field.value) || math.IsInf(field.value, 0) || math.IsInf(float64(float32(field.value)), 0) {
			return fmt.Errorf("config %q training.%s must be finite, non-negative, and representable in float32", source, field.name)
		}
	}
	return nil
}
