package train

import (
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestLossNormalizerMatchesCausalDenominator(t *testing.T) {
	cfg := &ArchConfig{
		ModelDim:  8,
		VocabSize: 16,
		SeqLen:    4,
		Training: arch.TrainingSpec{
			Objective:   arch.ObjectiveCausal,
			BatchTokens: 8,
		},
	}
	batch := trainBatch{
		x: []int{1, 2, 3, 4, 5, 6, 7, 8},
		y: []int{2, 3, 4, 5, 6, 7, 8, 9},
	}
	prepared, err := prepareObjectiveBatch(cfg, batch, 0, arch.ObjectiveCausal)
	if err != nil {
		t.Fatalf("prepareObjectiveBatch dense: %v", err)
	}
	if !prepared.lossNormalizerSet || prepared.lossNormalizer != 8 {
		t.Fatalf(
			"dense loss normalizer set=%v value=%g, want true/8",
			prepared.lossNormalizerSet,
			prepared.lossNormalizer,
		)
	}

	batch.lossMask = []float32{1, 1, 0, 1, 0, 0, 1, 0}
	prepared, err = prepareObjectiveBatch(cfg, batch, 0, arch.ObjectiveCausal)
	if err != nil {
		t.Fatalf("prepareObjectiveBatch masked: %v", err)
	}
	if prepared.lossNormalizer != 4 {
		t.Fatalf("masked loss normalizer=%g want 4", prepared.lossNormalizer)
	}

	for i := range batch.lossMask {
		batch.lossMask[i] = 0
	}
	prepared, err = prepareObjectiveBatch(cfg, batch, 0, arch.ObjectiveCausal)
	if err != nil {
		t.Fatalf("prepareObjectiveBatch zero: %v", err)
	}
	if prepared.lossNormalizer != 0 || !prepared.lossNormalizerSet {
		t.Fatalf(
			"zero loss normalizer set=%v value=%g, want true/0",
			prepared.lossNormalizerSet,
			prepared.lossNormalizer,
		)
	}
}
