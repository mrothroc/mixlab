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

func TestLossNormalizerMatchesMaskedObjectiveDenominator(t *testing.T) {
	for _, objective := range []string{arch.ObjectiveMLM, arch.ObjectiveMNTP} {
		t.Run(objective, func(t *testing.T) {
			cfg := &ArchConfig{
				ModelDim:  8,
				VocabSize: 16,
				SeqLen:    4,
				Training: arch.TrainingSpec{
					Objective:        objective,
					BatchTokens:      8,
					MLMMaskProb:      1,
					MLMMaskTokenID:   1,
					MLMMaskTokenProb: 1,
				},
			}
			batch := trainBatch{
				x: []int{2, 3, 4, 5, 6, 7, 8, 9},
				y: []int{3, 4, 5, 6, 7, 8, 9, 10},
			}
			prepared, err := prepareObjectiveBatch(cfg, batch, 0, objective)
			if err != nil {
				t.Fatalf("prepareObjectiveBatch: %v", err)
			}
			var want float32
			for _, value := range prepared.lossMask {
				if value > 0 {
					want += value
				}
			}
			if !prepared.lossNormalizerSet || prepared.lossNormalizer != want {
				t.Fatalf(
					"loss normalizer set=%v value=%g, want true/%g",
					prepared.lossNormalizerSet,
					prepared.lossNormalizer,
					want,
				)
			}
			if want == 0 {
				t.Fatal("test fixture produced no masked positions")
			}
		})
	}
}
