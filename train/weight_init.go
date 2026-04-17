package train

import (
	"fmt"
	"math"
	"math/rand"

	ir "github.com/mrothroc/mixlab/arch"
)

type WeightShape struct {
	Name        string
	Shape       []int
	IsNormScale bool
	InitOne     bool
}

func computeWeightShapes(cfg *ArchConfig) ([]WeightShape, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
	}

	metas, err := ir.CollectWeightShapesWithBigramRecurrenceAndParallel(
		cfg.ModelDim,
		cfg.VocabSize,
		cfg.SeqLen,
		cfg.EffectiveMLPMult(),
		cfg.TieEmbeddings,
		cfg.BlockScales,
		cfg.ResidMix,
		cfg.UNet,
		cfg.ParallelResidual,
		cfg.BigramVocabSize,
		cfg.EffectiveBigramDim(),
		cfg.Blocks,
		cfg.Recurrence,
	)
	if err != nil {
		return nil, err
	}

	shapes := make([]WeightShape, len(metas))
	for i, m := range metas {
		shapes[i] = WeightShape{
			Name:        m.Name,
			Shape:       m.Shape,
			IsNormScale: m.IsNormScale,
			InitOne:     m.InitOne,
		}
	}
	return shapes, nil
}

func initWeightData(shapes []WeightShape, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	weights := make([][]float32, len(shapes))
	for i, ws := range shapes {
		n := 1
		for _, d := range ws.Shape {
			n *= d
		}
		data := make([]float32, n)
		if len(ws.Shape) == 1 {
			if ws.IsNormScale || ws.InitOne {
				for j := range data {
					data[j] = 1.0
				}
			}
		} else if len(ws.Shape) >= 2 {
			fanIn := ws.Shape[0]
			fanOut := ws.Shape[1]
			limit := float64(math.Sqrt(6.0 / float64(fanIn+fanOut)))
			for j := range data {
				data[j] = float32(rng.Float64()*2*limit - limit)
			}
		}
		weights[i] = data
	}
	return weights
}

func shouldDecayWeight(shape []int) bool {
	return len(shape) >= 2
}
