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
	InitValue   float32
	InitZero    bool
	InitMode    string
}

func computeWeightShapes(cfg *ArchConfig) ([]WeightShape, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
	}

	metas, err := ir.CollectWeightShapesWithNgramsRecurrenceAndParallel(
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
		cfg.TrigramVocabSize,
		cfg.EffectiveTrigramDim(),
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
			InitValue:   m.InitValue,
			InitZero:    m.InitZero,
			InitMode:    m.InitMode,
		}
	}
	return shapes, nil
}

func initWeightData(shapes []WeightShape, seed int64, weightInit string, weightInitStd float32) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	weights := make([][]float32, len(shapes))
	for i, ws := range shapes {
		n := 1
		for _, d := range ws.Shape {
			n *= d
		}
		data := make([]float32, n)
		if applySpecialWeightInit(data, ws, rng) {
			weights[i] = data
			continue
		}
		switch {
		case ws.InitValue != 0:
			for j := range data {
				data[j] = ws.InitValue
			}
		case ws.InitZero:
			// Leave the buffer zeroed for explicitly transparent starts such as SparseAttnGate.
		case len(ws.Shape) == 1:
			if ws.IsNormScale || ws.InitOne {
				for j := range data {
					data[j] = 1.0
				}
			}
		case len(ws.Shape) >= 2:
			if weightInit == "normal" {
				std := float64(weightInitStd)
				if std <= 0 {
					std = 0.02
				}
				for j := range data {
					data[j] = float32(rng.NormFloat64() * std)
				}
			} else {
				fanIn := ws.Shape[0]
				fanOut := ws.Shape[1]
				limit := float64(math.Sqrt(6.0 / float64(fanIn+fanOut)))
				for j := range data {
					data[j] = float32(rng.Float64()*2*limit - limit)
				}
			}
		}
		weights[i] = data
	}
	return weights
}

func applySpecialWeightInit(data []float32, ws WeightShape, rng *rand.Rand) bool {
	switch ws.InitMode {
	case "torch_linear_uniform":
		if len(ws.Shape) < 2 || ws.Shape[0] <= 0 {
			return false
		}
		bound := 1.0 / math.Sqrt(float64(ws.Shape[0]))
		for i := range data {
			data[i] = float32(rng.Float64()*2*bound - bound)
		}
		return true
	case "torch_depthwise_conv1d_uniform":
		if len(ws.Shape) < 1 || ws.Shape[0] <= 0 {
			return false
		}
		bound := 1.0 / math.Sqrt(float64(ws.Shape[0]))
		for i := range data {
			data[i] = float32(rng.Float64()*2*bound - bound)
		}
		return true
	case "gated_deltanet_A_log", "A_log":
		for i := range data {
			v := rng.Float64() * 16.0
			if v == 0 {
				v = math.SmallestNonzeroFloat64
			}
			data[i] = float32(math.Log(v))
		}
		return true
	case "gated_deltanet_dt_bias", "dt_bias":
		const (
			dtMin       = 0.001
			dtMax       = 0.1
			dtInitFloor = 1e-4
		)
		logMin := math.Log(dtMin)
		logMax := math.Log(dtMax)
		for i := range data {
			dt := math.Exp(rng.Float64()*(logMax-logMin) + logMin)
			if dt < dtInitFloor {
				dt = dtInitFloor
			}
			data[i] = float32(inverseSoftplus(dt))
		}
		return true
	default:
		return false
	}
}

func inverseSoftplus(x float64) float64 {
	return x + math.Log(-math.Expm1(-x))
}

func shouldDecayWeight(shape []int) bool {
	return len(shape) >= 2
}
