package train

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	ir "github.com/mrothroc/mixlab/arch"
)

type WeightShape struct {
	Name          string
	Shape         []int
	IsNormScale   bool
	InitOne       bool
	InitValue     float32
	InitZero      bool
	InitMode      string
	InitLogArange bool
	InitDtBias    bool
	DtMin         float64
	DtMax         float64
	GPTBERTScale  float32
	GPT2Scale     float32
	ModelDim      int
}

func computeWeightShapes(cfg *ArchConfig) ([]WeightShape, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
	}

	metas, err := ir.CollectWeightShapesFromConfig(cfg)
	if err != nil {
		return nil, err
	}

	shapes := make([]WeightShape, len(metas))
	for i, m := range metas {
		shapes[i] = WeightShape{
			Name:          m.Name,
			Shape:         m.Shape,
			IsNormScale:   m.IsNormScale,
			InitOne:       m.InitOne,
			InitValue:     m.InitValue,
			InitZero:      m.InitZero,
			InitMode:      m.InitMode,
			InitLogArange: m.InitLogArange,
			InitDtBias:    m.InitDtBias,
			DtMin:         m.DtMin,
			DtMax:         m.DtMax,
			GPTBERTScale:  m.GPTBERTScale,
			GPT2Scale:     m.GPT2Scale,
			ModelDim:      cfg.ModelDim,
		}
	}
	return shapes, nil
}

func initWeightData(shapes []WeightShape, seed int64, weightInit string, weightInitStd float32) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	weightInit = strings.ToLower(strings.TrimSpace(weightInit))
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
			switch weightInit {
			case "normal":
				std := float64(weightInitStd)
				if std <= 0 {
					std = 0.02
				}
				for j := range data {
					data[j] = float32(rng.NormFloat64() * std)
				}
			case "gptbert":
				std := gptBERTInitStd(ws)
				scale := float64(1.0)
				if ws.GPTBERTScale > 0 {
					scale = float64(ws.GPTBERTScale)
				}
				for j := range data {
					data[j] = float32(truncatedNormal(rng, std) * scale)
				}
			case "gpt2":
				std := float64(weightInitStd)
				if std <= 0 {
					std = 0.02
				}
				scale := float64(1.0)
				if ws.GPT2Scale > 0 {
					scale = float64(ws.GPT2Scale)
				}
				for j := range data {
					data[j] = float32(rng.NormFloat64() * std * scale)
				}
			default:
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

func gptBERTInitStd(ws WeightShape) float64 {
	hidden := ws.ModelDim
	if hidden <= 0 {
		hidden = inferModelDim(ws.Shape)
	}
	if hidden <= 0 {
		hidden = 1
	}
	return math.Sqrt(2.0 / (5.0 * float64(hidden)))
}

func inferModelDim(shape []int) int {
	if len(shape) == 0 {
		return 0
	}
	if len(shape) == 1 {
		return shape[0]
	}
	a, b := shape[0], shape[1]
	if a <= 0 {
		return b
	}
	if b <= 0 {
		return a
	}
	if a < b {
		return a
	}
	return b
}

func truncatedNormal(rng *rand.Rand, std float64) float64 {
	if std <= 0 {
		return 0
	}
	limit := 2 * std
	for {
		v := rng.NormFloat64() * std
		if v >= -limit && v <= limit {
			return v
		}
	}
}

func applySpecialWeightInit(data []float32, ws WeightShape, rng *rand.Rand) bool {
	if ws.InitLogArange && len(ws.Shape) == 2 {
		n := ws.Shape[1]
		for j := range data {
			data[j] = float32(math.Log(float64(j%n + 1)))
		}
		return true
	}
	if ws.InitDtBias && len(ws.Shape) == 1 {
		dtMin := ws.DtMin
		if dtMin <= 0 {
			dtMin = 0.001
		}
		dtMax := ws.DtMax
		if dtMax <= dtMin {
			dtMax = 0.1
		}
		logMin := math.Log(dtMin)
		logMax := math.Log(dtMax)
		for i := range data {
			dt := math.Exp(logMin + rng.Float64()*(logMax-logMin))
			data[i] = float32(inverseSoftplus(dt))
		}
		return true
	}
	switch ws.InitMode {
	case "dwa_alpha":
		if len(data) > 0 {
			data[len(data)-1] = 1
		}
		return true
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
