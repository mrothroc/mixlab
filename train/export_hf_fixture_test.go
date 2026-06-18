package train

import (
	"fmt"
	"math"
	"strings"
)

func scaleHFExportWeightsToTrainedMagnitude(weights [][]float32, shapes []WeightShape) error {
	if len(weights) != len(shapes) {
		return fmt.Errorf("weights=%d shapes=%d", len(weights), len(shapes))
	}
	scaled := 0
	for i, ws := range shapes {
		if ws.InitMode == "dwa_alpha" {
			// DWA alpha vectors init to [0,...,0,1] (identity residual), which would
			// make the aggregation a no-op and hide history-ordering bugs. Replace
			// with deterministic distinct positive weights summing to 1 so the
			// parity/oracle forward exercises real weighted aggregation.
			setDeterministicDWAAlpha(weights[i])
			scaled++
			continue
		}
		if !isTrainedMagnitudeCandidate(ws) {
			continue
		}
		before := weightRMS(weights[i])
		addDeterministicTrainedStructure(weights[i], i)
		afterStructure := weightRMS(weights[i])
		if afterStructure == 0 {
			return fmt.Errorf("weight %d %q has zero RMS after deterministic structure", i, ws.Name)
		}
		target := trainedMagnitudeTargetRMS(ws)
		if target < trainedFixtureMinRMS {
			target = trainedFixtureMinRMS
		}
		if minScaled := before * trainedFixtureMinScaleRatio; minScaled > target {
			target = minScaled
		}
		scale := target / afterStructure
		for j := range weights[i] {
			weights[i][j] *= float32(scale)
		}
		after := weightRMS(weights[i])
		if after < trainedFixtureMinRMS*0.99 {
			return fmt.Errorf("weight %d %q RMS=%g below trained-magnitude floor %g", i, ws.Name, after, trainedFixtureMinRMS)
		}
		if before > 0 && after < before*trainedFixtureMinScaleRatio*0.99 {
			return fmt.Errorf("weight %d %q RMS before=%g after=%g did not leave init scale", i, ws.Name, before, after)
		}
		scaled++
	}
	if scaled == 0 {
		return fmt.Errorf("no tensors eligible for trained-magnitude scaling")
	}
	return nil
}

// setDeterministicDWAAlpha overwrites a DWA alpha vector with distinct positive
// weights summing to 1 (alpha[i] = (i+1)/sum), so the aggregation is non-trivial
// and order-sensitive while keeping the residual scale stable.
func setDeterministicDWAAlpha(alpha []float32) {
	n := len(alpha)
	if n == 0 {
		return
	}
	total := float64(n*(n+1)) / 2
	for i := range alpha {
		alpha[i] = float32(float64(i+1) / total)
	}
}

func isTrainedMagnitudeCandidate(ws WeightShape) bool {
	if len(ws.Shape) < 2 || ws.InitZero || ws.InitOne || ws.IsNormScale || ws.InitDtBias || ws.InitLogArange {
		return false
	}
	return true
}

func trainedMagnitudeTargetRMS(ws WeightShape) float64 {
	name := strings.ToLower(ws.Name)
	switch {
	case strings.Contains(name, "embed") || strings.Contains(name, "table"):
		return 0.35
	case strings.Contains(name, "ff") || strings.Contains(name, "gate") || strings.Contains(name, "up") || strings.Contains(name, "down"):
		return 0.40
	case strings.Contains(name, "router"):
		return 0.25
	case strings.Contains(name, "pos") || strings.Contains(name, "relative"):
		return 0.30
	default:
		return 0.30
	}
}

func addDeterministicTrainedStructure(w []float32, weightIndex int) {
	for j := range w {
		s := math.Sin(float64((weightIndex+1)*(j+3))) + 0.5*math.Cos(float64((weightIndex+7)*(j+1)))
		w[j] += float32(0.05 * s)
	}
}

func weightRMS(w []float32) float64 {
	if len(w) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range w {
		sum += float64(v) * float64(v)
	}
	return math.Sqrt(sum / float64(len(w)))
}
