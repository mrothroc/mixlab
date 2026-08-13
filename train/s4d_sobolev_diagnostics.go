package train

import (
	"fmt"
	"math"
	"sort"
	"strings"

	arch "github.com/mrothroc/mixlab/arch"
)

type telemetryS4DSobolev struct {
	Blocks []telemetryS4DSobolevBlock `json:"blocks"`
}

type telemetryS4DSobolevBlock struct {
	BlockIndexes   []int   `json:"block_indexes"`
	WeightIndex    int     `json:"weight_index"`
	Granularity    string  `json:"granularity"`
	Trainable      bool    `json:"trainable"`
	WeightDecay    float64 `json:"weight_decay"`
	Bounded        bool    `json:"bounded"`
	BoundLower     float64 `json:"bound_lower,omitempty"`
	BoundUpper     float64 `json:"bound_upper,omitempty"`
	Count          int     `json:"count"`
	Minimum        float64 `json:"min"`
	P01            float64 `json:"p01"`
	P05            float64 `json:"p05"`
	Median         float64 `json:"p50"`
	P95            float64 `json:"p95"`
	P99            float64 `json:"p99"`
	Maximum        float64 `json:"max"`
	Mean           float64 `json:"mean"`
	CountAbsGT1    int     `json:"count_abs_gt_1"`
	CountAbsGT2    int     `json:"count_abs_gt_2"`
	NearLowerBound int     `json:"near_lower_bound"`
	NearUpperBound int     `json:"near_upper_bound"`
	NyquistMinimum float64 `json:"nyquist_multiplier_min"`
	NyquistMedian  float64 `json:"nyquist_multiplier_p50"`
	NyquistMaximum float64 `json:"nyquist_multiplier_max"`
}

func sampleS4DSobolevDiagnostics(trainer any, bindings []arch.S4DSobolevWeightBinding) (*telemetryS4DSobolev, error) {
	if len(bindings) == 0 {
		return nil, nil
	}
	indexes := make([]int, len(bindings))
	for i, binding := range bindings {
		indexes[i] = binding.WeightIndex
	}
	rawWeights, err := readSelectedTrainerWeights(trainer, indexes)
	if err != nil {
		return nil, err
	}
	result := &telemetryS4DSobolev{Blocks: make([]telemetryS4DSobolevBlock, len(bindings))}
	for i, binding := range bindings {
		values := make([]float64, len(rawWeights[i]))
		for j, raw := range rawWeights[i] {
			values[j] = arch.EffectiveS4DSobolevBeta(float64(raw), binding.Spec)
		}
		result.Blocks[i] = summarizeS4DSobolevBeta(binding, values)
	}
	return result, nil
}

func summarizeS4DSobolevBeta(binding arch.S4DSobolevWeightBinding, values []float64) telemetryS4DSobolevBlock {
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	summary := telemetryS4DSobolevBlock{
		BlockIndexes: append([]int(nil), binding.BlockIndexes...), WeightIndex: binding.WeightIndex,
		Granularity: arch.EffectiveS4DSobolevGranularity(binding.Spec),
		Trainable:   arch.EffectiveS4DSobolevTrainable(binding.Spec),
		WeightDecay: arch.EffectiveS4DSobolevWeightDecay(binding.Spec), Count: len(sorted),
	}
	if len(sorted) == 0 {
		return summary
	}
	summary.Minimum, summary.Maximum = sorted[0], sorted[len(sorted)-1]
	summary.P01, summary.P05 = linearQuantile(sorted, 0.01), linearQuantile(sorted, 0.05)
	summary.Median = linearQuantile(sorted, 0.5)
	summary.P95, summary.P99 = linearQuantile(sorted, 0.95), linearQuantile(sorted, 0.99)
	lo, hi, bounded := arch.S4DSobolevBounds(binding.Spec)
	summary.Bounded, summary.BoundLower, summary.BoundUpper = bounded, lo, hi
	nearBound := (hi - lo) * 0.01
	nyquist := make([]float64, len(sorted))
	var sum float64
	for i, value := range sorted {
		sum += value
		if math.Abs(value) > 1 {
			summary.CountAbsGT1++
		}
		if math.Abs(value) > 2 {
			summary.CountAbsGT2++
		}
		if bounded && value <= lo+nearBound {
			summary.NearLowerBound++
		}
		if bounded && value >= hi-nearBound {
			summary.NearUpperBound++
		}
		nyquist[i] = math.Pow(1.5, value)
	}
	summary.Mean = sum / float64(len(sorted))
	sort.Float64s(nyquist)
	summary.NyquistMinimum = nyquist[0]
	summary.NyquistMedian = linearQuantile(nyquist, 0.5)
	summary.NyquistMaximum = nyquist[len(nyquist)-1]
	return summary
}

func linearQuantile(sorted []float64, q float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	position := q * float64(len(sorted)-1)
	lower, upper := int(math.Floor(position)), int(math.Ceil(position))
	if lower == upper {
		return sorted[lower]
	}
	fraction := position - float64(lower)
	return sorted[lower]*(1-fraction) + sorted[upper]*fraction
}

func formatS4DSobolevDiagnostics(diagnostics *telemetryS4DSobolev) string {
	if diagnostics == nil {
		return ""
	}
	parts := make([]string, 0, len(diagnostics.Blocks))
	for _, block := range diagnostics.Blocks {
		parts = append(parts, fmt.Sprintf(
			"blocks=%v beta[p01=%.3g p50=%.3g p99=%.3g min=%.3g max=%.3g] |beta|>1=%d |beta|>2=%d near_bounds=%d/%d nyquist=[%.3g %.3g %.3g]",
			block.BlockIndexes, block.P01, block.Median, block.P99, block.Minimum, block.Maximum,
			block.CountAbsGT1, block.CountAbsGT2, block.NearLowerBound, block.NearUpperBound,
			block.NyquistMinimum, block.NyquistMedian, block.NyquistMaximum))
	}
	return strings.Join(parts, "; ")
}
