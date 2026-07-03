package arch

import (
	"fmt"
	"math"
	"strings"
)

const (
	MinimalPairSourceJSONL  = "jsonl"
	MinimalPairSourceBinary = "bin"
	MinimalPairLossLogistic = "logistic"
	MinimalPairLossHinge    = "hinge"
	MinimalPairEnergyMean   = "mean"
	MinimalPairEnergySpan   = "differing_span"

	EnergyPairLossLogistic = 0
	EnergyPairLossHinge    = 1
)

// MinimalPairSpec configures explicit clean/corrupt pair data for energy
// ranking heads.
type MinimalPairSpec struct {
	Source            string  `json:"source,omitempty"`
	Path              string  `json:"path,omitempty"`
	Loss              string  `json:"loss,omitempty"`
	Margin            float64 `json:"margin,omitempty"`
	PairBatchFraction float64 `json:"pair_batch_fraction,omitempty"`
	EnergyAggregation string  `json:"energy_aggregation,omitempty"`
}

func (m *MinimalPairSpec) applyDefaults() {
	if m == nil {
		return
	}
	m.Source = strings.ToLower(strings.TrimSpace(m.Source))
	if m.Source == "" {
		m.Source = MinimalPairSourceJSONL
	}
	m.Loss = strings.ToLower(strings.TrimSpace(m.Loss))
	if m.Loss == "" {
		m.Loss = MinimalPairLossLogistic
	}
	if m.Margin == 0 {
		m.Margin = 1
	}
	if m.PairBatchFraction == 0 {
		m.PairBatchFraction = 1
	}
	m.EnergyAggregation = strings.ToLower(strings.TrimSpace(m.EnergyAggregation))
	if m.EnergyAggregation == "" {
		m.EnergyAggregation = MinimalPairEnergyMean
	}
}

func (m MinimalPairSpec) LossKind() int {
	switch strings.ToLower(strings.TrimSpace(m.Loss)) {
	case MinimalPairLossHinge:
		return EnergyPairLossHinge
	default:
		return EnergyPairLossLogistic
	}
}

func (m MinimalPairSpec) EnergyAggregationMode() string {
	switch strings.ToLower(strings.TrimSpace(m.EnergyAggregation)) {
	case MinimalPairEnergySpan:
		return MinimalPairEnergySpan
	default:
		return MinimalPairEnergyMean
	}
}

func (m MinimalPairSpec) UsesDifferingSpanEnergy() bool {
	return m.EnergyAggregationMode() == MinimalPairEnergySpan
}

func validateMinimalPairSpec(cfg *ArchConfig, source string) error {
	if cfg == nil || cfg.Training.MinimalPair == nil {
		return nil
	}
	m := cfg.Training.MinimalPair
	switch m.Source {
	case MinimalPairSourceJSONL, MinimalPairSourceBinary:
	default:
		return fmt.Errorf("config %q training.minimal_pair.source=%q must be \"jsonl\" or \"bin\"", source, m.Source)
	}
	if strings.TrimSpace(m.Path) == "" {
		return fmt.Errorf("config %q training.minimal_pair.path is required", source)
	}
	switch m.Loss {
	case MinimalPairLossLogistic, MinimalPairLossHinge:
	default:
		return fmt.Errorf("config %q training.minimal_pair.loss=%q must be \"logistic\" or \"hinge\"", source, m.Loss)
	}
	if m.Margin <= 0 || math.IsNaN(m.Margin) || math.IsInf(m.Margin, 0) {
		return fmt.Errorf("config %q training.minimal_pair.margin=%g must be finite and > 0", source, m.Margin)
	}
	if m.PairBatchFraction <= 0 || m.PairBatchFraction > 1 || math.IsNaN(m.PairBatchFraction) || math.IsInf(m.PairBatchFraction, 0) {
		return fmt.Errorf("config %q training.minimal_pair.pair_batch_fraction=%g must be in (0,1]", source, m.PairBatchFraction)
	}
	switch m.EnergyAggregation {
	case MinimalPairEnergyMean, MinimalPairEnergySpan:
	default:
		return fmt.Errorf("config %q training.minimal_pair.energy_aggregation=%q must be \"mean\" or \"differing_span\"", source, m.EnergyAggregation)
	}
	return nil
}
