package arch

import (
	"bytes"
	"encoding/json"
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
	MinimalPairScoreEnergy  = "energy_scalar"
	MinimalPairScoreMLMPLL  = "mlm_span_pll"

	EnergyPairLossLogistic = 0
	EnergyPairLossHinge    = 1
)

// MinimalPairSpec configures explicit clean/corrupt pair data for native energy
// ranking heads or scorer-head span-PLL ranking regularization.
type MinimalPairSpec struct {
	Source            string  `json:"source,omitempty"`
	Path              string  `json:"path,omitempty"`
	Loss              string  `json:"loss,omitempty"`
	Margin            float64 `json:"margin,omitempty"`
	PairBatchFraction float64 `json:"pair_batch_fraction,omitempty"`
	EnergyAggregation string  `json:"energy_aggregation,omitempty"`
	ScoreSource       string  `json:"score_source,omitempty"`
	ScoreHead         string  `json:"score_head,omitempty"`
	LossWeight        float64 `json:"loss_weight,omitempty"`

	lossWeightSet bool
}

func (m *MinimalPairSpec) UnmarshalJSON(data []byte) error {
	type Alias MinimalPairSpec
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	var alias Alias
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&alias); err != nil {
		return err
	}
	*m = MinimalPairSpec(alias)
	_, m.lossWeightSet = raw["loss_weight"]
	return nil
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
	m.ScoreSource = strings.ToLower(strings.TrimSpace(m.ScoreSource))
	if m.ScoreSource == "" {
		m.ScoreSource = MinimalPairScoreEnergy
	}
	m.ScoreHead = strings.TrimSpace(m.ScoreHead)
	if !m.lossWeightSet && m.LossWeight == 0 {
		m.LossWeight = 1
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

func (m MinimalPairSpec) ScoreSourceMode() string {
	switch strings.ToLower(strings.TrimSpace(m.ScoreSource)) {
	case MinimalPairScoreMLMPLL:
		return MinimalPairScoreMLMPLL
	default:
		return MinimalPairScoreEnergy
	}
}

func (m MinimalPairSpec) UsesMLMSpanPLL() bool {
	return m.ScoreSourceMode() == MinimalPairScoreMLMPLL
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
	switch m.ScoreSource {
	case MinimalPairScoreEnergy, MinimalPairScoreMLMPLL:
	default:
		return fmt.Errorf("config %q training.minimal_pair.score_source=%q must be \"energy_scalar\" or \"mlm_span_pll\"", source, m.ScoreSource)
	}
	if m.LossWeight < 0 || math.IsNaN(m.LossWeight) || math.IsInf(m.LossWeight, 0) {
		return fmt.Errorf("config %q training.minimal_pair.loss_weight=%g must be finite and >= 0", source, m.LossWeight)
	}
	return nil
}
