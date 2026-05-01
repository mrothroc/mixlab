package arch

import (
	"bytes"
	"encoding/json"
)

// MTPSpec controls parameter-free multi-token prediction. The model still
// emits one shared LM head; training loss combines cross-entropy over the
// next-N shifted targets.
type MTPSpec struct {
	N                 int       `json:"n,omitempty"`
	LossWeights       []float32 `json:"loss_weights,omitempty"`
	UntieEmbedAtFrac  float64   `json:"untie_embed_at_frac,omitempty"`
	nSet              bool
	lossWeightsSet    bool
	untieEmbedFracSet bool
}

// UnmarshalJSON records whether optional scalar fields were present so
// validation can distinguish omitted defaults from explicitly invalid zeros.
func (m *MTPSpec) UnmarshalJSON(data []byte) error {
	var raw struct {
		N                *int      `json:"n"`
		LossWeights      []float32 `json:"loss_weights"`
		UntieEmbedAtFrac *float64  `json:"untie_embed_at_frac"`
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&raw); err != nil {
		return err
	}
	if raw.N != nil {
		m.N = *raw.N
		m.nSet = true
	}
	m.LossWeights = raw.LossWeights
	m.lossWeightsSet = raw.LossWeights != nil
	if raw.UntieEmbedAtFrac != nil {
		m.UntieEmbedAtFrac = *raw.UntieEmbedAtFrac
		m.untieEmbedFracSet = true
	}
	return nil
}

// EffectiveN returns the configured MTP horizon count. A nil or omitted spec
// preserves the current single-token objective.
func (m *MTPSpec) EffectiveN() int {
	if m == nil || m.N <= 0 {
		return 1
	}
	return m.N
}

// EffectiveLossWeights returns one non-negative coefficient per prediction
// horizon. Defaults follow the modded-nanogpt PR #178 geometric pattern.
func (m *MTPSpec) EffectiveLossWeights() []float32 {
	n := m.EffectiveN()
	if m != nil && len(m.LossWeights) == n {
		out := make([]float32, n)
		copy(out, m.LossWeights)
		return out
	}
	out := make([]float32, n)
	w := float32(1)
	for i := range out {
		out[i] = w
		w *= 0.5
	}
	return out
}

// EffectiveUntieEmbedAtFrac returns the fraction of training at which a tied
// embedding/head pair should split. Omitted means never untie.
func (m *MTPSpec) EffectiveUntieEmbedAtFrac() float64 {
	if m == nil {
		return 1.0
	}
	if !m.untieEmbedFracSet && m.UntieEmbedAtFrac == 0 {
		return 1.0
	}
	return m.UntieEmbedAtFrac
}

// MTPEnabled reports whether training should include auxiliary future-token
// objectives.
func (c *ArchConfig) MTPEnabled() bool {
	return c != nil && c.MTP != nil && c.MTP.EffectiveN() > 1
}

// MTPUntieEnabled reports whether training should split an initially tied
// embedding/head pair according to mtp.untie_embed_at_frac.
func (c *ArchConfig) MTPUntieEnabled() bool {
	if c == nil || c.MTP == nil || !c.TieEmbeddings {
		return false
	}
	return c.MTP.EffectiveUntieEmbedAtFrac() < 1.0
}

// EffectiveMTPUntieStep returns the first step that should use the untied LM
// head. A return value <= 0 means untied from step 0 when MTPUntieEnabled is
// true, and no schedule otherwise.
func (c *ArchConfig) EffectiveMTPUntieStep() int {
	if !c.MTPUntieEnabled() {
		return 0
	}
	total := c.Training.TotalSteps()
	if total <= 0 {
		return 0
	}
	return int(c.MTP.EffectiveUntieEmbedAtFrac() * float64(total))
}

// ReservesUntiedHeadWeight reports whether the fixed weight layout needs a
// separate LM head tensor even when the initial program uses tied embeddings.
func (c *ArchConfig) ReservesUntiedHeadWeight() bool {
	if c == nil {
		return false
	}
	return !c.TieEmbeddings || c.MTPUntieEnabled()
}
