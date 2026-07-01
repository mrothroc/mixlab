package arch

import (
	"bytes"
	"encoding/json"
	"strings"
)

// RTDSpec configures ELECTRA-style replaced-token detection for multihead
// training. R1 supports generator="tied"; dedicated generator objects are
// parsed so they can fail with a clear staged-feature validation error.
type RTDSpec struct {
	Generator                  string                 `json:"generator,omitempty"`
	GeneratorHead              string                 `json:"generator_head,omitempty"`
	MaskProb                   float64                `json:"mask_prob,omitempty"`
	SampleTemperature          float64                `json:"sample_temperature,omitempty"`
	DiscriminatorLossWeight    float64                `json:"discriminator_loss_weight,omitempty"`
	DedicatedGenerator         *RTDDedicatedGenerator `json:"-"`
	maskProbSet                bool
	sampleTemperatureSet       bool
	discriminatorLossWeightSet bool
}

type RTDDedicatedGenerator struct {
	Type                string  `json:"type,omitempty"`
	ModelDim            int     `json:"model_dim,omitempty"`
	Layers              int     `json:"layers,omitempty"`
	Heads               int     `json:"heads,omitempty"`
	MLPMult             float64 `json:"mlp_mult,omitempty"`
	GeneratorLossWeight float64 `json:"generator_loss_weight,omitempty"`
}

func (r *RTDSpec) UnmarshalJSON(data []byte) error {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	type wire struct {
		Generator               json.RawMessage `json:"generator,omitempty"`
		GeneratorHead           string          `json:"generator_head,omitempty"`
		MaskProb                float64         `json:"mask_prob,omitempty"`
		SampleTemperature       float64         `json:"sample_temperature,omitempty"`
		DiscriminatorLossWeight float64         `json:"discriminator_loss_weight,omitempty"`
	}
	var w wire
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&w); err != nil {
		return err
	}
	r.GeneratorHead = w.GeneratorHead
	r.MaskProb = w.MaskProb
	r.SampleTemperature = w.SampleTemperature
	r.DiscriminatorLossWeight = w.DiscriminatorLossWeight
	if len(w.Generator) > 0 {
		var s string
		if err := json.Unmarshal(w.Generator, &s); err == nil {
			r.Generator = s
		} else {
			var d RTDDedicatedGenerator
			if err := json.Unmarshal(w.Generator, &d); err != nil {
				return err
			}
			if d.Type == "" {
				d.Type = "dedicated"
			}
			r.Generator = d.Type
			r.DedicatedGenerator = &d
		}
	}
	_, r.maskProbSet = raw["mask_prob"]
	_, r.sampleTemperatureSet = raw["sample_temperature"]
	_, r.discriminatorLossWeightSet = raw["discriminator_loss_weight"]
	return nil
}

func (r *RTDSpec) applyDefaults(defaultMaskProb float64) {
	if r == nil {
		return
	}
	r.Generator = strings.ToLower(strings.TrimSpace(r.Generator))
	if r.Generator == "" {
		r.Generator = "tied"
	}
	r.GeneratorHead = strings.TrimSpace(r.GeneratorHead)
	if !r.maskProbSet && r.MaskProb == 0 {
		r.MaskProb = defaultMaskProb
	}
	if !r.sampleTemperatureSet && r.SampleTemperature == 0 {
		r.SampleTemperature = 1
	}
	if !r.discriminatorLossWeightSet && r.DiscriminatorLossWeight == 0 {
		r.DiscriminatorLossWeight = 50
	}
}
