package arch

import (
	"bytes"
	"encoding/json"
)

// MultiheadHeadSpec configures one head of a shared-trunk, multi-objective
// training run. UnmarshalJSON records field presence so defaults distinguish an
// explicit value from an omitted one.
// MultiheadSpec configures shared-trunk, multi-objective training heads.
type MultiheadHeadSpec struct {
	Name             string         `json:"name"`
	Objective        string         `json:"objective"`
	LossWeight       float64        `json:"loss_weight,omitempty"`
	LayerAggregation string         `json:"layer_aggregation,omitempty"`
	OutputHead       string         `json:"output_head,omitempty"`
	TieEmbeddings    bool           `json:"tie_embeddings,omitempty"`
	FinalNorm        bool           `json:"final_norm,omitempty"`
	Diffusion        *DiffusionSpec `json:"diffusion,omitempty"`

	lossWeightSet    bool
	tieEmbeddingsSet bool
	finalNormSet     bool
}

func (h *MultiheadHeadSpec) UnmarshalJSON(data []byte) error {
	type Alias MultiheadHeadSpec
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
	*h = MultiheadHeadSpec(alias)
	_, h.lossWeightSet = raw["loss_weight"]
	_, h.tieEmbeddingsSet = raw["tie_embeddings"]
	_, h.finalNormSet = raw["final_norm"]
	return nil
}
