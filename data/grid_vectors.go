package data

import (
	"encoding/hex"
	"fmt"
	"strings"
)

func validateGridVectorProvenance(m *DatasetManifest) error {
	var sources map[string]GridVectorSource
	if m.GridProvenance != nil {
		sources = m.GridProvenance.Vectors
	}
	if m.Grid.VectorDim == 0 {
		if len(sources) != 0 {
			return fmt.Errorf("grid vector provenance requires vector_dim")
		}
		return nil
	}
	if len(sources) != len(m.Splits) {
		return fmt.Errorf("grid vector provenance required for every split")
	}
	for split := range m.Splits {
		s := sources[split]
		hash, err := hex.DecodeString(s.SHA256)
		if strings.TrimSpace(s.Name) == "" || s.Width != m.Grid.VectorDim || err != nil || len(hash) != 32 {
			return fmt.Errorf("grid split %q invalid vector source name, width or sha256", split)
		}
	}
	return nil
}

// Padded rows are never scored; a real vector keeps later coordinate consumers
// valid even when zero would be outside their supported domain.
func padGridVectors(b *GridBatch) {
	s := b.Geometry.VectorDim
	if s == 0 || b.Count == 0 {
		return
	}
	for row := b.Count; row < b.BatchSize; row++ {
		copy(b.Vectors[row*s:(row+1)*s], b.Vectors[:s])
	}
}
