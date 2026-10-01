package arch

import (
	"fmt"
	"path"
)

type GridAugmentationSpec struct {
	Dihedral bool `json:"dihedral"`
}

// MatchGridWeightPatterns uses path.Match over the complete logical name.
// Dots have no special meaning; every pattern must match a declared weight.
func MatchGridWeightPatterns(patterns, names []string) (map[string]bool, error) {
	out := make(map[string]bool)
	for _, pattern := range patterns {
		if pattern == "" {
			return nil, fmt.Errorf("empty weight pattern")
		}
		if _, err := path.Match(pattern, ""); err != nil {
			return nil, fmt.Errorf("invalid weight pattern %q: %w", pattern, err)
		}
		matched := false
		for _, name := range names {
			ok, _ := path.Match(pattern, name)
			if ok {
				out[name], matched = true, true
			}
		}
		if !matched {
			return nil, fmt.Errorf("weight pattern %q matches no logical weights", pattern)
		}
	}
	return out, nil
}

func resolveGridTrainableWeights(metas []WeightMeta, training TrainingSpec) error {
	names := make([]string, len(metas))
	for n := range metas {
		names[n] = metas[n].Name
	}
	freeze, err := MatchGridWeightPatterns(training.Freeze, names)
	if err != nil {
		return fmt.Errorf("training.freeze: %w", err)
	}
	if _, err := MatchGridWeightPatterns(training.InitAllowMissing, names); err != nil {
		return fmt.Errorf("training.init_allow_missing: %w", err)
	}
	active := 0
	for n := range metas {
		metas[n].Frozen = metas[n].Frozen || freeze[metas[n].Name]
		if !metas[n].Frozen {
			active++
		}
	}
	if len(training.Freeze) > 0 && active == 0 {
		return fmt.Errorf("training.freeze leaves no trainable weights in the selected output graph")
	}
	return nil
}
