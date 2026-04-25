//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"strings"

	"github.com/mrothroc/mixlab/gpu"
)

func validateOptimizerSpecCoverage(spec gpu.TrainerOptimizerSpec, shapes []WeightShape) error {
	if len(shapes) == 0 {
		if len(spec.Weights) != 0 {
			return fmt.Errorf("optimizer spec defines %d weight assignments for 0 weights", len(spec.Weights))
		}
		return nil
	}

	if len(spec.Groups) == 0 {
		return fmt.Errorf("optimizer spec has no optimizer groups")
	}

	coverage := make([]int, len(shapes))
	for i := range coverage {
		coverage[i] = -1
	}

	for weightIdx, weightSpec := range spec.Weights {
		if weightIdx >= len(shapes) {
			return fmt.Errorf("optimizer spec defines extra weight assignment at index %d; expected %d weights", weightIdx, len(shapes))
		}
		if weightSpec.GroupIndex < 0 || weightSpec.GroupIndex >= len(spec.Groups) {
			return fmt.Errorf(
				"weight %q references optimizer group %d, but only %d groups exist",
				shapes[weightIdx].Name,
				weightSpec.GroupIndex,
				len(spec.Groups),
			)
		}
		if coverage[weightIdx] >= 0 {
			return fmt.Errorf(
				"weight %q is assigned more than once (groups %d and %d)",
				shapes[weightIdx].Name,
				coverage[weightIdx],
				weightSpec.GroupIndex,
			)
		}
		coverage[weightIdx] = weightSpec.GroupIndex
	}

	var missing []string
	for i, groupIdx := range coverage {
		if groupIdx < 0 {
			missing = append(missing, shapes[i].Name)
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("missing optimizer assignments for weights: %s", strings.Join(missing, ", "))
	}

	return nil
}
