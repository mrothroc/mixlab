package train

import (
	"fmt"
	"strings"

	"github.com/mrothroc/mixlab/diloco/optimizer"
)

// LocalLearnerWireTensors gives each positional model tensor a unique container
// name, matching native weights-only checkpoints. Payload slices remain owned
// by the caller; numerical snapshots retain their original, possibly repeated,
// metadata names.
func LocalLearnerWireTensors(model []LocalLearnerTensor) []optimizer.Tensor {
	out := make([]optimizer.Tensor, len(model))
	for i, t := range model {
		out[i] = optimizer.Tensor{Name: fmt.Sprintf("w%d_%s", i, t.Name), Shape: t.Shape, Values: t.Data}
	}
	return out
}

// LocalLearnerModelFromWire validates the ordered container and removes exactly
// one canonical positional prefix. The numerical engine validates raw names
// and shapes against its model layout. Payload slices remain caller-owned.
func LocalLearnerModelFromWire(values []optimizer.Tensor) ([]LocalLearnerTensor, error) {
	if err := optimizer.Validate(values); err != nil {
		return nil, err
	}
	out := make([]LocalLearnerTensor, len(values))
	for i, t := range values {
		name, ok := strings.CutPrefix(t.Name, fmt.Sprintf("w%d_", i))
		if !ok || name == "" {
			return nil, fmt.Errorf("model tensor %d requires a canonical indexed name", i)
		}
		out[i] = LocalLearnerTensor{Name: name, Shape: t.Shape, Data: t.Values}
	}
	return out, nil
}
