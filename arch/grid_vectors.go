package arch

import "fmt"

// Grid reshapes preserve record boundaries; unlike sequence custom blocks,
// symbolic dimensions other than the leading batch dimension are not accepted.
func gridReshape(op OpSpec, inputs [][]int, batch int) ([]int, error) {
	if len(inputs) != 1 || len(op.Params) != 1 {
		return nil, fmt.Errorf("grid reshape requires one input and shape")
	}
	values, ok := op.Params["shape"].([]interface{})
	if !ok || len(values) < 2 || len(values) > 4 {
		return nil, fmt.Errorf("grid reshape shape must have rank 2 through 4")
	}
	out := make([]int, len(values))
	for n, value := range values {
		if value == "B" && n == 0 {
			out[n] = batch
		} else {
			v, err := gridInteger(value)
			if err != nil {
				return nil, fmt.Errorf("grid reshape: %w", err)
			}
			out[n] = v
		}
	}
	before, _ := gridSize(inputs[0])
	after, err := gridSize(out)
	if err != nil || before != after || out[0] != batch || inputs[0][0] != batch {
		return nil, fmt.Errorf("grid reshape must preserve batch dimension and element count")
	}
	return out, nil
}
