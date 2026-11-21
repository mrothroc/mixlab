package arch

import (
	"fmt"
	"math"
)

func spatialPadParams(params map[string]interface{}) ([]int, error) {
	if len(params) != 3 {
		return nil, fmt.Errorf("pad requires axis1, axis2 and mode only")
	}
	values := make([]int, 0, 5)
	for _, name := range []string{"axis1", "axis2"} {
		v, ok := params[name].([]interface{})
		if !ok || len(v) != 2 {
			return nil, fmt.Errorf("pad %s requires [before,after]", name)
		}
		for _, amount := range v {
			n, err := gridInteger(amount)
			if err != nil || n < 0 {
				return nil, fmt.Errorf("pad %s amounts must be nonnegative int32", name)
			}
			values = append(values, n)
		}
	}
	switch params["mode"] {
	case "zero":
		values = append(values, 0)
	case "edge":
		values = append(values, 1)
	case "circular":
		values = append(values, 2)
	default:
		return nil, fmt.Errorf("pad mode must be zero, edge or circular")
	}
	return values, nil
}

func spatialPadShape(op OpSpec, inputs [][]int) ([]int, error) {
	if len(inputs) != 1 || len(inputs[0]) != 4 {
		return nil, fmt.Errorf("pad requires one rank-4 NHWC input")
	}
	p, err := spatialPadParams(op.Params)
	if err != nil {
		return nil, err
	}
	out := append([]int(nil), inputs[0]...)
	for axis := 1; axis <= 2; axis++ {
		lo, hi := p[(axis-1)*2], p[(axis-1)*2+1]
		if p[4] == 2 && (lo > out[axis] || hi > out[axis]) {
			return nil, fmt.Errorf("circular padding cannot exceed its input axis size")
		}
		n := int64(out[axis]) + int64(lo) + int64(hi)
		if n > math.MaxInt32 {
			return nil, fmt.Errorf("padded axis exceeds int32")
		}
		out[axis] = int(n)
	}
	if _, err = gridSize(out); err != nil {
		return nil, err
	}
	return out, nil
}
