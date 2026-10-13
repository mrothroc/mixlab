package arch

import (
	"fmt"
	"math"
	"reflect"
)

func groupNormParams(params map[string]interface{}) (int, float32, error) {
	for name := range params {
		if name != "groups" && name != "eps" {
			return 0, 0, fmt.Errorf("unsupported group_norm parameter %q", name)
		}
	}
	groups, err := gridInteger(params["groups"])
	if err != nil || groups <= 0 {
		return 0, 0, fmt.Errorf("group_norm groups must be a positive integer")
	}
	eps := float32(1e-5)
	if v, ok := params["eps"]; ok {
		eps, err = valueToFloat(v)
	}
	if err != nil || eps <= 0 || math.IsNaN(float64(eps)) || math.IsInf(float64(eps), 0) {
		return 0, 0, fmt.Errorf("group_norm eps must be positive and finite in float32")
	}
	return groups, eps, nil
}

func groupNormShape(op OpSpec, inputs [][]int) ([]int, error) {
	groups, _, err := groupNormParams(op.Params)
	if err != nil {
		return nil, err
	}
	if len(inputs) != 3 || len(inputs[0]) != 4 {
		return nil, fmt.Errorf("group_norm requires NHWC input, scale [C], bias [C]")
	}
	c := inputs[0][3]
	if c <= 0 || c%groups != 0 {
		return nil, fmt.Errorf("group_norm groups must divide channels (%d)", c)
	}
	if !reflect.DeepEqual(inputs[1], []int{c}) || !reflect.DeepEqual(inputs[2], []int{c}) {
		return nil, fmt.Errorf("group_norm scale and bias must have shape [%d]", c)
	}
	return append([]int(nil), inputs[0]...), nil
}
