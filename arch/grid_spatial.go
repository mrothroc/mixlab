package arch

import (
	"fmt"
	"math"
)

// spatialOpParams keeps dilation scoped to conv2d; pooling and transposed
// convolution retain their original parameter contracts.
func spatialOpParams(op OpSpec) (kernel, stride, padding, dilation int, err error) {
	dilation = 1
	params := op.Params
	if value, present := params["dilation"]; present {
		if op.Op != "conv2d" {
			return 0, 0, 0, 0, fmt.Errorf("dilation is supported only for conv2d")
		}
		dilation, err = gridInteger(value)
		if err != nil || dilation < 1 || dilation > math.MaxInt32 {
			return 0, 0, 0, 0, fmt.Errorf("dilation must be a positive integer <= %d", math.MaxInt32)
		}
		params = make(map[string]interface{}, len(op.Params)-1)
		for key, v := range op.Params {
			if key != "dilation" {
				params[key] = v
			}
		}
	}
	kernel, stride, padding, err = spatialParams(params)
	return
}

func gridConvOutputSize(input, kernel, stride, padding, dilation int) (int, error) {
	// Parameters are bounded to int32; widen before multiplying or adding.
	effective := int64(dilation)*int64(kernel-1) + 1
	padded := int64(input) + 2*int64(padding)
	if effective > math.MaxInt32 || padded > math.MaxInt32 {
		return 0, fmt.Errorf("convolution effective kernel or padded input exceeds int32 dimensions")
	}
	if padded < effective {
		return 0, fmt.Errorf("effective kernel exceeds padded input")
	}
	return int((padded-effective)/int64(stride) + 1), nil
}
