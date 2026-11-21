package arch

import "fmt"

type spatialPair [2]int

func (p spatialPair) String() string {
	if p[0] == p[1] {
		return fmt.Sprint(p[0])
	}
	return fmt.Sprintf("[%d,%d]", p[0], p[1])
}

type conv2DParameters struct {
	Kernel, Stride, Padding, Dilation spatialPair
}

func parseSpatialPair(value any) (spatialPair, error) {
	var values []any
	switch v := value.(type) {
	case []any:
		values = v
	case []int:
		for _, x := range v {
			values = append(values, x)
		}
	default:
		n, err := gridInteger(value)
		return spatialPair{n, n}, err
	}
	if len(values) != 2 {
		return spatialPair{}, fmt.Errorf("expected an integer or a pair of two integers")
	}
	var result spatialPair
	for i, v := range values {
		n, err := gridInteger(v)
		if err != nil {
			return result, err
		}
		result[i] = n
	}
	return result, nil
}

func convolutionParameters(op OpSpec) (conv2DParameters, error) {
	p := conv2DParameters{Stride: spatialPair{1, 1}, Dilation: spatialPair{1, 1}}
	if op.Op != "conv2d" {
		k, s, pad, d, err := spatialOpParams(op)
		return conv2DParameters{spatialPair{k, k}, spatialPair{s, s}, spatialPair{pad, pad}, spatialPair{d, d}}, err
	}
	for name, value := range op.Params {
		if name == "output_padding" {
			n, err := gridInteger(value)
			if err != nil || n != 0 {
				return p, fmt.Errorf("output_padding must be 0")
			}
			continue
		}
		var dest *spatialPair
		switch name {
		case "kernel":
			dest = &p.Kernel
		case "stride":
			dest = &p.Stride
		case "padding":
			dest = &p.Padding
		case "dilation":
			dest = &p.Dilation
		default:
			return p, fmt.Errorf("unsupported spatial parameter %q", name)
		}
		v, err := parseSpatialPair(value)
		if err != nil {
			return p, fmt.Errorf("%s requires an integer or integer pair: %w", name, err)
		}
		*dest = v
	}
	for a := 0; a < 2; a++ {
		if p.Kernel[a] <= 0 || p.Stride[a] <= 0 || p.Dilation[a] <= 0 || p.Padding[a] < 0 {
			return p, fmt.Errorf("kernel/stride/dilation must be positive and padding nonnegative on each axis")
		}
	}
	return p, nil
}

func (p conv2DParameters) intParams() []int {
	if p.Kernel[0] == p.Kernel[1] && p.Stride[0] == p.Stride[1] && p.Padding[0] == p.Padding[1] && p.Dilation[0] == p.Dilation[1] {
		values := []int{p.Kernel[0], p.Stride[0], p.Padding[0]}
		if p.Dilation[0] != 1 {
			values = append(values, p.Dilation[0])
		}
		return values
	}
	return []int{p.Kernel[0], p.Kernel[1], p.Stride[0], p.Stride[1], p.Padding[0], p.Padding[1], p.Dilation[0], p.Dilation[1]}
}
