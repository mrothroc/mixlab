package arch

import (
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"
)

// BuildGridIRProgram builds either a prediction-only or supervised grid graph.
func BuildGridIRProgram(cfg *ArchConfig, supervised bool) (*Program, error) {
	p, _, err := buildGridGraph(cfg, supervised)
	return p, err
}

func gridWeightShape(syms []string, cfg *ArchConfig) ([]int, error) {
	shape := make([]int, len(syms))
	for j, s := range syms {
		switch s {
		case "B":
			shape[j] = cfg.Training.BatchSize
		case "C":
			shape[j] = cfg.InputAdapter.Channels
		case "IH":
			shape[j] = cfg.InputAdapter.Height
		case "IW":
			shape[j] = cfg.InputAdapter.Width
		default:
			n, err := strconv.Atoi(s)
			if err != nil {
				return nil, fmt.Errorf("unknown grid shape symbol %q", s)
			}
			shape[j] = n
		}
	}
	if len(shape) == 0 {
		return nil, fmt.Errorf("empty grid shape")
	}
	_, err := gridSize(shape)
	return shape, err
}

func gridWeightInit(w WeightSpec, meta *WeightMeta, fanIn map[string]int) error {
	if w.Init == nil {
		return nil
	}
	i := w.Init
	if i.Weight != "" && i.Kind != "pytorch_conv_uniform" {
		return fmt.Errorf("weight %s: init.weight is only valid for paired convolution initialization", w.Name)
	}
	if i.Scale != 0 && i.Kind != "normal" && i.Kind != "uniform" {
		return fmt.Errorf("weight %s: init.scale is only valid for normal/uniform", w.Name)
	}
	if math.IsNaN(i.Scale) || math.IsInf(i.Scale, 0) || i.Scale < 0 {
		return fmt.Errorf("weight %s init.scale must be finite and nonnegative", w.Name)
	}
	switch i.Kind {
	case "zero":
		meta.InitZero = true
	case "one":
		meta.InitOne = true
		meta.InitMode = "custom_one"
	case "normal", "uniform":
		if i.Scale <= 0 {
			return fmt.Errorf("weight %s requires positive init.scale", w.Name)
		}
		meta.InitMode = "custom_" + i.Kind
		meta.InitScale = i.Scale
	case "pytorch_conv_uniform":
		name := w.Name
		if i.Weight != "" {
			name = i.Weight
		}
		fan := fanIn[name]
		if fan <= 0 {
			return fmt.Errorf("weight %s init must reference a convolution weight", w.Name)
		}
		if name != w.Name && len(meta.Shape) != 1 {
			return fmt.Errorf("weight %s paired initialization requires a vector bias", w.Name)
		}
		meta.InitMode = "torch_linear_bias_uniform"
		meta.PyTorchLinearFanIn = fan
	default:
		return fmt.Errorf("weight %s unknown init.kind=%q", w.Name, i.Kind)
	}
	return nil
}

func buildGridGraph(cfg *ArchConfig, supervised bool) (*Program, []WeightMeta, error) {
	if cfg == nil || !cfg.GridEnabled() || cfg.DenseRegression == nil || len(cfg.Blocks) != 1 {
		return nil, nil, fmt.Errorf("invalid grid config")
	}
	spec := cfg.Blocks[0]
	i := cfg.InputAdapter
	inputShape := []int{cfg.Training.BatchSize, i.Height, i.Width, i.Channels}
	if _, err := gridSize(inputShape); err != nil {
		return nil, nil, err
	}
	shapes := map[string][]int{"x": inputShape}
	metas := make([]WeightMeta, len(spec.Weights))
	resolved := spec
	resolved.Weights = append([]WeightSpec(nil), spec.Weights...)
	for n, w := range spec.Weights {
		if w.Name == "" || shapes[w.Name] != nil {
			return nil, nil, fmt.Errorf("duplicate/empty grid weight %q", w.Name)
		}
		s, err := gridWeightShape(w.Shape, cfg)
		if err != nil {
			return nil, nil, err
		}
		shapes[w.Name] = s
		metas[n] = WeightMeta{Name: spec.Name + "." + w.Name, Shape: s}
		resolved.Weights[n].Shape = make([]string, len(s))
		for k, d := range s {
			resolved.Weights[n].Shape[k] = strconv.Itoa(d)
		}
	}
	fanIn := map[string]int{}
	for n, op := range spec.Ops {
		if op.Output == "" || len(op.Outputs) > 0 || shapes[op.Output] != nil {
			return nil, nil, fmt.Errorf("grid op %d requires a unique output name", n)
		}
		s, err := gridOpShape(op, shapes, fanIn)
		if err != nil {
			return nil, nil, fmt.Errorf("grid op %d (%s): %w", n, op.Output, err)
		}
		if _, err = gridSize(s); err != nil {
			return nil, nil, err
		}
		shapes[op.Output] = s
	}
	for n, w := range spec.Weights {
		if w.Init != nil && w.Init.Kind == "pytorch_conv_uniform" && w.Init.Weight != "" {
			paired := false
			for _, op := range spec.Ops {
				if len(op.Inputs) == 3 && (op.Op == "conv2d" || op.Op == "conv_transpose2d") && op.Inputs[1] == w.Init.Weight && op.Inputs[2] == w.Name {
					paired = true
				}
			}
			if !paired {
				return nil, nil, fmt.Errorf("weight %s init.weight must name its convolution kernel", w.Name)
			}
		}
		if err := gridWeightInit(w, &metas[n], fanIn); err != nil {
			return nil, nil, err
		}
	}
	name, ok := strings.CutPrefix(cfg.DenseRegression.Output, spec.Name+".")
	activation := name == "x"
	for _, op := range spec.Ops {
		activation = activation || name == op.Output
	}
	outputShape := []int{cfg.Training.BatchSize, i.Height, i.Width, cfg.DenseRegression.TargetChannels}
	if !ok || !activation || !reflect.DeepEqual(shapes[name], outputShape) {
		return nil, nil, fmt.Errorf("dense_regression.output=%q must name a graph activation with shape %v (got %v)", cfg.DenseRegression.Output, outputShape, shapes[name])
	}
	p := NewProgram(len(metas))
	p.DeclareInput("grid", TensorFloat32, inputShape)
	if _, err := emitCustomBlockIR(p, resolved, "grid", 0, 1, 1, cfg.Training.BatchSize, 1, 0); err != nil {
		return nil, nil, err
	}
	prefix := tmpName("grid_custom_"+strings.ToLower(strings.TrimSpace(spec.Name)), 0) + "_"
	selected := prefix + name
	if name == "x" {
		selected = "grid"
	}
	p.ScalarMul(selected, 1, "predictions")
	// Reachability is rooted only in the selected output, never in declarations.
	needed := map[string]bool{"predictions": true}
	keep := make([]bool, len(p.Ops))
	for n := len(p.Ops) - 1; n >= 0; n-- {
		op := p.Ops[n]
		for _, out := range op.Outputs {
			if needed[out] {
				keep[n] = true
			}
		}
		if keep[n] {
			for _, in := range op.Inputs {
				needed[in] = true
			}
		}
	}
	ops := p.Ops[:0]
	for n, op := range p.Ops {
		if keep[n] {
			ops = append(ops, op)
		}
	}
	p.Ops = ops
	for n := range metas {
		metas[n].Frozen = !needed[weightName(n)]
	}
	if err := resolveGridTrainableWeights(metas, cfg.Training); err != nil {
		return nil, nil, err
	}
	p.DeclareOutput("predictions", TensorFloat32, outputShape)
	if supervised {
		count, err := gridSize(outputShape)
		if err != nil {
			return nil, nil, err
		}
		p.DeclareInput("grid_targets", TensorFloat32, outputShape)
		p.DeclareInput("grid_loss_mask", TensorFloat32, outputShape)
		p.Sub("predictions", "grid_targets", "grid_error")
		p.Square("grid_error", "grid_square")
		p.Mul("grid_square", "grid_loss_mask", "grid_masked_square")
		p.Reshape("grid_masked_square", []int{count}, "grid_errors_flat")
		p.Reshape("grid_loss_mask", []int{count}, "grid_mask_flat")
		p.MeanAxis("grid_errors_flat", 0, "grid_error_mean")
		p.MeanAxis("grid_mask_flat", 0, "grid_mask_mean")
		p.Clamp("grid_mask_mean", 1/float32(count), 1, "grid_denominator")
		p.Div("grid_error_mean", "grid_denominator", "loss")
		p.DeclareOutput("loss", TensorFloat32, []int{1})
	}
	return p, metas, nil
}

func spatialParams(params map[string]interface{}) (kernel, stride, padding int, err error) {
	kernel, stride = 0, 1
	for name, v := range params {
		if name != "kernel" && name != "stride" && name != "padding" && name != "output_padding" {
			return 0, 0, 0, fmt.Errorf("unsupported spatial parameter %q", name)
		}
		f, e := strconv.ParseFloat(fmt.Sprint(v), 64)
		if e != nil || math.IsNaN(f) || math.IsInf(f, 0) || f != math.Trunc(f) || f > math.MaxInt32 || f < math.MinInt32 {
			return 0, 0, 0, fmt.Errorf("%s must be an integer", name)
		}
		n := int(f)
		switch name {
		case "kernel":
			kernel = n
		case "stride":
			stride = n
		case "padding":
			padding = n
		case "output_padding":
			if n != 0 {
				return 0, 0, 0, fmt.Errorf("output_padding must be 0")
			}
		}
	}
	if kernel <= 0 || stride <= 0 || padding < 0 {
		err = fmt.Errorf("kernel/stride must be positive and padding nonnegative")
	}
	return
}

func gridOpShape(op OpSpec, shapes map[string][]int, fanIn map[string]int) ([]int, error) {
	inputs := make([][]int, len(op.Inputs))
	for n, name := range op.Inputs {
		inputs[n] = shapes[name]
		if inputs[n] == nil {
			return nil, fmt.Errorf("unknown input %q", name)
		}
	}
	if len(inputs) == 0 {
		return nil, fmt.Errorf("missing inputs")
	}
	s := append([]int(nil), inputs[0]...)
	switch op.Op {
	case "conv2d", "conv_transpose2d":
		if len(inputs) < 2 || len(inputs) > 3 || len(s) != 4 || len(inputs[1]) != 4 {
			return nil, fmt.Errorf("convolution requires NHWC input, rank-4 weight, optional bias")
		}
		k, st, p, err := spatialParams(op.Params)
		if err != nil {
			return nil, err
		}
		w := inputs[1]
		cin, cout := w[3], w[0]
		if op.Op == "conv_transpose2d" {
			cin, cout = w[0], w[3]
		}
		if s[3] != cin || w[1] != k || w[2] != k {
			return nil, fmt.Errorf("convolution input/kernel channel mismatch")
		}
		if len(inputs) == 3 && !reflect.DeepEqual(inputs[2], []int{cout}) {
			return nil, fmt.Errorf("bias must have shape [%d]", cout)
		}
		fan := cin * k * k
		if op.Op == "conv_transpose2d" {
			fan = cout * k * k
		}
		if old := fanIn[op.Inputs[1]]; old != 0 && old != fan {
			return nil, fmt.Errorf("conflicting convolution fan-in")
		}
		fanIn[op.Inputs[1]] = fan
		for d := 1; d <= 2; d++ {
			if op.Op == "conv2d" {
				if s[d]+2*p < k {
					return nil, fmt.Errorf("kernel exceeds padded input")
				}
				s[d] = (s[d]+2*p-k)/st + 1
			} else {
				s[d] = (s[d]-1)*st - 2*p + k
			}
		}
		s[3] = cout
	case "max_pool2d":
		if len(inputs) != 1 || len(s) != 4 {
			return nil, fmt.Errorf("pool requires one NHWC input")
		}
		k, st, p, err := spatialParams(op.Params)
		if err != nil {
			return nil, err
		}
		if st != k || p != 0 {
			return nil, fmt.Errorf("pool requires stride=kernel and padding=0")
		}
		s[1] /= k
		s[2] /= k
	case "relu", "stop_gradient":
		if len(inputs) != 1 || len(op.Params) > 0 {
			return nil, fmt.Errorf("%s requires one input and no parameters", op.Op)
		}
	case "concat":
		axis, err := gridInteger(op.Params["axis"])
		if err != nil || axis != 3 || len(op.Params) != 1 || len(inputs) != 2 || len(s) != 4 || len(inputs[1]) != 4 {
			return nil, fmt.Errorf("grid concat requires two NHWC inputs and axis=3")
		}
		for d := 0; d < 3; d++ {
			if s[d] != inputs[1][d] {
				return nil, fmt.Errorf("concat spatial mismatch")
			}
		}
		s[3] += inputs[1][3]
	case "transpose":
		values, ok := op.Params["axes"].([]interface{})
		if !ok || len(inputs) != 1 || len(op.Params) != 1 || len(values) != len(s) {
			return nil, fmt.Errorf("grid transpose requires a complete axes permutation")
		}
		seen := map[int]bool{}
		for j, v := range values {
			a, err := gridInteger(v)
			if err != nil || a < 0 || a >= len(s) || seen[a] {
				return nil, fmt.Errorf("invalid transpose axes")
			}
			seen[a] = true
			s[j] = inputs[0][a]
		}
	case "slice":
		if len(inputs) != 1 || len(op.Params) != 4 {
			return nil, fmt.Errorf("grid slice requires start,end,step,axis")
		}
		v := make([]int, 4)
		for j, key := range []string{"start", "end", "step", "axis"} {
			n, err := gridInteger(op.Params[key])
			if err != nil {
				return nil, err
			}
			v[j] = n
		}
		start, end, step, axis := v[0], v[1], v[2], v[3]
		if axis < 0 || axis >= len(s) || step <= 0 || start < 0 || end <= start || end > s[axis] {
			return nil, fmt.Errorf("grid slice is outside input shape")
		}
		s[axis] = (end - start + step - 1) / step
	case "add", "sub", "mul":
		if len(inputs) != 2 || !reflect.DeepEqual(s, inputs[1]) || len(op.Params) > 0 {
			return nil, fmt.Errorf("grid %s requires equal-shaped inputs and no parameters", op.Op)
		}
	default:
		return nil, fmt.Errorf("unsupported grid op %q", op.Op)
	}
	return s, nil
}

func gridInteger(v any) (int, error) {
	f, err := strconv.ParseFloat(fmt.Sprint(v), 64)
	if err != nil || math.IsNaN(f) || math.IsInf(f, 0) || f != math.Trunc(f) || f > math.MaxInt32 || f < math.MinInt32 {
		return 0, fmt.Errorf("grid parameter must be an int32")
	}
	return int(f), nil
}
