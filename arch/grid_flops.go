package arch

import "strings"

func estimateGridFLOPs(cfg *ArchConfig) FLOPsEstimate {
	est := FLOPsEstimate{}
	p, metas, err := buildGridGraph(cfg, false)
	if err != nil {
		return est
	}
	for _, m := range metas {
		n := int64(1)
		for _, d := range m.Shape {
			n *= int64(d)
		}
		est.ParamCount += n
	}
	est.ExpandedParamCount = est.ParamCount
	shapes := map[string][]int{"x": {cfg.Training.BatchSize, cfg.InputAdapter.Height, cfg.InputAdapter.Width, cfg.InputAdapter.Channels}}
	for _, w := range cfg.Blocks[0].Weights {
		s, e := gridWeightShape(w.Shape, cfg)
		if e != nil {
			return est
		}
		shapes[w.Name] = s
	}
	// Count only reachable operations; a full inventory may include idle stages.
	used := map[string]bool{}
	for _, op := range p.Ops {
		for _, out := range op.Outputs {
			used[out] = true
		}
	}
	prefix := tmpName("grid_custom_"+strings.ToLower(strings.TrimSpace(cfg.Blocks[0].Name)), 0) + "_"
	for _, op := range cfg.Blocks[0].Ops {
		s, e := gridOpShape(op, shapes, map[string]int{})
		if e != nil {
			return FLOPsEstimate{}
		}
		shapes[op.Output] = s
		if !used[prefix+op.Output] {
			continue
		}
		n := int64(1)
		for _, d := range s {
			n *= int64(d)
		}
		switch op.Op {
		case "conv2d":
			w := shapes[op.Inputs[1]]
			est.ForwardFLOPs += 2 * n * int64(w[1]) * int64(w[2]) * int64(w[3])
		case "conv_transpose2d":
			in := shapes[op.Inputs[0]]
			w := shapes[op.Inputs[1]]
			est.ForwardFLOPs += 2 * int64(in[0]) * int64(in[1]) * int64(in[2]) * int64(in[3]) * int64(w[1]) * int64(w[2]) * int64(w[3])
		case "relu", "add", "sub", "mul":
			est.ForwardFLOPs += n
		case "max_pool2d":
			k, _, _, _ := spatialParams(op.Params)
			est.ForwardFLOPs += n * int64(k*k-1)
		}
	}
	// Forward is analytical; backend convolution backward costs are not modeled.
	return est
}
