package arch

import "fmt"

// GridConvolutionDiagnostics resolves graph geometry using the same shape rules
// as IR construction. These are candidates, not a claim that lazy backend
// execution identified a particular source op.
func GridConvolutionDiagnostics(cfg *ArchConfig) ([]string, error) {
	if cfg == nil || !cfg.GridEnabled() || len(cfg.Blocks) != 1 {
		return nil, fmt.Errorf("convolution diagnostics require a grid graph")
	}
	i := cfg.InputAdapter
	shapes := map[string][]int{"x": {cfg.Training.BatchSize, i.Height, i.Width, i.Channels}}
	for _, w := range cfg.Blocks[0].Weights {
		s, err := gridWeightShape(w.Shape, cfg)
		if err != nil {
			return nil, err
		}
		shapes[w.Name] = s
	}
	var result []string
	for _, op := range cfg.Blocks[0].Ops {
		s, err := gridOpShape(op, shapes, map[string]int{})
		if err != nil {
			return nil, err
		}
		shapes[op.Output] = s
		if op.Op == "conv2d" || op.Op == "conv_transpose2d" {
			_, stride, padding, dilation, err := spatialOpParams(op)
			if err != nil {
				return nil, err
			}
			result = append(result, fmt.Sprintf("%s.%s op=%s input=%v weight=%v output=%v stride=%d padding=%d dilation=%d",
				cfg.Blocks[0].Name, op.Output, op.Op, shapes[op.Inputs[0]], shapes[op.Inputs[1]], s, stride, padding, dilation))
		}
	}
	return result, nil
}
