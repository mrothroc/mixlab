package arch

// Custom-block JSON schema types: a custom block declares its named weights
// (with symbolic shapes) and the ordered ops it emits. The Custom* aliases are
// kept source-compatible for older tests and callers.

// WeightSpec declares a named weight for a custom block with symbolic shape.
type WeightSpec struct {
	Name  string   `json:"name"`
	Shape []string `json:"shape"`
}

// CustomWeightSpec is kept as a source-compatible alias for older tests and callers.
type CustomWeightSpec = WeightSpec

// OpSpec declares one operation in a custom block.
type OpSpec struct {
	Op      string                 `json:"op"`
	Inputs  []string               `json:"inputs"`
	Output  string                 `json:"output"`
	Outputs []string               `json:"outputs"`
	Params  map[string]interface{} `json:"params"`
}

// CustomOpSpec is kept as a source-compatible alias for older tests and callers.
type CustomOpSpec = OpSpec
