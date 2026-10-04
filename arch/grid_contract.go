package arch

import "fmt"

// attachGridContract consumes the shape/reachability analysis already performed
// by buildGridGraph. It never calls another builder or reinterprets op shapes.
func attachGridContract(p *Program, cfg *ArchConfig, metas []WeightMeta, shapes map[string][]int, needed map[string]bool, prefix, selected string) error {
	b := baseBlockContract(cfg.Blocks[0], ContractContext{})
	b.Kind = "dense_grid"
	for i := range metas {
		b.ParameterIndices = append(b.ParameterIndices, i)
	}
	b.Boundaries = []ContractBoundary{{ID: b.OwnerID + "/prediction", OwnerID: b.OwnerID, Stages: []string{"emit"}, ObservationUnit: "independent_grid_record", Commit: "evaluation_only_no_cross_record_state", Description: "Configured feed-forward graph; separate stage weights do not imply iterative refinement"}}
	p.BlockContracts = []BlockContract{b}
	pc := programContract(p)
	pc.SelectedOutput = selected
	for j := range pc.Representations {
		r := &pc.Representations[j]
		if len(r.Tensor.Shape) == 4 {
			r.Tensor = contractTensor(r.Tensor.DType, r.Tensor.Shape, "batch", "height", "width", "channel")
		}
		switch r.Binding {
		case "grid_loss_mask":
			r.Meaning = "target_loss_selection_not_input_validity_or_adaptation"
		case "grid_targets":
			r.Meaning = "supervised_target"
		case "grid":
			r.Meaning = "input_grid_nhwc"
		case "predictions":
			r.Meaning = "selected_output_grid_nhwc"
		}
	}
	// Preserve axis provenance through explicit transposes, including detached
	// activations; element-count equality cannot establish layout compatibility.
	axes := map[string][]string{"x": {"batch", "height", "width", "channel"}}
	for _, op := range cfg.Blocks[0].Ops {
		names := append([]string(nil), axes[op.Inputs[0]]...)
		if len(names) == 0 {
			for j := range shapes[op.Output] {
				names = append(names, fmt.Sprintf("axis_%d", j))
			}
		}
		if op.Op == "transpose" {
			original := append([]string(nil), names...)
			for j, v := range op.Params["axes"].([]interface{}) {
				a, _ := gridInteger(v)
				names[j] = original[a]
			}
		}
		if op.Op == "conv2d" || op.Op == "conv_transpose2d" || op.Op == "max_pool2d" {
			names = []string{"batch", "height", "width", "channel"}
		}
		axes[op.Output] = names
		out := prefix + op.Output
		if !needed[out] {
			continue
		}
		pc.Representations = append(pc.Representations, ContractRepresentation{ID: "internal:" + out, Binding: out, Direction: "internal", Meaning: "graph_activation", Tensor: contractTensor(TensorFloat32, shapes[op.Output], names...)})
		if op.Op == "stop_gradient" {
			in := prefix + op.Inputs[0]
			if op.Inputs[0] == "x" {
				in = "grid"
			}
			pc.DetachedEdges = append(pc.DetachedEdges, ContractEdge{in, out})
		}
	}
	// Record the exact detached operations and output selection so validation
	// catches changes to gradient boundaries and provenance, not just shapes.
	for _, op := range p.Ops {
		if op.Code == OpStopGradient || (len(op.Outputs) == 1 && op.Outputs[0] == "predictions") {
			pc.Ops = append(pc.Ops, contractOpBinding(op))
		}
	}
	if err := ValidateProgramContract(p, &pc); err != nil {
		return err
	}
	c := assembleModelContract(cfg, NativeFull, p, metas, pc)
	if err := ValidateModelContract(c); err != nil {
		return err
	}
	p.Contract = &pc
	return nil
}
