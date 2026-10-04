package arch

import (
	"fmt"
	"reflect"
)

func contractTensor(dtype int, shape []int, names ...string) ContractTensor {
	t := ContractTensor{DType: dtype, Shape: append([]int(nil), shape...)}
	for i, d := range shape {
		name := fmt.Sprintf("axis_%d", i)
		if len(names) > i {
			name = names[i]
		}
		t.Axes = append(t.Axes, ContractAxis{name, d})
		t.Packing = append(t.Packing, []string{name})
	}
	return t
}
func baseBlockContract(spec BlockSpec, ctx ContractContext) BlockContract {
	id := fmt.Sprintf("blocks[%d]", ctx.BlockIndex)
	group := id
	if spec.WeightGroup != "" {
		group = "weight_group:" + spec.WeightGroup
	}
	return BlockContract{BlockID: id, OwnerID: fmt.Sprintf("%s/site[%d]", id, ctx.SiteIndex), ParameterGroupID: group, Kind: blockTypeKey(spec), Described: true}
}
func pointwiseContract(spec BlockSpec, ctx ContractContext) (BlockContract, error) {
	return baseBlockContract(spec, ctx), nil
}
func recurrentContract(spec BlockSpec, ctx ContractContext) (BlockContract, error) {
	b := baseBlockContract(spec, ctx)
	d := spec.InnerDim
	if d <= 0 {
		d = ctx.ModelDim
	}
	t := contractTensor(TensorFloat32, []int{ctx.BatchSize, d}, "batch", "scan_feature")
	s := ContractState{ID: b.OwnerID + "/carry", OwnerID: b.OwnerID, Role: "recurrent_cache", Storage: "tensor", Tensor: &t, Binding: "internal_ir", Initialization: ContractInitialization{Kind: "zero"}, Lifetime: "within_call", Reset: "evaluation", Gradient: "through_training", Growth: "fixed_bound_dimensions", SchemaVersion: 1, Compatibility: []string{"shape", "dtype", "zero_reset", "sigmoid_decay"}}
	b.States = []ContractState{s}
	b.Boundaries = []ContractBoundary{{ID: b.OwnerID + "/scan", OwnerID: b.OwnerID, Stages: []string{"observe", "update", "emit"}, Reads: []string{s.ID}, Writes: []string{s.ID}, ObservationUnit: "token_in_sequence", Commit: "evaluation_only_no_cross_call_carry", Description: "h[t] = sigmoid(decay)*h[t-1] + (1-sigmoid(decay))*x[t]; each invocation starts at zero"}}
	b.Boundaries[0].Ordering = "causal_left_to_right"
	b.Boundaries[0].InputValidity = "all_scan_input_rows_observed_no_validity_mask"
	b.Boundaries[0].UpdateFrequency = "each_token"
	b.Boundaries[0].OutputTiming = "emit_updated_carry_including_current_token"
	return b, nil
}

func recordBlockContract(p *Program, provider BlockContractProvider, spec BlockSpec, ctx ContractContext, start, next int) error {
	b := baseBlockContract(spec, ctx)
	if provider == nil {
		b.Described = false
	} else {
		var err error
		b, err = provider(spec, ctx)
		if err != nil {
			return err
		}
	}
	for i := ctx.WeightIndex; i < next; i++ {
		b.ParameterIndices = append(b.ParameterIndices, i)
	}
	if b.Described && len(b.States) > 0 {
		code := OpScan
		if b.Kind == "ttt_mlp" {
			code = OpTTTMLPScan
		}
		var scan *Op
		for i := start; i < len(p.Ops); i++ {
			if p.Ops[i].Code == code {
				scan = &p.Ops[i]
				break
			}
		}
		if scan == nil {
			return contractError("invalid_binding", b.BlockID, "missing emitted scan")
		}
		if code == OpScan {
			d := b.States[0].Tensor.Shape[1]
			if !reflect.DeepEqual(scan.IntParams, []int{ctx.BatchSize, ctx.SeqLen, d}) || len(scan.Inputs) != 2 || scan.Inputs[1] != weightName(ctx.WeightIndex+5) {
				return contractError("invalid_binding", b.BlockID, "scan dimensions/decay differ from resolved block")
			}
		} else {
			layout, err := tttStateLayout(spec, ctx.WeightIndex, ctx.ModelDim, ctx.BlockIndex, ctx.SiteIndex)
			if err != nil {
				return err
			}
			if !reflect.DeepEqual(scan.IntParams, []int{ctx.BatchSize, ctx.SeqLen, layout.Heads, layout.HeadDim, layout.HiddenDim, layout.ChunkSize}) || len(scan.Inputs) != 12 || len(scan.Outputs) != 8 {
				return contractError("invalid_binding", b.BlockID, "TTT scan dimensions differ from resolved layout")
			}
			for j, i := range layout.InitialWeightIndices {
				if scan.Inputs[6+j] != weightName(i) {
					return contractError("invalid_initialization", b.BlockID, "TTT scan initializer differs from resolved layout")
				}
			}
		}
		for i := range b.States {
			b.States[i].Input = scan.Inputs[0]
			b.States[i].Output = scan.Outputs[0]
		}
		b.Bindings = append(b.Bindings, contractOpBinding(*scan))
		pc := ProgramContract{States: b.States, Ops: b.Bindings}
		if err := ValidateProgramContract(p, &pc); err != nil {
			return err
		}
	}
	p.BlockContracts = append(p.BlockContracts, b)
	return nil
}
