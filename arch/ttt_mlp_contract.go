package arch

import (
	"fmt"
	"math"
	"reflect"
)

// InitialStateSlices is also consumed by the runtime packer. Destination order
// is the existing w1,b1,w2,b2 layout, never inferred from parameter names.
func (l TTTMLPStateLayout) InitialStateSlices() []ContractInitSlice {
	lengths := []int{l.Heads * l.HeadDim * l.HiddenDim, l.Heads * l.HiddenDim, l.Heads * l.HiddenDim * l.HeadDim, l.Heads * l.HeadDim}
	var out []ContractInitSlice
	offset := 0
	for i, n := range lengths {
		out = append(out, ContractInitSlice{ParameterID: weightName(l.InitialWeightIndices[i]), DestinationStart: offset, Length: n})
		offset += n
	}
	return out
}

func tttStateLayout(spec BlockSpec, wi, D, blockIndex, ordinal int) (TTTMLPStateLayout, error) {
	if spec.Heads <= 0 || D%spec.Heads != 0 {
		return TTTMLPStateLayout{}, contractError("invalid_shape", "ttt_mlp", "model_dim must divide heads")
	}
	hidden, err := effectiveTTTMLPInnerHiddenDim(spec, D)
	if err != nil {
		return TTTMLPStateLayout{}, err
	}
	headDim := D / spec.Heads
	// Bound the full packed sum, not only its largest individual product.
	matrix, err := contractElements([]int{spec.Heads, headDim, hidden})
	if err != nil {
		return TTTMLPStateLayout{}, err
	}
	bias, err := contractElements([]int{spec.Heads, hidden})
	if err != nil {
		return TTTMLPStateLayout{}, err
	}
	if matrix > (int64(math.MaxInt)/4-bias-int64(D))/2 {
		return TTTMLPStateLayout{}, contractError("invalid_shape", "ttt_mlp", "packed state overflows")
	}
	stateSize := int(2*matrix + bias + int64(D))
	if _, err := contractElements([]int{stateSize}); err != nil {
		return TTTMLPStateLayout{}, err
	}
	prefix := fmt.Sprintf("ttt_state_%d", ordinal)
	return TTTMLPStateLayout{BlockIndex: blockIndex, Heads: spec.Heads, HeadDim: headDim, HiddenDim: hidden, ChunkSize: effectiveTTTMLPChunkSize(spec), StateSize: stateSize, InitialWeightIndices: [4]int{wi + 10, wi + 11, wi + 12, wi + 13}, StateInput: prefix + "_mlp", GradientInput: prefix + "_grad", ConvInput: prefix + "_conv", StateOutput: prefix + "_mlp_next", GradientOutput: prefix + "_grad_next", ConvOutput: prefix + "_conv_next"}, nil
}

func tttContractFromLayout(spec BlockSpec, ctx ContractContext, l TTTMLPStateLayout) BlockContract {
	b := baseBlockContract(spec, ctx)
	stateful := ctx.Profile == NativeTTTStateful
	lifetime, reset, gradient, binding := "within_call", "evaluation", "through_training", "internal_ir"
	if stateful {
		lifetime, reset, gradient, binding = "across_calls", "request_reset", "detached_between_calls", "explicit_ir"
	}
	t := contractTensor(TensorFloat32, []int{ctx.BatchSize, l.StateSize}, "batch", "packed_inner_state")
	for i, role := range []string{"adaptive_memory", "update_accumulator"} {
		in, out := l.StateInput, l.StateOutput
		if i == 1 {
			in, out = l.GradientInput, l.GradientOutput
		}
		init := ContractInitialization{Kind: "zero"}
		if i == 0 {
			init.Kind = "parameters"
			for row := 0; row < ctx.BatchSize; row++ {
				for _, part := range l.InitialStateSlices() {
					part.DestinationStart += row * l.StateSize
					init.Slices = append(init.Slices, part)
				}
			}
		}
		b.States = append(b.States, ContractState{ID: b.OwnerID + "/" + role, OwnerID: b.OwnerID, Role: role, Storage: "tensor", Tensor: &t, Binding: binding, Input: in, Output: out, Initialization: init, Lifetime: lifetime, Reset: reset, Gradient: gradient, Growth: "fixed_bound_dimensions", SchemaVersion: 1, Compatibility: []string{"shape", "dtype", "parameter_content", "packing_w1_b1_w2_b2", "chunk_policy"}})
	}
	if stateful {
		conv := contractTensor(TensorFloat32, []int{1, 2, 3, ctx.ModelDim}, "batch", "qk", "history", "feature")
		b.States = append(b.States, ContractState{ID: b.OwnerID + "/conv", OwnerID: b.OwnerID, Role: "recurrent_cache", Storage: "tensor", Tensor: &conv, Binding: "explicit_ir", Input: l.ConvInput, Output: l.ConvOutput, Initialization: ContractInitialization{Kind: "zero"}, Lifetime: lifetime, Reset: reset, Gradient: gradient, Growth: "fixed_bound_dimensions", SchemaVersion: 1, Compatibility: []string{"shape", "dtype", "convolution_order"}}, ContractState{ID: b.OwnerID + "/offset", OwnerID: b.OwnerID, Role: "control", Storage: "control", Control: &ContractControl{Type: "int", Initial: 0, Min: 0, MaxExclusive: l.ChunkSize}, Binding: "host", Initialization: ContractInitialization{Kind: "zero"}, Lifetime: lifetime, Reset: reset, Gradient: "not_differentiable", Growth: "fixed_bound_dimensions", SchemaVersion: 1, Compatibility: []string{"chunk_size", "algorithmic_offset"}})
	}
	var ids []string
	for _, s := range b.States {
		ids = append(ids, s.ID)
	}
	commit := "evaluation_only"
	if stateful {
		commit = "successful_fragment_after_output_read_no_whole_call_rollback"
	}
	b.Boundaries = []ContractBoundary{{ID: b.OwnerID + "/adapt", OwnerID: b.OwnerID, Stages: []string{"observe", "update", "emit"}, Reads: ids, Writes: append([]string(nil), ids...), ObservationUnit: "ordered_token", Commit: commit, Description: "Fused TTT causal query/update timing follows OpTTTMLPScan/OpTTTMLPStatefulScan; transport fragments respect algorithmic chunk offsets. No independent read-only query or refinement."}}
	b.Boundaries[0].Ordering = "causal_left_to_right"
	b.Boundaries[0].InputValidity = "all_supplied_tokens_observed_no_padding_mask"
	b.Boundaries[0].UpdateFrequency = "accumulate_each_token_commit_inner_mlp_and_clear_accumulator_at_algorithmic_chunk_end"
	b.Boundaries[0].OutputTiming = "query_uses_coefficient_scaled_accumulated_update_including_current_token"
	return b
}

func tttFullContract(spec BlockSpec, ctx ContractContext) (BlockContract, error) {
	l, err := tttStateLayout(spec, ctx.WeightIndex, ctx.ModelDim, ctx.BlockIndex, ctx.SiteIndex)
	if err != nil {
		return BlockContract{}, err
	}
	return tttContractFromLayout(spec, ctx, l), nil
}

func attachTTTStatefulContract(p *Program, cfg *ArchConfig, layouts []TTTMLPStateLayout, metas []WeightMeta, tokenCount int) error {
	for _, l := range layouts {
		ctx := ContractContext{ModelDim: cfg.ModelDim, SeqLen: tokenCount, BatchSize: 1, VocabSize: cfg.VocabSize, BlockIndex: l.BlockIndex, SiteIndex: l.BlockIndex, Profile: NativeTTTStateful}
		b := tttContractFromLayout(cfg.Blocks[l.BlockIndex], ctx, l)
		start := l.InitialWeightIndices[0] - 10
		count, _ := tttMLPWeightCount(cfg.Blocks[l.BlockIndex], cfg.BlockScales, false)
		for i := start; i < start+count; i++ {
			b.ParameterIndices = append(b.ParameterIndices, i)
		}
		for _, op := range p.Ops {
			if op.Code == OpTTTMLPStatefulScan && len(op.Outputs) > 1 && op.Outputs[1] == l.StateOutput {
				want := []int{1, tokenCount, l.Heads, l.HeadDim, l.HiddenDim, l.ChunkSize}
				if len(op.IntParams) != 7 || !reflect.DeepEqual(op.IntParams[:6], want) || op.IntParams[6] < 0 || op.IntParams[6]+tokenCount > l.ChunkSize || len(op.Inputs) != 14 || !reflect.DeepEqual(op.Inputs[11:], []string{l.StateInput, l.GradientInput, l.ConvInput}) || len(op.Outputs) != 4 || !reflect.DeepEqual(op.Outputs[1:], []string{l.StateOutput, l.GradientOutput, l.ConvOutput}) {
					return contractError("invalid_binding", b.BlockID, "stateful scan does not match resolved layout")
				}
				b.Bindings = append(b.Bindings, contractOpBinding(op))
			}
		}
		if len(b.Bindings) != 1 {
			return contractError("invalid_binding", b.BlockID, "missing stateful scan")
		}
		p.BlockContracts = append(p.BlockContracts, b)
	}
	pc := programContract(p)
	if err := ValidateProgramContract(p, &pc); err != nil {
		return err
	}
	c := assembleModelContract(cfg, NativeTTTStateful, p, metas, pc)
	if err := ValidateModelContract(c); err != nil {
		return err
	}
	p.Contract = &pc
	return nil
}
