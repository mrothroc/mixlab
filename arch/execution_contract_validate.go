package arch

import (
	"fmt"
	"math"
	"reflect"
	"slices"
)

func contractElements(shape []int) (int64, error) {
	if len(shape) == 0 {
		return 0, fmt.Errorf("empty tensor shape")
	}
	n := int64(1)
	for _, d := range shape {
		if d <= 0 || int64(d) > math.MaxInt64/n {
			return 0, fmt.Errorf("nonpositive or overflowing dimension")
		}
		n *= int64(d)
	}
	if n > math.MaxInt64/4 {
		return 0, fmt.Errorf("tensor byte count overflows")
	}
	return n, nil
}
func validateContractTensor(t ContractTensor, path string) error {
	if t.DType != TensorInt32 && t.DType != TensorFloat32 {
		return contractError("invalid_shape", path, "unsupported IR dtype")
	}
	if _, err := contractElements(t.Shape); err != nil {
		return contractError("invalid_shape", path, err.Error())
	}
	axes := map[string]int{}
	for _, a := range t.Axes {
		if a.Name == "" || a.Size <= 0 || axes[a.Name] != 0 {
			return contractError("invalid_shape", path, "invalid or duplicate logical axis")
		}
		axes[a.Name] = a.Size
	}
	if len(t.Packing) != len(t.Shape) {
		return contractError("invalid_shape", path, "packing rank differs from physical rank")
	}
	used := map[string]bool{}
	for i, group := range t.Packing {
		var dims []int
		for _, name := range group {
			if axes[name] == 0 || used[name] {
				return contractError("invalid_shape", path, "unknown or repeated packed axis")
			}
			used[name] = true
			dims = append(dims, axes[name])
		}
		n, err := contractElements(dims)
		if err != nil || n != int64(t.Shape[i]) {
			return contractError("invalid_shape", path, "logical packing does not match physical shape")
		}
	}
	if len(used) != len(axes) {
		return contractError("invalid_shape", path, "unpacked logical axis")
	}
	return nil
}
func contractEnum(v, path string, allowed ...string) error {
	if !slices.Contains(allowed, v) {
		return contractError("invalid_policy", path, "invalid value "+v)
	}
	return nil
}

func validateContractState(s ContractState, params map[string]ContractParameter) error {
	p := "states." + s.ID
	if s.ID == "" || s.OwnerID == "" {
		return contractError("invalid_owner", p, "state requires identity and owner")
	}
	if err := contractEnum(s.Role, p+".role", "adaptive_memory", "recurrent_cache", "update_accumulator", "control", "working_state"); err != nil {
		return err
	}
	if err := contractEnum(s.Lifetime, p+".lifetime", "within_call", "across_calls"); err != nil {
		return err
	}
	if err := contractEnum(s.Reset, p+".reset", "evaluation", "request_reset"); err != nil {
		return err
	}
	if (s.Lifetime == "within_call") != (s.Reset == "evaluation") {
		return contractError("invalid_policy", p, "lifetime/reset mismatch")
	}
	if err := contractEnum(s.Gradient, p+".gradient", "through_training", "detached_between_calls", "not_differentiable"); err != nil {
		return err
	}
	if err := contractEnum(s.Growth, p+".growth", "fixed_bound_dimensions", "input_length", "unknown"); err != nil {
		return err
	}
	if s.SchemaVersion != 1 || len(s.Compatibility) == 0 {
		return contractError("incompatible_schema", p, "missing version or compatibility dependencies")
	}
	var size int64
	switch s.Storage {
	case "tensor":
		if s.Tensor == nil || s.Control != nil {
			return contractError("invalid_shape", p, "tensor storage must contain only tensor metadata")
		}
		if err := validateContractTensor(*s.Tensor, p); err != nil {
			return err
		}
		size, _ = contractElements(s.Tensor.Shape)
		if err := contractEnum(s.Binding, p+".binding", "explicit_ir", "internal_ir"); err != nil {
			return err
		}
		if s.Input == "" || s.Output == "" {
			return contractError("invalid_binding", p, "missing tensor bindings")
		}
	case "control":
		if s.Control == nil || s.Tensor != nil || s.Binding != "host" || s.Input != "" || s.Output != "" {
			return contractError("invalid_binding", p, "control must be host-only")
		}
		c := s.Control
		if c.Type != "int" || c.Min < 0 || c.Initial < c.Min || c.Initial >= c.MaxExclusive {
			return contractError("invalid_policy", p, "invalid control range/initial value")
		}
	default:
		return contractError("invalid_policy", p, "unknown storage")
	}
	switch s.Initialization.Kind {
	case "zero":
		if len(s.Initialization.Slices) != 0 || s.Initialization.Value != "" {
			return contractError("invalid_initialization", p, "zero initialization has extra fields")
		}
	case "parameters":
		if s.Tensor == nil || len(s.Initialization.Slices) == 0 || s.Initialization.Value != "" {
			return contractError("invalid_initialization", p, "parameter initializer requires tensor slices")
		}
		end := int64(0)
		for _, part := range s.Initialization.Slices {
			param, ok := params[part.ParameterID]
			n, err := contractElements(param.Shape)
			if !ok || err != nil || part.SourceStart < 0 || part.Length <= 0 || int64(part.SourceStart) > n-int64(part.Length) || int64(part.DestinationStart) != end || int64(part.Length) > size-end {
				return contractError("invalid_initialization", p, "unresolved, overlapping or out-of-range initialization slice")
			}
			end += int64(part.Length)
		}
		if end != size {
			return contractError("invalid_initialization", p, "initialization does not cover tensor")
		}
	case "graph_value":
		if s.Initialization.Value == "" || len(s.Initialization.Slices) != 0 || s.Binding != "internal_ir" {
			return contractError("invalid_initialization", p, "invalid graph initializer")
		}
	default:
		return contractError("invalid_initialization", p, "unknown initializer")
	}
	return nil
}

// ValidateProgramContract checks host-visible bindings without executing the IR.
func ValidateProgramContract(p *Program, c *ProgramContract) error {
	if p == nil || c == nil {
		return contractError("invalid_binding", "program", "nil program or contract")
	}
	inputs, outputs, values := map[string]TensorDecl{}, map[string]TensorDecl{}, map[string]bool{}
	for _, d := range p.Inputs {
		inputs[d.Name] = d
		values[d.Name] = true
	}
	for _, d := range p.Outputs {
		outputs[d.Name] = d
	}
	for i := 0; i < p.NumWeights; i++ {
		values[weightName(i)] = true
	}
	for _, op := range p.Ops {
		for _, name := range op.Outputs {
			values[name] = true
		}
	}
	check := func(name string, t ContractTensor, decls map[string]TensorDecl) error {
		d, ok := decls[name]
		if !ok || d.DType != t.DType || !reflect.DeepEqual(d.Shape, t.Shape) {
			return contractError("invalid_binding", name, "IR name, dtype or physical shape mismatch")
		}
		return validateContractTensor(t, name)
	}
	seen := map[string]bool{}
	for _, r := range c.Representations {
		if r.ID == "" || seen[r.ID] {
			return contractError("invalid_binding", r.ID, "empty/duplicate representation")
		}
		seen[r.ID] = true
		switch r.Direction {
		case "input":
			if err := check(r.Binding, r.Tensor, inputs); err != nil {
				return err
			}
		case "output":
			if err := check(r.Binding, r.Tensor, outputs); err != nil {
				return err
			}
		case "internal":
			if !values[r.Binding] {
				return contractError("invalid_binding", r.ID, "missing internal value")
			}
			if err := validateContractTensor(r.Tensor, r.ID); err != nil {
				return err
			}
		default:
			return contractError("invalid_binding", r.ID, "unknown representation direction")
		}
	}
	bindings := map[string]bool{}
	for _, s := range c.States {
		if s.Tensor != nil {
			if err := validateContractTensor(*s.Tensor, s.ID); err != nil {
				return err
			}
		}
		switch s.Binding {
		case "explicit_ir":
			if s.Tensor == nil {
				return contractError("invalid_binding", s.ID, "missing tensor")
			}
			for _, key := range []string{"in:" + s.Input, "out:" + s.Output} {
				if bindings[key] {
					return contractError("invalid_owner", s.ID, "aliased state binding")
				}
				bindings[key] = true
			}
			if err := check(s.Input, *s.Tensor, inputs); err != nil {
				return err
			}
			if err := check(s.Output, *s.Tensor, outputs); err != nil {
				return err
			}
		case "internal_ir":
			if !values[s.Input] || !values[s.Output] {
				return contractError("invalid_binding", s.ID, "missing internal state producer")
			}
		}
		if s.Initialization.Kind == "graph_value" && !values[s.Initialization.Value] {
			return contractError("invalid_initialization", s.ID, "missing graph initializer")
		}
	}
	for _, edge := range c.DetachedEdges {
		found := false
		for _, op := range p.Ops {
			if op.Code == OpStopGradient && reflect.DeepEqual(op.Inputs, []string{edge.Input}) && reflect.DeepEqual(op.Outputs, []string{edge.Output}) {
				found = true
				break
			}
		}
		if !found {
			return contractError("invalid_effect", edge.Output, "missing detached edge")
		}
	}
	if c.SelectedOutput != "" {
		found := false
		for _, op := range p.Ops {
			if len(op.Outputs) == 1 && op.Outputs[0] == "predictions" && len(op.Inputs) == 1 && op.Inputs[0] == c.SelectedOutput {
				found = true
			}
		}
		if !found {
			return contractError("invalid_binding", "selected_output", "prediction provenance differs")
		}
	}
	for _, b := range c.Ops {
		found := false
		for _, op := range p.Ops {
			if op.Code == b.Code && reflect.DeepEqual(op.Inputs, b.Inputs) && reflect.DeepEqual(op.Outputs, b.Outputs) && reflect.DeepEqual(op.IntParams, b.IntParams) && reflect.DeepEqual(op.FloatParams, b.FloatParams) {
				found = true
				break
			}
		}
		if !found {
			return contractError("invalid_binding", "program.ops", "expected state/gradient operation differs from emitted IR")
		}
	}
	return nil
}

func ValidateModelContract(c *ModelContract) error {
	if c == nil || c.Format != ExecutionContractFormat || c.SchemaVersion != 1 {
		return contractError("incompatible_schema", "model", "unsupported schema")
	}
	if c.Profile != NativeFull && c.Profile != NativeTTTStateful {
		return contractError("unsupported_execution", "profile", "unknown profile")
	}
	if err := contractEnum(c.Coverage, "coverage", "complete", "partial"); err != nil {
		return err
	}
	for _, op := range c.Program.Ops {
		for _, v := range op.FloatParams {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return contractError("invalid_policy", "ops", "nonfinite operation policy")
			}
		}
	}
	owners, params, states := map[string]bool{}, map[string]ContractParameter{}, map[string]ContractState{}
	indices := map[int]bool{}
	for _, p := range c.Parameters {
		if p.ID == "" || p.GroupID == "" {
			return contractError("invalid_owner", "parameters", "missing identity/group")
		}
		if _, ok := params[p.ID]; ok {
			return contractError("invalid_owner", p.ID, "duplicate parameter")
		}
		if p.Index < 0 || indices[p.Index] || p.ID != weightName(p.Index) {
			return contractError("invalid_owner", p.ID, "invalid or duplicate parameter index")
		}
		indices[p.Index] = true
		if _, err := contractElements(p.Shape); err != nil {
			return contractError("invalid_shape", p.ID, err.Error())
		}
		params[p.ID] = p
	}
	for _, b := range c.Blocks {
		if b.OwnerID == "" || b.BlockID == "" || b.ParameterGroupID == "" || owners[b.OwnerID] {
			return contractError("invalid_owner", b.OwnerID, "missing/duplicate execution-site identity")
		}
		owners[b.OwnerID] = true
		for _, idx := range b.ParameterIndices {
			if p, ok := params[weightName(idx)]; !ok || p.GroupID != b.ParameterGroupID {
				return contractError("invalid_owner", b.OwnerID, "unknown parameter reference")
			}
		}
		if !b.Described && c.Coverage == "complete" {
			return contractError("incomplete_contract", b.BlockID, "undescribed block in complete contract")
		}
		for _, s := range b.States {
			if _, ok := states[s.ID]; ok || s.OwnerID != b.OwnerID {
				return contractError("invalid_owner", s.ID, "duplicate state or wrong owner")
			}
			if err := validateContractState(s, params); err != nil {
				return err
			}
			states[s.ID] = s
		}
	}
	if len(states) != len(c.Program.States) {
		return contractError("invalid_binding", "program.states", "state inventory differs from owners")
	}
	bindings := map[string]bool{}
	for _, s := range c.Program.States {
		if !reflect.DeepEqual(states[s.ID], s) {
			return contractError("invalid_binding", s.ID, "state binding differs from owner")
		}
		if s.Binding == "explicit_ir" {
			for _, key := range []string{"in:" + s.Input, "out:" + s.Output} {
				if bindings[key] {
					return contractError("invalid_owner", s.ID, "aliased state binding")
				}
				bindings[key] = true
			}
		}
	}
	boundaries := map[string]bool{}
	for _, b := range c.Blocks {
		for _, e := range b.Boundaries {
			if e.ID == "" || boundaries[e.ID] || e.OwnerID != b.OwnerID || e.Commit == "" || e.ObservationUnit == "" || len(e.Stages) == 0 {
				return contractError("invalid_effect", e.ID, "invalid boundary identity/policy")
			}
			boundaries[e.ID] = true
			for _, stage := range e.Stages {
				if err := contractEnum(stage, e.ID, "observe", "update", "refine", "emit"); err != nil {
					return err
				}
			}
			for _, id := range append(slices.Clone(e.Reads), e.Writes...) {
				s, ok := states[id]
				if !ok || s.OwnerID != e.OwnerID {
					return contractError("invalid_effect", e.ID, "undeclared or foreign state access")
				}
			}
			if slices.Contains(e.Stages, "refine") {
				for _, id := range e.Writes {
					if states[id].Role != "working_state" {
						return contractError("invalid_effect", e.ID, "refinement cannot implicitly mutate persistent state")
					}
				}
			}
		}
	}
	representations := map[string]bool{}
	for _, r := range c.Program.Representations {
		if r.ID == "" || r.Binding == "" || representations[r.ID] {
			return contractError("invalid_binding", r.ID, "missing or duplicate representation identity")
		}
		representations[r.ID] = true
		if err := contractEnum(r.Direction, r.ID, "input", "output", "internal"); err != nil {
			return err
		}
		if err := validateContractTensor(r.Tensor, r.ID); err != nil {
			return err
		}
	}
	caps := map[string]bool{}
	for _, cap := range c.Capabilities {
		if cap.Name == "" || cap.Reason == "" || caps[cap.Name] {
			return contractError("invalid_policy", "capabilities", "missing/duplicate capability or reason")
		}
		caps[cap.Name] = true
		if err := contractEnum(cap.Status, cap.Name, "supported", "unsupported", "unknown"); err != nil {
			return err
		}
		if c.Coverage == "partial" && cap.Status == "supported" {
			return contractError("incomplete_contract", cap.Name, "partial model cannot establish managed capability")
		}
	}
	for _, m := range c.Memory {
		if m.Reason == "" || (m.Bytes != nil && *m.Bytes < 0) {
			return contractError("invalid_policy", m.Kind, "invalid memory estimate")
		}
	}
	return nil
}

func RequireExecution(c *ModelContract, r ExecutionRequest) error {
	if err := ValidateModelContract(c); err != nil {
		return err
	}
	if r.ShareState {
		return contractError("unsupported_execution", "state_sharing", "mutable state aliasing is not implemented")
	}
	if !slices.Contains([]string{"complete", "streaming", "refinement", "save-restore", "clone"}, r.Capability) {
		return contractError("unsupported_execution", "require", "unknown requirement")
	}
	if c.Coverage != "complete" {
		return contractError("incomplete_contract", "coverage", "managed execution requires complete coverage")
	}
	if r.Capability == "complete" {
		return nil
	}
	for _, cap := range c.Capabilities {
		if cap.Name == r.Capability {
			if cap.Status == "supported" {
				return nil
			}
			return contractError("unsupported_execution", r.Capability, cap.Reason)
		}
	}
	return contractError("unsupported_execution", r.Capability, "capability not established")
}
