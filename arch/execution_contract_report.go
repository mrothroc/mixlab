package arch

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"runtime/debug"
	"sort"
)

func programContract(p *Program) ProgramContract {
	c := ProgramContract{}
	for _, side := range []struct {
		direction string
		decls     []TensorDecl
	}{{"input", p.Inputs}, {"output", p.Outputs}} {
		for _, d := range side.decls {
			c.Representations = append(c.Representations, ContractRepresentation{ID: side.direction + ":" + d.Name, Binding: d.Name, Direction: side.direction, Meaning: d.Name, Tensor: contractTensor(d.DType, d.Shape)})
		}
	}
	for _, b := range p.BlockContracts {
		c.States = append(c.States, b.States...)
		c.Ops = append(c.Ops, b.Bindings...)
	}
	return c
}

func assembleModelContract(cfg *ArchConfig, profile ExecutionProfile, p *Program, metas []WeightMeta, pc ProgramContract) *ModelContract {
	c := &ModelContract{Format: ExecutionContractFormat, SchemaVersion: 1, Profile: profile, ComputeDType: cfg.Training.EffectiveComputeDType(), Coverage: "complete", SourceRevision: "unknown", Blocks: append([]BlockContract(nil), p.BlockContracts...), Program: pc}
	if profile == NativeTTTStateful {
		c.ComputeDType = "float32"
		c.Lifecycle = &ContractLifecycle{StateOwnership: "request_owned_session_bound_no_aliasing", Concurrency: "same_goroutine_locked_os_thread_not_concurrent", ResetFailure: "old_handles_freed_before_reallocation_no_rollback", Close: "state_close_idempotent_session_close_closes_all_states", OutputRetention: []string{"Prefill:all_fragment_logits", "PrefillLast:last_token_logits", "Decode:last_token_logits", "program_variants:session_cached_excluded_from_state_payload"}}
	}
	if info, ok := debug.ReadBuildInfo(); ok {
		modified := false
		for _, s := range info.Settings {
			if s.Key == "vcs.revision" {
				c.SourceRevision = s.Value
			}
			if s.Key == "vcs.modified" && s.Value == "true" {
				modified = true
			}
		}
		if modified {
			c.SourceRevision += "+modified"
		}
	}
	groups := map[int]string{}
	described := map[string]bool{}
	for bi := range c.Blocks {
		b := &c.Blocks[bi]
		described[b.BlockID] = true
		if len(b.ParameterIndices) > 0 {
			if shared := groups[b.ParameterIndices[0]]; shared != "" {
				b.ParameterGroupID = shared
			}
		}
		for _, i := range b.ParameterIndices {
			groups[i] = b.ParameterGroupID
		}
	}
	// Some specialized/parallel builders do not use EmitBlock. Never infer
	// coverage from the registry alone when no execution-site binding exists.
	for i, b := range cfg.Blocks {
		id := fmt.Sprintf("blocks[%d]", i)
		if !described[id] {
			missing := baseBlockContract(b, ContractContext{BlockIndex: i, SiteIndex: i})
			missing.Described = false
			c.Blocks = append(c.Blocks, missing)
		}
	}
	for _, b := range c.Blocks {
		if !b.Described {
			c.Coverage = "partial"
			c.Diagnostics = append(c.Diagnostics, ContractError{"incomplete_contract", b.BlockID, "execution-site contract is not implemented"})
		}
	}
	used := map[string]bool{}
	for _, op := range p.Ops {
		for _, in := range op.Inputs {
			used[in] = true
		}
	}
	for i, m := range metas {
		group := groups[i]
		if group == "" {
			group = "model"
		}
		c.Parameters = append(c.Parameters, ContractParameter{ID: weightName(i), Index: i, Name: m.Name, GroupID: group, Shape: append([]int(nil), m.Shape...), Frozen: m.Frozen, Buffer: m.IsBuffer, Reachable: used[weightName(i)]})
		if m.IsBuffer {
			c.Coverage = "partial"
			c.Diagnostics = append(c.Diagnostics, ContractError{"incomplete_contract", weightName(i), "mutable model-buffer effects are not described"})
		}
	}
	for _, name := range []string{"streaming", "refinement", "save-restore", "clone"} {
		status, reason := "unsupported", "no runtime implementation for this profile"
		if c.Coverage == "partial" {
			status, reason = "unknown", "not all execution sites are described"
		} else if name == "streaming" && profile == NativeTTTStateful {
			status, reason = "supported", "existing batch-one TTT request session; ordered chunk fragments"
		}
		c.Capabilities = append(c.Capabilities, ContractCapability{name, status, reason})
	}
	var persistent, carry int64
	overflow := false
	for _, s := range pc.States {
		if s.Tensor == nil {
			continue
		}
		n, err := contractElements(s.Tensor.Shape)
		if err != nil {
			overflow = true
			continue
		}
		bytes := n * 4
		dst := &carry
		if s.Lifetime == "across_calls" {
			dst = &persistent
		}
		if *dst > math.MaxInt64-bytes {
			overflow = true
		} else {
			*dst += bytes
		}
	}
	if c.Coverage == "complete" && !overflow {
		c.Memory = append(c.Memory, ContractMemory{"persistent_tensor_payload", &persistent, "FP32 tensors only; excludes host controls, weights, programs, outputs and allocator overhead"}, ContractMemory{"conceptual_within_call_carry", &carry, "logical carry only, not peak scan/autodiff allocation"})
	} else {
		c.Memory = append(c.Memory, ContractMemory{"persistent_tensor_payload", nil, "incomplete inventory or overflowing total"}, ContractMemory{"conceptual_within_call_carry", nil, "incomplete inventory or overflowing total"})
	}
	c.Memory = append(c.Memory, ContractMemory{"host_control_payload", nil, "Go control/object allocation is implementation-dependent"}, ContractMemory{"peak_device_memory", nil, "requires runtime measurement; not inferred from tensor payload"})
	return c
}

// DescribeProgramContract aggregates an already-built graph without rebuilding
// it or loading weights. Callers receive a deep, independently editable snapshot.
func DescribeProgramContract(cfg *ArchConfig, profile ExecutionProfile, p *Program, metas []WeightMeta) (*ModelContract, error) {
	if cfg == nil || p == nil {
		return nil, contractError("invalid_binding", "model", "nil config or program")
	}
	if len(metas) != p.NumWeights {
		return nil, contractError("invalid_binding", "parameters", "weight inventory differs from emitted program")
	}
	pc := programContract(p)
	if p.Contract != nil {
		pc = *p.Contract
	}
	c := assembleModelContract(cfg, profile, p, metas, pc).Clone()
	// Name the reversible [B*T,D] packing explicitly at sequence boundaries.
	B, T := 0, 0
	for _, r := range c.Program.Representations {
		if r.Binding == "tokens" && len(r.Tensor.Shape) == 2 {
			B, T = r.Tensor.Shape[0], r.Tensor.Shape[1]
		}
	}
	if B > 0 {
		for i := range c.Program.Representations {
			r := &c.Program.Representations[i]
			s := r.Tensor.Shape
			if r.Binding == "tokens" {
				r.Tensor = contractTensor(r.Tensor.DType, s, "batch", "time")
			}
			if r.Binding == "logits" && len(s) == 2 && s[0] == B*T {
				r.Tensor = ContractTensor{DType: r.Tensor.DType, Shape: s, Axes: []ContractAxis{{"batch", B}, {"time", T}, {"vocab", s[1]}}, Packing: [][]string{{"batch", "time"}, {"vocab"}}}
			}
		}
	}
	if err := ValidateProgramContract(p, &c.Program); err != nil {
		return nil, err
	}
	if err := ValidateModelContract(c); err != nil {
		return nil, err
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		return nil, err
	}
	c.ConfigDigest = contractHash(data)
	CanonicalizeContract(c)
	c.SchemaDigest, err = ContractSchemaDigest(c)
	return c, err
}

// DescribeModelContract inspects a normalized configuration using native graph
// builders only. No checkpoint, dataset, device, or MLX initialization is needed.
func DescribeModelContract(cfg *ArchConfig, profile ExecutionProfile) (*ModelContract, error) {
	if cfg == nil {
		return nil, contractError("invalid_config", "config", "nil configuration")
	}
	if profile == "" {
		profile = NativeFull
	}
	var p *Program
	var metas []WeightMeta
	var err error
	switch profile {
	case NativeTTTStateful:
		n := 0
		for _, b := range cfg.Blocks {
			if blockTypeKey(b) == "ttt_mlp" {
				n++
			}
		}
		p, _, err = BuildTTTMLPStatefulInferenceIRProgram(cfg, 1, make([]int, n))
	case NativeFull:
		if cfg.GridEnabled() {
			p, metas, err = buildGridGraph(cfg, true)
		} else {
			p, err = BuildIRProgramFromConfig(cfg)
		}
	default:
		return nil, contractError("unsupported_execution", "contract_profile", "unknown profile")
	}
	if err != nil {
		return nil, contractError("unsupported_execution", string(profile), err.Error())
	}
	if metas == nil {
		metas, err = CollectWeightShapesFromConfig(cfg)
		if err != nil {
			return nil, err
		}
	}
	return DescribeProgramContract(cfg, profile, p, metas)
}

func contractHash(data []byte) string { s := sha256.Sum256(data); return hex.EncodeToString(s[:]) }

// CanonicalizeContract sorts identity-keyed collections only. Stage, initializer,
// axis, and op orders are semantic and remain untouched.
func CanonicalizeContract(c *ModelContract) {
	sort.Slice(c.Blocks, func(i, j int) bool { return c.Blocks[i].OwnerID < c.Blocks[j].OwnerID })
	for i := range c.Blocks {
		b := &c.Blocks[i]
		sort.Slice(b.States, func(i, j int) bool { return b.States[i].ID < b.States[j].ID })
		sort.Slice(b.Boundaries, func(i, j int) bool { return b.Boundaries[i].ID < b.Boundaries[j].ID })
	}
	sort.Slice(c.Parameters, func(i, j int) bool { return c.Parameters[i].ID < c.Parameters[j].ID })
	sort.Slice(c.Program.States, func(i, j int) bool { return c.Program.States[i].ID < c.Program.States[j].ID })
	sort.Slice(c.Program.Representations, func(i, j int) bool { return c.Program.Representations[i].ID < c.Program.Representations[j].ID })
	sort.Slice(c.Capabilities, func(i, j int) bool { return c.Capabilities[i].Name < c.Capabilities[j].Name })
	sort.Slice(c.Memory, func(i, j int) bool { return c.Memory[i].Kind < c.Memory[j].Kind })
}

func ContractSchemaDigest(c *ModelContract) (string, error) {
	if err := ValidateModelContract(c); err != nil {
		return "", err
	}
	semantic := c.Clone()
	CanonicalizeContract(semantic)
	semantic.ConfigDigest = ""
	semantic.SchemaDigest = ""
	semantic.SourceRevision = ""
	semantic.CheckpointIdentity = nil
	semantic.Diagnostics = nil
	for i := range semantic.Blocks {
		for j := range semantic.Blocks[i].Boundaries {
			semantic.Blocks[i].Boundaries[j].Description = ""
		}
	}
	for i := range semantic.Capabilities {
		semantic.Capabilities[i].Reason = ""
	}
	for i := range semantic.Memory {
		semantic.Memory[i].Reason = ""
	}
	data, err := json.Marshal(semantic)
	if err != nil {
		return "", err
	}
	return contractHash(data), nil
}
