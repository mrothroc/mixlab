package arch

import (
	"encoding/json"
	"math"
	"reflect"
	"strings"
	"testing"
)

func contractFixture(t testing.TB, name string) *ArchConfig {
	t.Helper()
	cfg, err := LoadArchConfigQuiet("testdata/execution_contracts/" + name + ".json")
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}
func fixtureContract(t testing.TB, name string, profile ExecutionProfile) *ModelContract {
	t.Helper()
	c, err := DescribeModelContract(contractFixture(t, name), profile)
	if err != nil {
		t.Fatal(err)
	}
	return c
}
func TestExecutionContractTTTLayoutAndMemory(t *testing.T) {
	cfg := contractFixture(t, "ttt")
	for _, offset := range []int{0, 3} {
		p, l, err := BuildTTTMLPStatefulInferenceIRProgram(cfg, 1, []int{offset})
		if err != nil {
			t.Fatal(err)
		}
		if l[0].StateSize != 296 || len(p.Contract.States) != 4 {
			t.Fatalf("layout=%+v states=%d", l, len(p.Contract.States))
		}
		if err := ValidateProgramContract(p, p.Contract); err != nil {
			t.Fatal(err)
		}
		for _, s := range p.Contract.States {
			if s.Storage == "control" && (s.Control.Initial != 0 || s.Control.MaxExclusive != 4) {
				t.Fatal(s)
			}
		}
		parts := l[0].InitialStateSlices()
		want := []int{128, 32, 128, 8}
		end := 0
		for j, part := range parts {
			if part.Length != want[j] || part.DestinationStart != end {
				t.Fatal(parts)
			}
			end += part.Length
		}
		p.Inputs[1].Shape = []int{1, 295}
		if ValidateProgramContract(p, p.Contract) == nil {
			t.Fatal("accepted mismatched packed input")
		}
	}
	c := fixtureContract(t, "ttt", NativeTTTStateful)
	if err := RequireExecution(c, ExecutionRequest{Capability: "streaming"}); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, m := range c.Memory {
		if m.Kind == "persistent_tensor_payload" {
			found = true
			if m.Bytes == nil || *m.Bytes != 2560 {
				t.Fatal(m)
			}
		}
	}
	if !found || c.CheckpointIdentity != nil || c.Lifecycle == nil {
		t.Fatal("missing payload/lifecycle or invented checkpoint identity")
	}
	for _, cap := range []string{"refinement", "save-restore", "clone"} {
		if RequireExecution(c, ExecutionRequest{Capability: cap}) == nil {
			t.Fatal("unsupported capability", cap)
		}
	}
	if RequireExecution(c, ExecutionRequest{Capability: "streaming", ShareState: true}) == nil {
		t.Fatal("accepted alias")
	}
}

func TestExecutionContractFullAndPartialCoverage(t *testing.T) {
	for _, name := range []string{"ttt", "recurrent"} {
		c := fixtureContract(t, name, NativeFull)
		if c.Coverage != "complete" {
			t.Fatal(c.Diagnostics)
		}
		if RequireExecution(c, ExecutionRequest{Capability: "streaming"}) == nil {
			t.Fatal("full forward claimed streaming")
		}
	}
	cfg := contractFixture(t, "recurrent")
	cfg.Blocks = []BlockSpec{{Type: "plain", Heads: 1}}
	c, err := DescribeModelContract(cfg, NativeFull)
	if err != nil {
		t.Fatal(err)
	}
	if c.Coverage != "partial" || len(c.Diagnostics) == 0 {
		t.Fatal("missing provider was treated as stateless")
	}
	if RequireExecution(c, ExecutionRequest{Capability: "complete"}) == nil {
		t.Fatal("partial accepted")
	}
	for _, m := range c.Memory {
		if m.Bytes != nil {
			t.Fatal("unknown memory reported as known", m)
		}
	}
	if _, err := DescribeModelContract(cfg, NativeTTTStateful); err == nil {
		t.Fatal("unsupported stateful profile accepted")
	}
	cfg.Blocks = []BlockSpec{{Type: "swiglu"}, {Type: "geglu"}, {Type: "mlp"}}
	c, err = DescribeModelContract(cfg, NativeFull)
	if err != nil {
		t.Fatal(err)
	}
	if c.Coverage != "complete" || len(c.Program.States) != 0 {
		t.Fatal("pointwise contracts", c)
	}
}

func TestExecutionContractSchemaRejectsMutations(t *testing.T) {
	base := fixtureContract(t, "ttt", NativeTTTStateful)
	// Locate the TTT owner without depending on the canonical block ordering.
	owner := 0
	for i, b := range base.Blocks {
		if b.Kind == "ttt_mlp" {
			owner = i
		}
	}
	cases := map[string]func(*ModelContract){
		"negative shape":      func(c *ModelContract) { c.Blocks[owner].States[0].Tensor.Shape[0] = -1 },
		"overflow":            func(c *ModelContract) { c.Blocks[owner].States[0].Tensor.Shape = []int{math.MaxInt, 8} },
		"duplicate owner":     func(c *ModelContract) { c.Blocks = append(c.Blocks, c.Blocks[owner]) },
		"foreign state owner": func(c *ModelContract) { c.Blocks[owner].States[0].OwnerID = "foreign" },
		"undeclared write":    func(c *ModelContract) { c.Blocks[owner].Boundaries[0].Writes = []string{"absent"} },
		"unsafe refinement":   func(c *ModelContract) { c.Blocks[owner].Boundaries[0].Stages = []string{"refine"} },
		"initializer": func(c *ModelContract) {
			for i := range c.Blocks[owner].States {
				s := &c.Blocks[owner].States[i]
				if s.Initialization.Kind == "parameters" {
					s.Initialization.Slices[0].ParameterID = "missing"
				}
			}
		},
		"overlap": func(c *ModelContract) {
			for i := range c.Blocks[owner].States {
				s := &c.Blocks[owner].States[i]
				if s.Initialization.Kind == "parameters" {
					s.Initialization.Slices[1].DestinationStart = 0
				}
			}
		},
		"offset": func(c *ModelContract) {
			for i := range c.Blocks[owner].States {
				s := &c.Blocks[owner].States[i]
				if s.Control != nil {
					s.Control.Initial = s.Control.MaxExclusive
				}
			}
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			c := base.Clone()
			mutate(c)
			if ValidateModelContract(c) == nil {
				t.Fatal("invalid contract accepted")
			}
		})
	}
}

func TestExecutionContractDigestAndSnapshots(t *testing.T) {
	c := fixtureContract(t, "ttt", NativeTTTStateful)
	data, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	var decoded ModelContract
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	digest, err := ContractSchemaDigest(&decoded)
	if err != nil || digest != c.SchemaDigest {
		t.Fatal("roundtrip", digest, err)
	}
	prose := c.Clone()
	prose.Diagnostics = nil
	prose.SourceRevision = "different"
	prose.ConfigDigest = "different"
	for i := range prose.Blocks {
		for j := range prose.Blocks[i].Boundaries {
			prose.Blocks[i].Boundaries[j].Description = "new prose"
		}
	}
	for i := range prose.Capabilities {
		prose.Capabilities[i].Reason = "new explanation"
	}
	digest, err = ContractSchemaDigest(prose)
	if err != nil || digest != c.SchemaDigest {
		t.Fatal("prose changed semantics", err)
	}
	for _, what := range []string{"offset", "initializer", "effect", "dtype", "shape"} {
		t.Run(what, func(t *testing.T) {
			changed := c.Clone()
			for bi := range changed.Blocks {
				b := &changed.Blocks[bi]
				if b.Kind != "ttt_mlp" {
					continue
				}
				for si := range b.States {
					s := &b.States[si]
					switch what {
					case "offset":
						if s.Control != nil {
							s.Control.MaxExclusive++
						}
					case "initializer":
						if s.Initialization.Kind == "parameters" {
							s.Initialization.Slices[0].ParameterID = s.Initialization.Slices[2].ParameterID
						}
					case "dtype":
						if s.Tensor != nil {
							s.Tensor.DType = TensorInt32
						}
					}
					for pi := range changed.Program.States {
						if changed.Program.States[pi].ID == s.ID {
							changed.Program.States[pi] = *s
						}
					}
				}
				if what == "effect" {
					b.Boundaries[0].Stages = []string{"update", "emit"}
				}
			}
			if what == "shape" {
				r := &changed.Program.Representations[0]
				r.Tensor.Shape[0]++
				r.Tensor.Axes[0].Size++
			}
			d, err := ContractSchemaDigest(changed)
			if err != nil {
				t.Fatal(err)
			}
			if d == c.SchemaDigest {
				t.Fatal("semantic change did not change digest")
			}
		})
	}
	if !reflect.DeepEqual(c, decoded.Clone()) {
		t.Fatal("report alias/roundtrip changed original")
	}
	cfg := contractFixture(t, "ttt")
	cfg.SourcePath = "/different/path"
	other, err := DescribeModelContract(cfg, NativeTTTStateful)
	if err != nil || c.ConfigDigest != other.ConfigDigest {
		t.Fatal("path affects digest", err)
	}
}

func TestExecutionContractGridBindings(t *testing.T) {
	cfg, err := LoadArchConfigQuiet("../examples/grid_two_stage_2.json")
	if err != nil {
		t.Fatal(err)
	}
	p, metas, err := buildGridGraph(cfg, true)
	if err != nil {
		t.Fatal(err)
	}
	c, err := DescribeProgramContract(cfg, NativeFull, p, metas)
	if err != nil {
		t.Fatal(err)
	}
	if c.Coverage != "complete" || len(c.Program.States) != 0 || len(c.Program.DetachedEdges) != 1 {
		t.Fatalf("grid contract %+v", c)
	}
	if !strings.HasSuffix(c.Program.SelectedOutput, "output2") {
		t.Fatal("wrong provenance")
	}
	if !c.Parameters[0].Frozen || !c.Parameters[0].Reachable {
		t.Fatal("frozen != unreachable")
	}
	for _, r := range c.Program.Representations {
		if r.Binding == "grid" && !reflect.DeepEqual(r.Tensor.Shape, []int{2, 8, 8, 2}) {
			t.Fatal(r)
		}
		if r.Binding == "grid_loss_mask" && !strings.Contains(r.Meaning, "target_loss") {
			t.Fatal(r)
		}
	}
	bad := c.Clone()
	for i := range bad.Program.Representations {
		r := &bad.Program.Representations[i]
		if r.Binding == "predictions" {
			r.Tensor = contractTensor(TensorFloat32, []int{2, 1, 8, 8}, "batch", "channel", "height", "width")
		}
	}
	if ValidateProgramContract(p, &bad.Program) == nil {
		t.Fatal("equal element count accepted as layout compatibility")
	}
	cfg.DenseRegression.Output = "network.output1"
	cfg.Training.Freeze = nil
	p, metas, err = buildGridGraph(cfg, false)
	if err != nil {
		t.Fatal(err)
	}
	c, err = DescribeProgramContract(cfg, NativeFull, p, metas)
	if err != nil {
		t.Fatal(err)
	}
	if c.Parameters[2].Reachable || !c.Parameters[2].Frozen || len(c.Program.DetachedEdges) != 0 {
		t.Fatal("pruned stage misreported")
	}
}

func BenchmarkExecutionContractTTT(b *testing.B) {
	cfg := contractFixture(b, "ttt")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DescribeModelContract(cfg, NativeTTTStateful); err != nil {
			b.Fatal(err)
		}
	}
}
