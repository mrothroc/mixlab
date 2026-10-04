package arch

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestExecutionContractSharingSeparatesState(t *testing.T) {
	for _, sharing := range []string{`"recurrence":[0,0],`, ""} {
		group := ""
		if sharing == "" {
			group = `,"weight_group":"shared"`
		}
		cfg, err := ParseArchConfig([]byte(`{"model_dim":4,"vocab_size":16,"seq_len":3,`+sharing+`"blocks":[{"type":"gated_linear_ssm"`+group+`},{"type":"gated_linear_ssm"`+group+`}],"training":{"batch_tokens":6}}`), "sharing")
		if err != nil {
			t.Fatal(err)
		}
		c, err := DescribeModelContract(cfg, NativeFull)
		if err != nil {
			t.Fatal(err)
		}
		if len(c.Blocks) != 2 || c.Coverage != "complete" {
			t.Fatal(c.Diagnostics)
		}
		a, b := c.Blocks[0], c.Blocks[1]
		if !reflect.DeepEqual(a.ParameterIndices, b.ParameterIndices) || a.ParameterGroupID != b.ParameterGroupID {
			t.Fatal("shared weights not identified")
		}
		if a.OwnerID == b.OwnerID || a.States[0].ID == b.States[0].ID || a.States[0].Output == b.States[0].Output {
			t.Fatal("shared weights aliased state")
		}
		if a.States[0].Reset != "evaluation" || b.States[0].Reset != "evaluation" {
			t.Fatal("implicit cross-call carry")
		}
	}
}

func TestExecutionContractMetadataPreservesLegacyIR(t *testing.T) {
	for _, name := range []string{"ttt", "recurrent"} {
		t.Run(name, func(t *testing.T) {
			cfg := contractFixture(t, name)
			before, err := BuildIRProgramFromConfig(cfg)
			if err != nil {
				t.Fatal(err)
			}
			metas, err := CollectWeightShapesFromConfig(cfg)
			if err != nil {
				t.Fatal(err)
			}
			old := map[string]BlockRegistration{}
			for _, b := range cfg.Blocks {
				k := blockTypeKey(b)
				old[k] = registry[k]
				reg := registry[k]
				reg.DescribeContract = nil
				registry[k] = reg
			}
			defer func() {
				for k, reg := range old {
					registry[k] = reg
				}
			}()
			after, err := BuildIRProgramFromConfig(cfg)
			if err != nil {
				t.Fatal(err)
			}
			x, _ := json.Marshal(before)
			y, _ := json.Marshal(after)
			if string(x) != string(y) {
				t.Fatal("contract changed serialized IR")
			}
			afterMetas, err := CollectWeightShapesFromConfig(cfg)
			if err != nil || !reflect.DeepEqual(metas, afterMetas) {
				t.Fatal("contract changed checkpoint layout", err)
			}
		})
	}
}

func TestExecutionContractScanMutationRejected(t *testing.T) {
	cfg := contractFixture(t, "recurrent")
	p, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	c := programContract(p)
	for i := range p.Ops {
		if p.Ops[i].Code == OpScan {
			p.Ops[i].IntParams[2]++
			break
		}
	}
	if ValidateProgramContract(p, &c) == nil {
		t.Fatal("accepted scan width drift")
	}
}

func TestExecutionContractPackingAndAliasing(t *testing.T) {
	c := fixtureContract(t, "ttt", NativeTTTStateful)
	for _, r := range c.Program.Representations {
		if r.Binding == "logits" {
			if !reflect.DeepEqual(r.Tensor.Packing, [][]string{{"batch", "time"}, {"vocab"}}) {
				t.Fatal("missing logical flattening")
			}
			bad := r.Tensor
			bad.Packing = [][]string{{"batch", "batch"}, {"vocab"}}
			if validateContractTensor(bad, "logits") == nil {
				t.Fatal("accepted noninvertible packing")
			}
		}
	}
	for bi := range c.Blocks {
		b := &c.Blocks[bi]
		if len(b.States) < 2 {
			continue
		}
		b.States[1].Input = b.States[0].Input
		b.States[1].Output = b.States[0].Output
		for pi := range c.Program.States {
			if c.Program.States[pi].ID == b.States[1].ID {
				c.Program.States[pi] = b.States[1]
			}
		}
	}
	if ValidateModelContract(c) == nil {
		t.Fatal("accepted mutable state alias")
	}
}

func TestExecutionContractClassificationAndUpdatePolicy(t *testing.T) {
	cfg := parseClassificationTestConfig(t, classificationTestConfig(BlockSpec{Type: "ttt_mlp", Heads: 2}))
	c, err := DescribeModelContract(cfg, NativeFull)
	if err != nil {
		t.Fatal(err)
	}
	if c.Coverage != "complete" {
		t.Fatal(c.Diagnostics)
	}
	if _, err := DescribeModelContract(cfg, NativeTTTStateful); err == nil {
		t.Fatal("classification accepted for causal stateful profile")
	}
	cfg = contractFixture(t, "ttt")
	before, err := DescribeModelContract(cfg, NativeTTTStateful)
	if err != nil {
		t.Fatal(err)
	}
	cfg.Blocks[0].InnerLRBase = 0.2
	after, err := DescribeModelContract(cfg, NativeTTTStateful)
	if err != nil {
		t.Fatal(err)
	}
	if before.SchemaDigest == after.SchemaDigest {
		t.Fatal("inner update policy absent from semantic digest")
	}
}
