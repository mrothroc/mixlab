package arch

import (
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestGridVectorsConfigAndIR(t *testing.T) {
	raw, err := os.ReadFile("../examples/grid_vectors_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := ParseArchConfig(raw, "vectors")
	if err != nil {
		t.Fatal(err)
	}
	for _, supervised := range []bool{false, true} {
		p, err := BuildGridIRProgram(cfg, supervised)
		if err != nil {
			t.Fatal(err)
		}
		found := false
		for _, in := range p.Inputs {
			if in.Name == "grid_vectors" {
				found = true
				if !reflect.DeepEqual(in.Shape, []int{2, 1}) || in.DType != TensorFloat32 {
					t.Fatal(in)
				}
			}
		}
		if !found {
			t.Fatal("missing vector input")
		}
		contractFound := false
		for _, representation := range p.Contract.Representations {
			if representation.Binding == "grid_vectors" {
				contractFound = true
				if representation.Meaning != "per_record_vector_unnormalized_unaugmented" || !reflect.DeepEqual(representation.Tensor.Axes, []ContractAxis{{Name: "batch", Size: 2}, {Name: "feature", Size: 1}}) {
					t.Fatal(representation)
				}
			}
		}
		if !contractFound {
			t.Fatal("missing vector execution contract")
		}
		if len(p.Ops) == 0 || p.Ops[0].Inputs[0] != "grid_vectors" {
			t.Fatal("v not bound", p.Ops)
		}
	}
	n, _, err := ParameterCountsFromConfig(cfg)
	if err != nil || n != 2 {
		t.Fatal(n, err)
	}
	if EstimateFLOPs(cfg).ForwardFLOPs <= 0 {
		t.Fatal("missing FLOPs")
	}
	encoded, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	again, err := ParseArchConfig(encoded, "roundtrip")
	if err != nil || again.InputAdapter.VectorDim != 1 {
		t.Fatal(err)
	}
	for _, value := range []string{"-1", "1.5", "null", "true", `"1"`, "2147483648"} {
		t.Run(value, func(t *testing.T) {
			_, err := ParseArchConfig([]byte(strings.Replace(string(raw), `"vector_dim": 1`, `"vector_dim": `+value, 1)), "invalid")
			if err == nil {
				t.Fatal("accepted invalid vector_dim")
			}
		})
	}
	if _, err := ParseArchConfig([]byte(strings.Replace(string(raw), `"vector_dim": 1`, `"vector_dim": 1, "unknown_field": 7`, 1)), "unknown"); err == nil {
		t.Fatal("input adapter decoding stopped rejecting unknown fields")
	}
	if err := validateInputAdapter(&ArchConfig{InputAdapter: &InputAdapterSpec{Kind: InputAdapterTokenEmbedding, VectorDim: 1}}, "non-grid"); err == nil {
		t.Fatal("accepted vector_dim outside grid adapter")
	}
}

func TestGridVectorsDisabledAndPruned(t *testing.T) {
	cfg := gridTestConfig(t)
	before, err := BuildGridIRProgram(cfg, true)
	if err != nil {
		t.Fatal(err)
	}
	cfg.InputAdapter.VectorDim = 0
	after, err := BuildGridIRProgram(cfg, true)
	if err != nil || !reflect.DeepEqual(before, after) {
		t.Fatal("zero changed graph", err)
	}
	cfg.InputAdapter.VectorDim = 3
	after, err = BuildGridIRProgram(cfg, true)
	if err != nil {
		t.Fatal(err)
	}
	for _, in := range after.Inputs {
		if in.Name == "grid_vectors" {
			t.Fatal("unused vector input not pruned")
		}
	}
	cfg.Blocks[0].Ops[0].Inputs[0] = "v"
	cfg.InputAdapter.VectorDim = 0
	if _, err = BuildGridIRProgram(cfg, true); err == nil {
		t.Fatal("v enabled without vector_dim")
	}
}

func TestGridVectorAliasReservedOnlyWhenEnabled(t *testing.T) {
	cfg := gridTestConfig(t)
	cfg.Blocks[0].Weights = append(cfg.Blocks[0].Weights, WeightSpec{Name: "v", Shape: []string{"1"}})
	if _, err := BuildGridIRProgram(cfg, false); err != nil {
		t.Fatal("existing weight name rejected", err)
	}
	cfg.InputAdapter.VectorDim = 1
	if _, err := BuildGridIRProgram(cfg, false); err == nil {
		t.Fatal("v cannot name both a weight and an input")
	}
}

func TestGridVectorReshapeValidation(t *testing.T) {
	for _, tc := range []struct {
		shape []interface{}
		input []int
		valid bool
	}{
		{[]interface{}{"B", 2., 1., 1.}, []int{3, 2}, true},
		{[]interface{}{3., 2.}, []int{3, 1, 2, 1}, true},
		{[]interface{}{2., 3.}, []int{3, 2}, false},
		{[]interface{}{"B", 2.}, []int{2, 3}, false},
		{[]interface{}{"B", -1.}, []int{3, 2}, false},
		{[]interface{}{"B", 2.5}, []int{3, 2}, false},
		{[]interface{}{"B", "D"}, []int{3, 2}, false},
		{[]interface{}{"B", 3.}, []int{3, 2}, false},
	} {
		_, err := gridReshape(OpSpec{Params: map[string]interface{}{"shape": tc.shape}}, [][]int{tc.input}, 3)
		if (err == nil) != tc.valid {
			t.Fatalf("shape=%v input=%v: %v", tc.shape, tc.input, err)
		}
	}
}
