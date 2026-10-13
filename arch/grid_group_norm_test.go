package arch

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"testing"
)

func TestGridGroupNormConfigAndWeights(t *testing.T) {
	b, err := os.ReadFile("../examples/grid_group_norm_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := ParseArchConfig(b, "group_norm")
	if err != nil {
		t.Fatal(err)
	}
	p, w, err := buildGridGraph(cfg, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(w) != 6 || !w[2].InitOne || !w[3].InitZero || w[2].Frozen || w[3].Frozen {
		t.Fatalf("affine metadata: %+v", w)
	}
	if n, _, err := ParameterCountsFromConfig(cfg); err != nil || n != 89 {
		t.Fatalf("count %d: %v", n, err)
	}
	if got := EstimateFLOPs(cfg).ForwardFLOPs; got != 8*8*4*36+8*8*8+8*8*4+8*8*4*9+4 {
		t.Fatalf("FLOPs=%d", got)
	}
	var found bool
	for _, op := range p.Ops {
		if op.Code == OpGroupNorm {
			found = true
			if !reflect.DeepEqual(op.IntParams, []int{2}) || !reflect.DeepEqual(op.FloatParams, []float32{1e-5}) || op.Inputs[1] != "w2" || op.Inputs[2] != "w3" {
				t.Fatalf("bad lowering %+v", op)
			}
		}
	}
	if !found {
		t.Fatal("missing GroupNorm")
	}
	cfg.Blocks[0].Ops[1].Params["eps"] = 1e-5
	explicit, err := BuildGridIRProgram(cfg, true)
	if err != nil || !reflect.DeepEqual(p.Ops, explicit.Ops) {
		t.Fatal("explicit default epsilon changed IR", err)
	}
	encoded, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	round, err := ParseArchConfig(encoded, "roundtrip")
	if err != nil {
		t.Fatal(err)
	}
	p2, err := BuildGridIRProgram(round, true)
	if err != nil || !reflect.DeepEqual(p.Ops, p2.Ops) {
		t.Fatal("roundtrip changed IR", err)
	}
	cfg.DenseRegression.Output = "network.hidden"
	cfg.DenseRegression.TargetChannels = 4
	_, w, err = buildGridGraph(cfg, false)
	if err != nil || !w[2].Frozen || !w[3].Frozen {
		t.Fatal("unused affine must be frozen", err)
	}
}

func TestGridGroupNormValidation(t *testing.T) {
	shapes := map[string][]int{"x": {1, 3, 5, 4}, "scale": {4}, "bias": {4}}
	for _, groups := range []int{1, 2, 4} {
		op := OpSpec{Op: "group_norm", Inputs: []string{"x", "scale", "bias"}, Params: map[string]interface{}{"groups": groups}}
		s, err := gridOpShape(op, shapes, nil)
		if err != nil || !reflect.DeepEqual(s, shapes["x"]) {
			t.Fatal(s, err)
		}
	}
	for _, params := range []map[string]interface{}{
		nil, {"groups": 0}, {"groups": -1}, {"groups": 1.5}, {"groups": 3}, {"groups": 8},
		{"groups": 2, "eps": 0}, {"groups": 2, "eps": -1}, {"groups": 2, "eps": math.NaN()},
		{"groups": 2, "eps": math.Inf(1)}, {"groups": 2, "eps": 1e-50}, {"groups": 2, "eps": 1e50},
		{"groups": 2, "axis": 3},
	} {
		op := OpSpec{Op: "group_norm", Inputs: []string{"x", "scale", "bias"}, Params: params}
		if _, err := gridOpShape(op, shapes, nil); err == nil {
			t.Fatalf("accepted params %v", params)
		}
	}
	op := OpSpec{Op: "group_norm", Params: map[string]interface{}{"groups": 2}}
	for _, inputs := range [][][]int{
		{{1, 3, 5, 4}}, {{1, 3, 4}, {4}, {4}}, {{1, 3, 5, 4}, {1, 4}, {4}}, {{1, 3, 5, 4}, {4}, {2}},
	} {
		if _, err := groupNormShape(op, inputs); err == nil {
			t.Fatalf("accepted shapes %v", inputs)
		}
	}
}
