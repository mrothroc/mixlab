package arch

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"strings"
	"testing"
)

func polarTestConfig(t *testing.T) *ArchConfig {
	t.Helper()
	raw, err := os.ReadFile("../examples/grid_polar_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := ParseArchConfig(raw, "polar")
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func TestGridPolarGraphContract(t *testing.T) {
	cfg := polarTestConfig(t)
	p, err := BuildGridIRProgram(cfg, true)
	if err != nil {
		t.Fatal(err)
	}
	checks := GridPolarInputChecks(p)
	if len(checks) != 2 {
		t.Fatal(checks)
	}
	metas, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(metas) != 6 {
		t.Fatal(metas)
	}
	for _, m := range metas {
		if m.Frozen {
			t.Fatal("live polar weights frozen", m)
		}
	}
	found := false
	for _, r := range p.Contract.Representations {
		if strings.HasSuffix(r.Binding, "_features") {
			found = true
			raw, _ := json.Marshal(r)
			if !strings.Contains(string(raw), "radius") {
				t.Fatal(string(raw))
			}
		}
	}
	if !found {
		t.Fatal("missing polar representation")
	}
	raw, _ := json.Marshal(cfg)
	cfg2, err := ParseArchConfig(raw, "roundtrip")
	if err != nil {
		t.Fatal(err)
	}
	p2, err := BuildGridIRProgram(cfg2, true)
	if err != nil || !reflect.DeepEqual(p, p2) {
		t.Fatal("roundtrip", err)
	}
	// Unselected polar branches are pruned, including their vector requirement.
	cfg.DenseRegression.Output = "network.base"
	cfg.Blocks[0].Ops = append(cfg.Blocks[0].Ops, OpSpec{Op: "slice", Inputs: []string{"x"}, Output: "base", Params: map[string]any{"start": 0, "end": 1, "step": 1, "axis": 3}})
	p, err = BuildGridIRProgram(cfg, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(GridPolarInputChecks(p)) != 0 || len(p.Inputs) != 1 {
		t.Fatal("unreachable polar inputs retained")
	}
}

func TestGridPolarValidation(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*ArchConfig)
	}{
		{"short", func(c *ArchConfig) { c.Blocks[0].Ops[0].Params["n_r"] = 11 }},
		{"rays", func(c *ArchConfig) { c.Blocks[0].Ops[6].Params["n_rays"] = 15 }},
		{"full_resolution", func(c *ArchConfig) { c.Blocks[0].Ops[6].Params["height"] = 7 }},
		{"no_vector", func(c *ArchConfig) { c.InputAdapter.VectorDim = 0 }},
		{"vector_alias", func(c *ArchConfig) { c.Blocks[0].Ops[0].Inputs[1] = "radial.bias" }},
		{"augmentation", func(c *ArchConfig) { c.Training.GridAugmentation = &GridAugmentationSpec{Dihedral: true} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := polarTestConfig(t)
			tc.mutate(c)
			if _, err := BuildGridIRProgram(c, true); err == nil {
				t.Fatal("accepted invalid polar graph")
			}
		})
	}
	for _, key := range []string{"n_rays", "n_r", "dr", "tx", "extra"} {
		for _, bad := range []any{nil, true, -1, 0, math.Inf(1), math.NaN(), []any{0, 0}, []any{0, 2}, []any{0}, "oops"} {
			c := polarTestConfig(t)
			c.Blocks[0].Ops[0].Params[key] = bad
			if _, err := BuildGridIRProgram(c, true); err == nil {
				t.Fatalf("accepted %s=%v", key, bad)
			}
		}
	}
}

func TestGridPolarTransitiveAncestry(t *testing.T) {
	// Every whitelisted op propagates dependencies; only an explicit stop ends them.
	for _, kind := range []string{"relu", "pad", "reshape", "slice", "transpose", "add", "sub", "mul", "concat", "max_pool2d", "group_norm", "conv2d", "conv_transpose2d"} {
		t.Run(kind, func(t *testing.T) {
			c := polarTestConfig(t)
			c.Blocks[0].Weights = append(c.Blocks[0].Weights, WeightSpec{Name: "source", Shape: []string{"B", "8", "8", "2"}})
			op := OpSpec{Op: kind, Inputs: []string{"source"}, Output: "ancestor"}
			switch kind {
			case "pad":
				op.Params = map[string]any{"axis1": []any{0, 0}, "axis2": []any{0, 0}, "mode": "edge"}
			case "reshape":
				op.Params = map[string]any{"shape": []any{"B", "8", "8", "2"}}
			case "slice":
				op.Params = map[string]any{"start": 0, "end": 8, "step": 1, "axis": 1}
			case "transpose":
				op.Params = map[string]any{"axes": []any{0, 1, 2, 3}}
			case "add", "sub", "mul":
				op.Inputs = []string{"x", "source"}
			case "concat":
				op.Inputs = []string{"x", "source"}
				op.Params = map[string]any{"axis": 3}
			case "max_pool2d":
				op.Params = map[string]any{"kernel": 1, "stride": 1}
			case "group_norm":
				op.Inputs = append(op.Inputs, "radial.bias", "angular.bias")
				op.Params = map[string]any{"groups": 1, "eps": 1e-5}
			case "conv2d":
				op.Inputs = append(op.Inputs, "radial.weight")
				op.Params = map[string]any{"kernel": []any{1, 3}, "padding": []any{0, 1}}
			case "conv_transpose2d":
				c.Blocks[0].Weights = append(c.Blocks[0].Weights, WeightSpec{Name: "transpose_weight", Shape: []string{"2", "1", "1", "2"}})
				op.Inputs = append(op.Inputs, "transpose_weight")
				op.Params = map[string]any{"kernel": 1}
			}
			c.Blocks[0].Ops[0].Inputs[0] = "ancestor"
			c.Blocks[0].Ops = append([]OpSpec{op}, c.Blocks[0].Ops...)
			_, err := BuildGridIRProgram(c, true)
			if err == nil || !strings.Contains(err.Error(), "trainable ancestor") {
				t.Fatalf("ancestry lost: %v", err)
			}
			// Freezing is not a bypass.
			c.Training.Freeze = []string{"*"}
			if _, err = BuildGridIRProgram(c, true); err == nil || !strings.Contains(err.Error(), "trainable ancestor") {
				t.Fatal(err)
			}
			c.Training.Freeze = nil
			c.Blocks[0].Ops[1].Inputs[0] = "stopped"
			c.Blocks[0].Ops = append(c.Blocks[0].Ops[:1], append([]OpSpec{{Op: "stop_gradient", Inputs: []string{"ancestor"}, Output: "stopped"}}, c.Blocks[0].Ops[1:]...)...)
			if kind == "concat" { // downstream branch expects two channels
				c.Blocks[0].Ops[2].Inputs[0] = "sliced"
				c.Blocks[0].Ops = append(c.Blocks[0].Ops[:2], append([]OpSpec{{Op: "slice", Inputs: []string{"stopped"}, Output: "sliced", Params: map[string]any{"start": 0, "end": 2, "step": 1, "axis": 3}}}, c.Blocks[0].Ops[2:]...)...)
			}
			if _, err = BuildGridIRProgram(c, true); err != nil {
				t.Fatalf("explicit stop failed: %v", err)
			}
		})
	}
}

func TestGridPolarBackmapAncestryAndSharedBranches(t *testing.T) {
	c := polarTestConfig(t)
	extra := OpSpec{Op: "cartesian_to_polar", Inputs: []string{"cartesian", "v"}, Output: "again", Params: map[string]any{"n_rays": 16, "n_r": 12, "dr": 1, "tx": []any{0, 1}}}
	c.Blocks[0].Ops = append(c.Blocks[0].Ops, extra)
	if _, err := BuildGridIRProgram(c, true); err == nil || !strings.Contains(err.Error(), "trainable ancestor") {
		t.Fatal(err)
	}
	// Stopping one shared branch must not hide the other branch's dependency.
	c.Blocks[0].Ops = c.Blocks[0].Ops[:len(c.Blocks[0].Ops)-1]
	c.Blocks[0].Ops = append(c.Blocks[0].Ops,
		OpSpec{Op: "stop_gradient", Inputs: []string{"cartesian"}, Output: "stopped_back"},
		OpSpec{Op: "add", Inputs: []string{"stopped_back", "cartesian"}, Output: "shared_back"})
	extra.Inputs[0] = "shared_back"
	c.Blocks[0].Ops = append(c.Blocks[0].Ops, extra)
	if _, err := BuildGridIRProgram(c, true); err == nil || !strings.Contains(err.Error(), "trainable ancestor") {
		t.Fatal(err)
	}
}

func TestGridPolarRecordValidation(t *testing.T) {
	c := PolarInputCheck{Height: 8, Width: 8, RowColumn: 1, ColColumn: 2}
	for _, v := range []float32{-1, 8, float32(math.NaN()), float32(math.Inf(1))} {
		err := c.Validate([]float32{9, 0, 0, 9, v, 1}, 3, []string{"good", "bad"})
		if err == nil || !strings.Contains(err.Error(), `record "bad"`) {
			t.Fatal(err)
		}
	}
	if err := c.Validate([]float32{9, 0, 7, 9, 2.5, 3.25}, 3, nil); err != nil {
		t.Fatal(err)
	}
}
