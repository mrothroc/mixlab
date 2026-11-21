package arch

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestGridRectangularScalarParity(t *testing.T) {
	for _, d := range []int{1, 2, 4} {
		cfg := gridTestConfig(t)
		op := &cfg.Blocks[0].Ops[0]
		op.Params["dilation"], op.Params["padding"] = d, d
		before, err := BuildGridIRProgram(cfg, true)
		if err != nil {
			t.Fatal(err)
		}
		for key, value := range map[string]int{"kernel": 3, "stride": 1, "padding": d, "dilation": d} {
			op.Params[key] = []interface{}{value, value}
		}
		after, err := BuildGridIRProgram(cfg, true)
		if err != nil || !reflect.DeepEqual(before, after) {
			t.Fatal("scalar/pair changed program", err)
		}
	}
}

func TestGridRectangularAccountingAndRoundtrip(t *testing.T) {
	raw, err := os.ReadFile("../examples/grid_rectangular_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := ParseArchConfig(raw, "rect")
	if err != nil {
		t.Fatal(err)
	}
	w, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, weight := range w {
		if weight.PyTorchLinearFanIn != 18 {
			t.Fatal(weight)
		}
	}
	n, _, err := ParameterCountsFromConfig(cfg)
	if err != nil || n != 57 {
		t.Fatal(n, err)
	}
	if f := EstimateFLOPs(cfg).ForwardFLOPs; f != 14080 {
		t.Fatal(f)
	}
	p, err := BuildGridIRProgram(cfg, true)
	if err != nil {
		t.Fatal(err)
	}
	conv, pad := 0, 0
	for _, op := range p.Ops {
		if op.Code == OpConv2D {
			conv++
			if len(op.IntParams) != 8 {
				t.Fatal(op)
			}
		}
		if op.Code == OpSpatialPad {
			pad++
		}
	}
	if conv != 2 || pad != 2 {
		t.Fatal(conv, pad)
	}
	b, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	cfg2, err := ParseArchConfig(b, "roundtrip")
	if err != nil {
		t.Fatal(err)
	}
	p2, err := BuildGridIRProgram(cfg2, true)
	if err != nil || !reflect.DeepEqual(p, p2) {
		t.Fatal("roundtrip changed graph", err)
	}
	diagnostics, err := GridConvolutionDiagnostics(cfg)
	if err != nil || len(diagnostics) != 2 {
		t.Fatal(diagnostics, err)
	}
}

func TestGridRectangularShapesAndRejections(t *testing.T) {
	shapes := map[string][]int{"x": {2, 11, 21, 2}, "w": {3, 3, 5, 2}}
	op := OpSpec{Op: "conv2d", Inputs: []string{"x", "w"}, Params: map[string]interface{}{"kernel": []any{3, 5}, "stride": []any{2, 3}, "padding": []any{1, 4}, "dilation": []any{2, 1}}}
	s, err := gridOpShape(op, shapes, map[string]int{})
	if err != nil || !reflect.DeepEqual(s, []int{2, 5, 9, 3}) {
		t.Fatal(s, err)
	}
	for _, key := range []string{"kernel", "stride", "padding", "dilation"} {
		old := op.Params[key]
		for _, bad := range []any{nil, true, []any{}, []any{1}, []any{1, 2, 3}, []any{1, 1.5}, []any{1, -1}, []any{1, nil}, []any{1, math.Inf(1)}, []any{1, float64(math.MaxInt32) + 1}} {
			op.Params[key] = bad
			if _, err := gridOpShape(op, shapes, map[string]int{}); err == nil {
				t.Fatalf("accepted %s=%v", key, bad)
			}
		}
		op.Params[key] = old
		if key != "padding" {
			op.Params[key] = []any{1, 0}
			if _, err := convolutionParameters(op); err == nil {
				t.Fatalf("accepted zero %s", key)
			}
			op.Params[key] = old
		}
	}
	missing := OpSpec{Op: "conv2d", Params: map[string]interface{}{}}
	if _, err := convolutionParameters(missing); err == nil {
		t.Fatal("accepted missing kernel")
	}
	op.Params["unexpected"] = 1
	if _, err := gridOpShape(op, shapes, map[string]int{}); err == nil {
		t.Fatal("unknown field")
	}
	delete(op.Params, "unexpected")
	for _, kind := range []string{"conv_transpose2d", "max_pool2d"} {
		for _, key := range []string{"kernel", "stride", "padding"} {
			params := map[string]interface{}{"kernel": 1}
			params[key] = []any{1, 1}
			_, _, _, _, err := spatialOpParams(OpSpec{Op: kind, Params: params})
			if err == nil {
				t.Fatal("broadened", kind, key)
			}
		}
	}
}

func TestGridSpatialPadValidation(t *testing.T) {
	shapes := map[string][]int{"x": {2, 3, 4, 2}}
	for _, mode := range []string{"zero", "edge", "circular"} {
		op := OpSpec{Op: "pad", Inputs: []string{"x"}, Params: map[string]interface{}{"axis1": []any{3, 3}, "axis2": []any{4, 4}, "mode": mode}}
		s, err := gridOpShape(op, shapes, nil)
		if err != nil || !reflect.DeepEqual(s, []int{2, 9, 12, 2}) {
			t.Fatal(s, err)
		}
		for _, bad := range []any{nil, 1, []any{1}, []any{1, 2, 3}, []any{-1, 0}, []any{1.2, 0}, []any{math.MaxInt32, 0}} {
			op.Params["axis1"] = bad
			if _, err := gridOpShape(op, shapes, nil); err == nil {
				t.Fatal("accepted pad", bad)
			}
		}
	}
	for _, raw := range []string{`{"axis1":[0,0],"axis2":[0,0]}`, `{"axis1":[0,0],"axis2":[0,0],"mode":"reflect"}`, `{"axis1":[0,0],"axis2":[0,0],"mode":"zero","extra":1}`} {
		var params map[string]interface{}
		_ = json.Unmarshal([]byte(raw), &params)
		if _, err := spatialPadParams(params); err == nil {
			t.Fatal(raw)
		}
	}
	_, err := spatialPadShape(OpSpec{Params: map[string]interface{}{"axis1": []any{4, 0}, "axis2": []any{0, 0}, "mode": "circular"}}, [][]int{shapes["x"]})
	if err == nil || !strings.Contains(err.Error(), "input axis") {
		t.Fatal(err)
	}
	valid := OpSpec{Params: map[string]interface{}{"axis1": []any{0, 0}, "axis2": []any{0, 0}, "mode": "zero"}}
	for _, inputs := range [][][]int{nil, {{1, 2, 3}}, {{1, 2, 3, 4}, {1, 2, 3, 4}}} {
		if _, err := spatialPadShape(valid, inputs); err == nil {
			t.Fatal("accepted invalid pad inputs", inputs)
		}
	}
	valid.Params["axis1"] = []any{1, 0}
	if _, err := spatialPadShape(valid, [][]int{{1, 1, 1, math.MaxInt32}}); err == nil {
		t.Fatal("accepted output product overflow")
	}
}
