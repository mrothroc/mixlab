package arch

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestGridDilationDefaultsAndAccounting(t *testing.T) {
	base := gridTestConfig(t)
	baseProgram, err := BuildGridIRProgram(base, true)
	if err != nil {
		t.Fatal(err)
	}
	baseWeights, err := CollectWeightShapesFromConfig(base)
	if err != nil {
		t.Fatal(err)
	}
	baseFLOPs := EstimateFLOPs(base)
	for _, d := range []int{1, 2, 4} {
		cfg := gridTestConfig(t)
		cfg.Blocks[0].Ops[0].Params["dilation"] = d
		cfg.Blocks[0].Ops[0].Params["padding"] = d
		p, err := BuildGridIRProgram(cfg, true)
		if err != nil {
			t.Fatal(err)
		}
		if d == 1 && !reflect.DeepEqual(p, baseProgram) {
			t.Fatal("explicit dilation=1 changed the default IR")
		}
		found := false
		for _, op := range p.Ops {
			if op.Code == OpConv2D {
				want := []int{3, 1, d}
				if d != 1 {
					want = append(want, d)
				}
				if !reflect.DeepEqual(op.IntParams, want) {
					t.Fatal(op.IntParams, want)
				}
				found = true
			}
		}
		if !found {
			t.Fatal("missing convolution")
		}
		weights, err := CollectWeightShapesFromConfig(cfg)
		if err != nil || !reflect.DeepEqual(weights, baseWeights) || !reflect.DeepEqual(EstimateFLOPs(cfg), baseFLOPs) {
			t.Fatal("dilation changed same-output FLOPs, weights or initialization", err)
		}
		raw, err := json.Marshal(cfg)
		if err != nil {
			t.Fatal(err)
		}
		round, err := ParseArchConfig(raw, "dilation")
		if err != nil {
			t.Fatal(err)
		}
		rp, err := BuildGridIRProgram(round, true)
		if err != nil || !reflect.DeepEqual(rp.Ops, p.Ops) {
			t.Fatal("dilation round trip changed IR", err)
		}
	}
}

func TestGridDilationShapesAndValidation(t *testing.T) {
	shapes := map[string][]int{"x": {2, 11, 13, 2}, "w": {3, 3, 3, 2}}
	op := OpSpec{Op: "conv2d", Inputs: []string{"x", "w"}, Params: map[string]interface{}{"kernel": 3, "stride": 2, "padding": 1, "dilation": 2}}
	got, err := gridOpShape(op, shapes, map[string]int{})
	if err != nil || !reflect.DeepEqual(got, []int{2, 5, 6, 3}) {
		t.Fatal(got, err)
	}
	for _, d := range []any{0, -1, 1.5, nil, true, "bad", math.NaN(), math.Inf(1), float64(math.MaxInt32) + 1, math.MaxInt32} {
		op.Params["dilation"] = d
		if _, err := gridOpShape(op, shapes, map[string]int{}); err == nil {
			t.Fatalf("accepted dilation %v", d)
		}
	}
	op.Params["dilation"] = 8
	if _, err := gridOpShape(op, shapes, map[string]int{}); err == nil || !strings.Contains(err.Error(), "effective kernel") {
		t.Fatal(err)
	}
	op.Params["dilation"] = 2
	for _, kind := range []string{"conv_transpose2d", "max_pool2d"} {
		op.Op = kind
		if _, _, _, _, err := spatialOpParams(op); err == nil {
			t.Fatal("accepted dilation on", kind)
		}
	}
	if _, err := gridConvOutputSize(8, 3, 1, math.MaxInt32, 2); err == nil {
		t.Fatal("accepted padded dimension overflow")
	}
}

func TestGridASPPExample(t *testing.T) {
	raw, err := os.ReadFile("../examples/grid_aspp_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := ParseArchConfig(raw, "aspp")
	if err != nil {
		t.Fatal(err)
	}
	p, err := BuildGridIRProgram(cfg, true)
	if err != nil {
		t.Fatal(err)
	}
	conv, concat := 0, 0
	for _, op := range p.Ops {
		if op.Code == OpConv2D {
			conv++
		}
		if op.Code == OpConcat {
			concat++
		}
	}
	if conv != 5 || concat != 3 || p.NumWeights != 10 {
		t.Fatal("incomplete ASPP graph", conv, concat, p.NumWeights)
	}
}
