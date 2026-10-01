package arch

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func TestGridPinnedReferenceCounts(t *testing.T) {
	for _, tt := range []struct {
		name          string
		total, active int64
	}{{"two_channel", 13274031, 10921405}, {"three_channel", 13275315, 10921984}} {
		b, err := os.ReadFile("testdata/grid_reference/" + tt.name + ".json")
		if err != nil {
			t.Fatal(err)
		}
		cfg, err := ParseArchConfig(b, tt.name)
		if err != nil {
			t.Fatal(err)
		}
		n, _, err := ParameterCountsFromConfig(cfg)
		if err != nil || n != tt.total {
			t.Fatalf("count %d %v", n, err)
		}
		metas, err := CollectWeightShapesFromConfig(cfg)
		if err != nil {
			t.Fatal(err)
		}
		var active int64
		for _, m := range metas {
			if !m.Frozen {
				n := int64(1)
				for _, d := range m.Shape {
					n *= int64(d)
				}
				active += n
			}
		}
		if active != tt.active {
			t.Fatal(active)
		}
		if EstimateFLOPs(cfg).ForwardFLOPs <= 0 {
			t.Fatal("missing spatial FLOPs")
		}
	}
}

func gridTestConfig(t *testing.T) *ArchConfig {
	t.Helper()
	b, err := os.ReadFile("../examples/grid_regression_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := ParseArchConfig(b, "grid")
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func TestGridConfigGraphAndWeights(t *testing.T) {
	cfg := gridTestConfig(t)
	if cfg.SeqLen != 0 || cfg.ModelDim != 0 || cfg.VocabSize != 0 || cfg.Training.BatchTokens != 0 {
		t.Fatal("grid synthesized token dimensions")
	}
	p, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if p.NumWeights != 2 || len(p.Inputs) != 3 {
		t.Fatalf("unexpected grid IR %+v", p)
	}
	w, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if w[0].PyTorchLinearFanIn != 18 || w[1].PyTorchLinearFanIn != 18 || w[0].Name != "network.conv.weight" {
		t.Fatalf("bad init metadata %+v", w)
	}
	n, _, err := ParameterCountsFromConfig(cfg)
	if err != nil || n != 19 {
		t.Fatalf("count=%d err=%v", n, err)
	}
	p, err = BuildGridIRProgram(cfg, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Inputs) != 1 || len(p.Outputs) != 1 {
		t.Fatal("prediction contains task inputs")
	}
}

func TestGridValidation(t *testing.T) {
	base, err := os.ReadFile("../examples/grid_regression_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ old, new, want string }{
		{`"batch_size": 2`, `"batch_size": 0`, "batch_size"},
		{`"batch_size": 2`, `"batch_size": 2, "batch_tokens": 128`, "batch_tokens"},
		{`"target_channels": 1`, `"target_channels": 1, "metric_scale": 0`, "metric_scale"},
		{`"optimizer": "adamw"`, `"optimizer": "muon"`, "optimizer"},
		{`"optimizer": "adamw"`, `"optimizer": "adamw", "seq_len_schedule": [[0,8]]`, "SeqLenSchedule"},
		{`"name": "grid_regression_tiny"`, `"name": "grid_regression_tiny", "model_dim": 8`, "ModelDim"},
		{`"name": "grid_regression_tiny"`, `"name": "grid_regression_tiny", "data": {"no_shard_shuffle":true}`, "Data"},
		{`"kernel": 3`, `"kernel": 3, "groups":2`, "unsupported"},
		{`"kernel": 3`, `"kernel": 3, "output_padding":1`, "output_padding"},
		{`"kind": "pytorch_conv_uniform"`, `"kind": "normal", "scale":0`, "positive"},
		{`"kernel": 3`, `"kernel": 3.5`, "integer"},
		{`"padding": 1`, `"padding": 0`, "shape"},
		{`"conv.bias"], "output"`, `"missing"], "output"`, "unknown input"},
		{`"output": "network.prediction"`, `"output": "network.missing"`, "output"},
	} {
		t.Run(tc.want, func(t *testing.T) {
			_, err := ParseArchConfig([]byte(strings.Replace(string(base), tc.old, tc.new, 1)), "bad")
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v want %s", err, tc.want)
			}
		})
	}
}

func TestGridLayoutValidation(t *testing.T) {
	for _, axes := range [][]interface{}{{0., 2., 1., 3.}, {0., 2.5, 1., 3.}, {0., 1., 1., 3.}} {
		cfg := gridTestConfig(t)
		cfg.Blocks[0].Ops = append(cfg.Blocks[0].Ops, OpSpec{Op: "transpose", Inputs: []string{"prediction"}, Output: "permuted", Params: map[string]interface{}{"axes": axes}})
		cfg.DenseRegression.Output = "network.permuted"
		_, err := BuildGridIRProgram(cfg, false)
		valid := axes[1] == 2.
		if (err == nil) != valid {
			t.Fatalf("axes %v: %v", axes, err)
		}
	}
}

func TestGridConfigRoundTrip(t *testing.T) {
	cfg := gridTestConfig(t)
	b, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = ParseArchConfig(b, "roundtrip"); err != nil {
		t.Fatal(err)
	}
}

func TestGridSelectedOutputPrunesUnusedStage(t *testing.T) {
	cfg := gridTestConfig(t)
	s := &cfg.Blocks[0]
	s.Weights = append(s.Weights, WeightSpec{Name: "unused", Shape: []string{"1", "1", "1", "1"}})
	s.Ops = append(s.Ops, OpSpec{Op: "conv2d", Inputs: []string{"prediction", "unused"}, Output: "second", Params: map[string]interface{}{"kernel": 1}})
	p, w, err := buildGridGraph(cfg, true)
	if err != nil {
		t.Fatal(err)
	}
	if !w[2].Frozen || p.NumWeights != 3 {
		t.Fatal("unreachable weight not retained/frozen")
	}
	for _, op := range p.Ops {
		for _, in := range op.Inputs {
			if in == "w2" {
				t.Fatal("unused stage executed")
			}
		}
	}
}

func TestGridOutputMustSelectActivation(t *testing.T) {
	cfg := gridTestConfig(t)
	cfg.Blocks[0].Weights = append(cfg.Blocks[0].Weights, WeightSpec{Name: "not_an_activation", Shape: []string{"2", "8", "8", "1"}})
	cfg.DenseRegression.Output = "network.not_an_activation"
	if _, err := BuildGridIRProgram(cfg, false); err == nil {
		t.Fatal("accepted a parameter as a named activation output")
	}
}
