package arch

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"
)

func backoutTestConfig() *ArchConfig {
	return &ArchConfig{
		Name:      "backout_test",
		ModelDim:  32,
		VocabSize: 64,
		SeqLen:    8,
		MLPMult:   2,
		Blocks: []BlockSpec{
			{Type: "plain", Heads: 4},
			{Type: "swiglu"},
			{Type: "plain", Heads: 4},
			{Type: "swiglu"},
		},
		Training: TrainingSpec{Steps: 10, LR: 0.001, BatchTokens: 16},
	}
}

func TestParseArchConfig_BackoutRoundTrip(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"name": "backout",
		"model_dim": 32,
		"vocab_size": 64,
		"seq_len": 8,
		"backout": {"save_layer": 1, "lambda_init": -0.5},
		"blocks": [
			{"type": "plain", "heads": 4},
			{"type": "swiglu"},
			{"type": "plain", "heads": 4}
		],
		"training": {"steps": 10, "lr": 0.001, "batch_tokens": 16}
	}`), "backout")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if cfg.Backout == nil {
		t.Fatal("Backout = nil")
	}
	if cfg.Backout.SaveLayer != 1 {
		t.Fatalf("save_layer=%d want 1", cfg.Backout.SaveLayer)
	}
	if cfg.Backout.LambdaInit != -0.5 {
		t.Fatalf("lambda_init=%g want -0.5", cfg.Backout.LambdaInit)
	}
	out, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if !strings.Contains(string(out), `"backout"`) ||
		!strings.Contains(string(out), `"save_layer":1`) ||
		!strings.Contains(string(out), `"lambda_init":-0.5`) {
		t.Fatalf("round-trip JSON missing backout fields: %s", out)
	}
}

func TestParseArchConfig_BackoutDefaultsLambda(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{
		"name": "backout_default_lambda",
		"model_dim": 32,
		"vocab_size": 64,
		"seq_len": 8,
		"backout": {"save_layer": 1},
		"blocks": [
			{"type": "plain", "heads": 4},
			{"type": "swiglu"},
			{"type": "plain", "heads": 4}
		],
		"training": {"steps": 10, "lr": 0.001, "batch_tokens": 16}
	}`), "backout_default_lambda")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	if cfg.Backout.LambdaInit != defaultBackoutLambda {
		t.Fatalf("lambda_init=%g want %g", cfg.Backout.LambdaInit, float32(defaultBackoutLambda))
	}
}

func TestParseArchConfig_BackoutValidation(t *testing.T) {
	base := `{
		"name": "bad_backout",
		"model_dim": 32,
		"vocab_size": 64,
		"seq_len": 8,
		"blocks": [
			{"type": "plain", "heads": 4},
			{"type": "swiglu"},
			{"type": "plain", "heads": 4}
		],
		"training": {"steps": 10, "lr": 0.001, "batch_tokens": 16},
		%s
	}`
	tests := []struct {
		name    string
		backout string
		want    string
	}{
		{name: "missing_save_layer", backout: `"backout": {}`, want: "backout.save_layer"},
		{name: "negative_save_layer", backout: `"backout": {"save_layer": -1}`, want: "backout.save_layer=-1"},
		{name: "past_end", backout: `"backout": {"save_layer": 3}`, want: "backout.save_layer=3"},
		{name: "last_layer", backout: `"backout": {"save_layer": 2}`, want: "avoid a no-op"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseArchConfig([]byte(fmt.Sprintf(base, tt.backout)), tt.name)
			if err == nil {
				t.Fatal("ParseArchConfig succeeded, want error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error %q does not contain %q", err, tt.want)
			}
		})
	}
}

func TestValidateConfig_BackoutRejectsInfAndUNet(t *testing.T) {
	cfg := backoutTestConfig()
	cfg.Backout = &BackoutSpec{SaveLayer: 1, LambdaInit: float32(math.Inf(1)), saveLayerSet: true, lambdaInitSet: true}
	if _, err := validateConfig(cfg, "bad_inf"); err == nil || !strings.Contains(err.Error(), "lambda_init") {
		t.Fatalf("validateConfig inf error=%v, want lambda_init error", err)
	}

	cfg = backoutTestConfig()
	cfg.UNet = true
	cfg.Backout = &BackoutSpec{SaveLayer: 1, LambdaInit: -1, saveLayerSet: true, lambdaInitSet: true}
	if _, err := validateConfig(cfg, "bad_unet"); err == nil || !strings.Contains(err.Error(), "not supported with unet") {
		t.Fatalf("validateConfig unet error=%v, want unet error", err)
	}
}

func TestBuildIRProgram_BackoutOmittedMatchesBase(t *testing.T) {
	base := backoutTestConfig()
	withNil := backoutTestConfig()
	withNil.Backout = nil

	baseProg, err := BuildIRProgramFromConfig(base)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig(base): %v", err)
	}
	nilProg, err := BuildIRProgramFromConfig(withNil)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig(withNil): %v", err)
	}
	if baseProg.NumWeights != nilProg.NumWeights {
		t.Fatalf("NumWeights base=%d nil=%d", baseProg.NumWeights, nilProg.NumWeights)
	}
	if !reflect.DeepEqual(baseProg.Ops, nilProg.Ops) {
		t.Fatal("omitted backout changed emitted IR ops")
	}
}

func TestBuildIRProgram_BackoutEmitsCaptureAndSubtractBeforeFinalNorm(t *testing.T) {
	cfg := backoutTestConfig()
	cfg.Backout = &BackoutSpec{SaveLayer: 1, LambdaInit: -1.0}

	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	metas, err := CollectWeightShapesFromConfig(cfg)
	if err != nil {
		t.Fatalf("CollectWeightShapesFromConfig: %v", err)
	}
	if len(metas) != prog.NumWeights {
		t.Fatalf("metas=%d NumWeights=%d", len(metas), prog.NumWeights)
	}
	lambdaIdx := len(metas) - 1
	if metas[lambdaIdx].Name != backoutLambdaWeightName {
		t.Fatalf("last weight=%q want %q", metas[lambdaIdx].Name, backoutLambdaWeightName)
	}
	if got, want := metas[lambdaIdx].Shape, []int{1}; !reflect.DeepEqual(got, want) {
		t.Fatalf("backout_lambda shape=%v want %v", got, want)
	}
	if metas[lambdaIdx].InitValue != -1.0 {
		t.Fatalf("backout_lambda InitValue=%g want -1", metas[lambdaIdx].InitValue)
	}

	captureIdx := -1
	scaleIdx := -1
	subIdx := -1
	finalNormIdx := -1
	for i, op := range prog.Ops {
		switch {
		case op.Code == OpScalarMul && len(op.Outputs) == 1 && op.Outputs[0] == backoutTensorName:
			captureIdx = i
		case op.Code == OpMul && len(op.Inputs) == 2 && op.Inputs[0] == backoutTensorName && op.Inputs[1] == weightName(lambdaIdx):
			scaleIdx = i
		case op.Code == OpSub && len(op.Inputs) == 2 && op.Inputs[0] == "x" && op.Inputs[1] == backoutTensorName+"_scaled" && len(op.Outputs) == 1 && op.Outputs[0] == "x":
			subIdx = i
		case op.Code == OpRMSNorm && len(op.Outputs) == 1 && op.Outputs[0] == "x_final_norm":
			finalNormIdx = i
		}
	}
	if captureIdx < 0 {
		t.Fatal("missing x_backout capture")
	}
	if scaleIdx < 0 {
		t.Fatal("missing backout_lambda scale op")
	}
	if subIdx < 0 {
		t.Fatal("missing backout subtract op")
	}
	if finalNormIdx < 0 {
		t.Fatal("missing final RMSNorm")
	}
	if captureIdx >= scaleIdx || scaleIdx >= subIdx || subIdx >= finalNormIdx {
		t.Fatalf("bad backout op order capture=%d scale=%d sub=%d final_norm=%d", captureIdx, scaleIdx, subIdx, finalNormIdx)
	}
}

func TestBuildIRProgram_BackoutComposesWithParallelResidualRecurrenceAndMTP(t *testing.T) {
	t.Run("parallel_residual", func(t *testing.T) {
		cfg := backoutTestConfig()
		cfg.ParallelResidual = true
		cfg.Backout = &BackoutSpec{SaveLayer: 1, LambdaInit: -1}
		if _, err := BuildIRProgramFromConfig(cfg); err != nil {
			t.Fatalf("BuildIRProgramFromConfig parallel: %v", err)
		}
	})

	t.Run("recurrence", func(t *testing.T) {
		cfg := backoutTestConfig()
		cfg.Recurrence = []int{0, 1, 0, 3}
		cfg.Backout = &BackoutSpec{SaveLayer: 2, LambdaInit: -1}
		if _, err := BuildIRProgramFromConfig(cfg); err != nil {
			t.Fatalf("BuildIRProgramFromConfig recurrence: %v", err)
		}
	})

	t.Run("mtp", func(t *testing.T) {
		cfg := backoutTestConfig()
		cfg.MTP = &MTPSpec{N: 2}
		cfg.Backout = &BackoutSpec{SaveLayer: 1, LambdaInit: -1}
		if _, err := BuildIRProgramFromConfig(cfg); err != nil {
			t.Fatalf("BuildIRProgramFromConfig mtp: %v", err)
		}
	})
}
