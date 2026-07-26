//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"os"
	"testing"
)

func TestContinuousLinearProjectorCPUMLXParity(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	cfg, err := ParseArchConfig([]byte(`{
		"name":"continuous_projector_parity",
		"model_dim":2,
		"seq_len":2,
		"positional_embedding":"none",
		"input_adapter":{"kind":"linear_frames","feature_dim":2,"bias":true,"norm":"none"},
		"blocks":[{"type":"swiglu"}],
		"training":{
			"objective":"classification",
			"classification":{"num_labels":2,"pooling":"mean","classifier_dropout":0},
			"optimizer":"adamw",
			"batch_tokens":2,
			"steps":1,
			"lr":0.001,
			"weight_decay":0
		}
	}`), "continuous_projector_parity")
	if err != nil {
		t.Fatal(err)
	}
	prog, err := BuildEvalIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	projection := []float32{1, -2, 0.5, 3}
	bias := []float32{0.25, -0.5}
	weights := make([][]float32, len(shapes))
	for i, shape := range shapes {
		weights[i] = make([]float32, shapeProduct(shape.Shape))
		if shape.IsNormScale || shape.InitOne {
			for j := range weights[i] {
				weights[i][j] = 1
			}
		}
		switch shape.Name {
		case "input_adapter_proj":
			copy(weights[i], projection)
		case "input_adapter_bias":
			copy(weights[i], bias)
		}
	}
	trainerInterface, err := initGPUTrainer(prog, cfg, weights, nil)
	if err != nil {
		t.Fatal(err)
	}
	trainer := trainerInterface.(*mlxGPUTrainer)
	defer trainer.CloseTrainer()

	frames := []float32{2, -1, -0.5, 4}
	batch := objectiveBatch{
		frames:               frames,
		classificationLabels: []int32{0},
		classificationMask:   []float32{1, 1},
	}
	if _, err := trainer.EvaluateObjectiveGPUWithOutputs(batch, 1, 2, []string{"x_hidden"}); err != nil {
		t.Fatal(err)
	}
	got, err := readTrainerOutput(trainer, "x_hidden", []int{1, 2, 2})
	if err != nil {
		t.Fatal(err)
	}
	want := make([]float32, 0, len(got))
	for row := 0; row < 2; row++ {
		x0, x1 := frames[row*2], frames[row*2+1]
		projected := []float64{
			float64(x0*projection[0] + x1*projection[2] + bias[0]),
			float64(x0*projection[1] + x1*projection[3] + bias[1]),
		}
		denom := math.Sqrt((projected[0]*projected[0]+projected[1]*projected[1])/2 + 1e-5)
		want = append(want, float32(projected[0]/denom), float32(projected[1]/denom))
	}
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-5 {
			t.Fatalf("x_hidden[%d]=%g want=%g diff=%g; all got=%v want=%v", i, got[i], want[i], diff, got, want)
		}
	}
}

func TestContinuousClassificationTinyTrainingSmoke(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	cfg, err := ParseArchConfig([]byte(`{
		"name":"continuous_training_smoke",
		"model_dim":8,
		"seq_len":16,
		"positional_embedding":"none",
		"input_adapter":{"kind":"linear_frames","feature_dim":1,"bias":true,"norm":"layernorm"},
		"blocks":[{"type":"swiglu"}],
		"training":{
			"objective":"classification",
			"classification":{"num_labels":2,"pooling":"mean","classifier_dropout":0},
			"optimizer":"adamw",
			"batch_tokens":32,
			"steps":30,
			"lr":0.01,
			"grad_clip":1,
			"weight_decay":0,
			"seed":17
		}
	}`), "continuous_training_smoke")
	if err != nil {
		t.Fatal(err)
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	trainerInterface, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	trainer := trainerInterface.(*mlxGPUTrainer)
	defer trainer.CloseTrainer()

	frames := make([]float32, 32)
	for pos := 0; pos < 16; pos++ {
		noise := float32((pos%5)-2) * 0.02
		frames[pos] = -1 + noise
		frames[16+pos] = 1 + noise
	}
	batch := objectiveBatch{
		frames: frames, classificationLabels: []int32{0, 1},
		classificationMask: make([]float32, 32),
	}
	for i := range batch.classificationMask {
		batch.classificationMask[i] = 1
	}
	first, err := trainer.EvaluateObjectiveGPU(batch, 2, 16)
	if err != nil {
		t.Fatal(err)
	}
	for step := 0; step < 30; step++ {
		loss, err := trainer.TrainObjectiveStepGPU(batch, 2, 16, float32(cfg.Training.LR))
		if err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			t.Fatalf("step %d non-finite loss=%g", step, loss)
		}
	}
	last, err := trainer.EvaluateObjectiveGPU(batch, 2, 16)
	if err != nil {
		t.Fatal(err)
	}
	if !(last < first) {
		t.Fatalf("continuous classification loss did not decrease: first=%g last=%g", first, last)
	}
}

func TestContinuousMamba3CanonicalLength16000Smoke(t *testing.T) {
	if os.Getenv("MIXLAB_CONTINUOUS_16K_TEST") != "1" {
		t.Skip("set MIXLAB_CONTINUOUS_16K_TEST=1 to run the raw-waveform shape smoke")
	}
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	cfg, err := ParseArchConfig([]byte(`{
		"name":"continuous_16k_smoke",
		"model_dim":8,
		"seq_len":16000,
		"positional_embedding":"none",
		"input_adapter":{"kind":"linear_frames","feature_dim":1,"bias":true,"norm":"layernorm"},
		"blocks":[
			{"type":"mamba3-canonical","inner_dim":8,"state_size":4,"n_groups":2,"dt_rank":2,"scan_chunk_size":64},
			{"type":"swiglu"}
		],
		"training":{
			"objective":"classification",
			"classification":{"num_labels":2,"pooling":"last","classifier_dropout":0},
			"optimizer":"adamw",
			"batch_tokens":16000,
			"steps":1,
			"lr":0.0001,
			"grad_clip":1,
			"weight_decay":0
		}
	}`), "continuous_16k_smoke")
	if err != nil {
		t.Fatal(err)
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	trainerInterface, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	trainer := trainerInterface.(*mlxGPUTrainer)
	defer trainer.CloseTrainer()
	batch := objectiveBatch{
		frames: make([]float32, 16000), classificationLabels: []int32{0},
		classificationPos: []int32{15999},
	}
	loss, err := trainer.TrainObjectiveStepGPU(batch, 1, 16000, float32(cfg.Training.LR))
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatalf("non-finite 16k training loss=%g", loss)
	}
}
