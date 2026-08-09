//go:build mlx && cgo && (darwin || linux)

package train

import (
	"math"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestS4DTinyDiscreteClassificationTraining(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	cfg, err := ParseArchConfig([]byte(`{
		"name":"s4d_discrete_smoke",
		"model_dim":8,
		"vocab_size":8,
		"seq_len":8,
		"positional_embedding":"none",
		"blocks":[{"type":"s4d","state_size":8},{"type":"swiglu"}],
		"training":{
			"objective":"classification",
			"classification":{"num_labels":2,"pooling":"last","classifier_dropout":0},
			"optimizer":"adamw",
			"batch_tokens":16,
			"steps":40,
			"lr":0.003,
			"scalar_lr":0.001,
			"grad_clip":1,
			"weight_decay":0,
			"seed":19
		}
	}`), "s4d_discrete_smoke")
	if err != nil {
		t.Fatal(err)
	}
	rawBatch := trainBatch{
		x: []int{
			1, 2, 2, 2, 2, 2, 2, 2,
			1, 6, 6, 6, 6, 6, 6, 6,
		},
		y:          make([]int, 16),
		labels:     []int32{0, 1},
		validMask:  repeatFloat32Train(16, 1),
		segmentIDs: make([]int32, 16),
	}
	batch, err := prepareObjectiveBatch(cfg, rawBatch, 0, arch.ObjectiveClassification)
	if err != nil {
		t.Fatal(err)
	}
	assertS4DTrainingDecreases(t, cfg, batch, 2, 8)
}

func TestS4DTinyContinuousClassificationTraining(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	cfg, err := ParseArchConfig([]byte(`{
		"name":"s4d_continuous_smoke",
		"model_dim":8,
		"seq_len":16,
		"positional_embedding":"none",
		"input_adapter":{"kind":"linear_frames","feature_dim":1,"bias":true,"norm":"layernorm"},
		"blocks":[{"type":"s4d","state_size":8},{"type":"swiglu"}],
		"training":{
			"objective":"classification",
			"classification":{"num_labels":2,"pooling":"last","classifier_dropout":0},
			"optimizer":"adamw",
			"batch_tokens":32,
			"steps":40,
			"lr":0.003,
			"scalar_lr":0.001,
			"grad_clip":1,
			"weight_decay":0,
			"seed":23
		}
	}`), "s4d_continuous_smoke")
	if err != nil {
		t.Fatal(err)
	}
	frames := make([]float32, 32)
	for pos := 0; pos < 16; pos++ {
		noise := float32((pos%5)-2) * 0.02
		frames[pos] = -1 + noise
		frames[16+pos] = 1 + noise
	}
	batch := objectiveBatch{
		frames:               frames,
		classificationLabels: []int32{0, 1},
		classificationMask:   repeatFloat32Train(32, 1),
		classificationPos:    []int32{15, 15},
	}
	assertS4DTrainingDecreases(t, cfg, batch, 2, 16)
}

func TestS4DSobolevFrequencyFilterTinyTraining(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	cfg, err := ParseArchConfig([]byte(`{
		"name":"s4d_sobolev_smoke",
		"model_dim":8,
		"seq_len":8,
		"positional_embedding":"none",
		"input_adapter":{"kind":"linear_frames","feature_dim":1,"bias":true,"norm":"none"},
		"blocks":[
			{"type":"s4d","state_size":8,"freq_scale":3,"sobolev_filter":{"beta_init":0,"learning_rate":0.01}},
			{"type":"swiglu"}
		],
		"training":{
			"objective":"classification",
			"classification":{"num_labels":2,"pooling":"last","classifier_dropout":0},
			"optimizer":"adamw",
			"batch_tokens":16,
			"steps":20,
			"lr":0.003,
			"scalar_lr":0.001,
			"grad_clip":1,
			"weight_decay":0,
			"seed":43
		}
	}`), "s4d_sobolev_smoke")
	if err != nil {
		t.Fatal(err)
	}
	frames := make([]float32, 16)
	for pos := 0; pos < 8; pos++ {
		noise := float32((pos%3)-1) * 0.02
		frames[pos] = -1 + noise
		frames[8+pos] = 1 + noise
	}
	batch := objectiveBatch{
		frames:               frames,
		classificationLabels: []int32{0, 1},
		classificationMask:   repeatFloat32Train(16, 1),
		classificationPos:    []int32{7, 7},
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
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	betaIndex := -1
	for i, shape := range shapes {
		if shape.Name == "s4d_sobolev_beta" {
			betaIndex = i
			break
		}
	}
	if betaIndex < 0 {
		t.Fatal("S4D Sobolev beta weight not found")
	}
	before, err := trainer.ReadWeights()
	if err != nil {
		t.Fatal(err)
	}
	first, err := trainer.EvaluateObjectiveGPU(batch, 2, 8)
	if err != nil {
		t.Fatal(err)
	}
	for step := 0; step < cfg.Training.Steps; step++ {
		loss, err := trainer.TrainObjectiveStepGPU(batch, 2, 8, float32(cfg.Training.LR))
		if err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			t.Fatalf("step %d non-finite loss=%g", step, loss)
		}
	}
	last, err := trainer.EvaluateObjectiveGPU(batch, 2, 8)
	if err != nil {
		t.Fatal(err)
	}
	if !(last < first) {
		t.Fatalf("Sobolev S4D loss did not decrease: first=%g last=%g", first, last)
	}
	after, err := trainer.ReadWeights()
	if err != nil {
		t.Fatal(err)
	}
	if diff := maxAbsDiffS4D(before[betaIndex], after[betaIndex]); diff <= 1e-8 {
		t.Fatalf("Sobolev beta did not change after training; diff=%g", diff)
	}
}

func TestS4DReferenceBidirectionalGroupedContinuousTraining(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	cfg, err := ParseArchConfig([]byte(`{
		"name":"s4d_reference_continuous_smoke",
		"model_dim":8,
		"seq_len":8,
		"positional_embedding":"none",
		"dropout":0.1,
		"tie_dropout":true,
		"norm_type":"layernorm",
		"norm_placement":"post_residual",
		"final_norm":false,
		"input_adapter":{"kind":"linear_frames","feature_dim":1,"bias":true,"norm":"none"},
		"blocks":[{
			"type":"s4d","state_size":8,"n_ssm":2,"bidirectional":true,
			"discretization":"bilinear","trainable_b":true,"state_lr":0.001,
			"output_transform":"glu"
		}],
		"training":{
			"objective":"classification",
			"classification":{"num_labels":2,"pooling":"mean","classifier_dropout":0},
			"optimizer":"adamw",
			"batch_tokens":16,
			"steps":40,
			"lr":0.002,
			"grad_clip":1,
			"weight_decay":0.01,
			"weight_decay_policy":"all",
			"seed":31
		}
	}`), "s4d_reference_continuous_smoke")
	if err != nil {
		t.Fatal(err)
	}
	frames := make([]float32, 16)
	for pos := 0; pos < 8; pos++ {
		noise := float32((pos%3)-1) * 0.02
		frames[pos] = -1 + noise
		frames[8+pos] = 1 + noise
	}
	batch := objectiveBatch{
		frames:               frames,
		classificationLabels: []int32{0, 1},
		classificationMask:   repeatFloat32Train(16, 1),
		classificationPos:    []int32{7, 7},
	}
	assertS4DTrainingDecreases(t, cfg, batch, 2, 8)
}

func TestS4DBatchNormReferenceStyleContinuousTrainingAndCheckpoint(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	cfg, err := ParseArchConfig([]byte(`{
		"name":"s4d_batchnorm_reference_smoke",
		"model_dim":8,
		"seq_len":8,
		"positional_embedding":"none",
		"norm_type":"batchnorm",
		"batchnorm_momentum":0.1,
		"input_adapter":{"kind":"linear_frames","feature_dim":1,"bias":true,"norm":"layernorm"},
		"blocks":[
			{"type":"s4d","state_size":8,"output_transform":"glu"},
			{"type":"swiglu"}
		],
		"training":{
			"objective":"classification",
			"classification":{"num_labels":2,"pooling":"last","classifier_dropout":0},
			"optimizer":"adamw",
			"batch_tokens":16,
			"steps":40,
			"lr":0.002,
			"scalar_lr":0.0005,
			"grad_clip":1,
			"weight_decay":0,
			"seed":29
		}
	}`), "s4d_batchnorm_reference_smoke")
	if err != nil {
		t.Fatal(err)
	}
	frames := make([]float32, 16)
	for pos := 0; pos < 8; pos++ {
		noise := float32((pos%3)-1) * 0.03
		frames[pos] = -1 + noise
		frames[8+pos] = 1 + noise
	}
	batch := objectiveBatch{
		frames:               frames,
		classificationLabels: []int32{0, 1},
		classificationMask:   repeatFloat32Train(16, 1),
		classificationPos:    []int32{7, 7},
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
	first, err := trainer.EvaluateObjectiveGPU(batch, 2, 8)
	if err != nil {
		t.Fatal(err)
	}
	for step := 0; step < cfg.Training.Steps; step++ {
		loss, err := trainer.TrainObjectiveStepGPU(batch, 2, 8, float32(cfg.Training.LR))
		if err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			t.Fatalf("step %d non-finite loss=%g", step, loss)
		}
	}
	last, err := trainer.EvaluateObjectiveGPU(batch, 2, 8)
	if err != nil {
		t.Fatal(err)
	}
	if !(last < first) {
		t.Fatalf("BatchNorm S4D classification loss did not decrease: first=%g last=%g", first, last)
	}

	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	weights, err := trainer.ReadWeights()
	if err != nil {
		t.Fatal(err)
	}
	var changedBuffers int
	for i, shape := range shapes {
		if !shape.IsBuffer {
			continue
		}
		if strings.HasSuffix(shape.Name, "_running_mean") {
			for _, value := range weights[i] {
				if math.Abs(float64(value)) > 1e-6 {
					changedBuffers++
					break
				}
			}
		}
	}
	if changedBuffers == 0 {
		t.Fatal("BatchNorm running means did not update during training")
	}

	path := filepath.Join(t.TempDir(), "batchnorm-trained.safetensors")
	if err := exportSafetensors(path, cfg, shapes, weights); err != nil {
		t.Fatal(err)
	}
	reloaded, err := loadSafetensorsWeights(path, shapes)
	if err != nil {
		t.Fatal(err)
	}
	for i, shape := range shapes {
		if !shape.IsBuffer {
			continue
		}
		if diff := maxAbsDiffS4D(weights[i], reloaded[i]); diff != 0 {
			t.Fatalf("%s checkpoint round-trip diff=%g", shape.Name, diff)
		}
	}
}

func TestBatchNormRejectsPaddedClassificationBatch(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	cfg, err := ParseArchConfig([]byte(`{
		"model_dim":8,"vocab_size":8,"seq_len":4,"norm_type":"batchnorm",
		"blocks":[{"type":"s4d","state_size":8}],
		"training":{
			"objective":"classification",
			"classification":{"num_labels":2},
			"batch_tokens":8,
			"steps":1,
			"lr":0.001
		}
	}`), "batchnorm-padding")
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
		x:                    []int{1, 2, 3, 4, 1, 2, 0, 0},
		classificationLabels: []int32{0, 1},
		classificationMask:   []float32{1, 1, 1, 1, 1, 1, 0, 0},
		classificationPos:    []int32{3, 1},
	}
	_, err = trainer.TrainObjectiveStepGPU(batch, 2, 4, float32(cfg.Training.LR))
	if err == nil || !strings.Contains(err.Error(), "does not support padded") {
		t.Fatalf("error=%v want padded-record rejection", err)
	}
}

func assertS4DTrainingDecreases(t *testing.T, cfg *ArchConfig, batch objectiveBatch, batchSize, seqLen int) {
	t.Helper()
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
	first, err := trainer.EvaluateObjectiveGPU(batch, batchSize, seqLen)
	if err != nil {
		t.Fatal(err)
	}
	for step := 0; step < cfg.Training.Steps; step++ {
		loss, err := trainer.TrainObjectiveStepGPU(batch, batchSize, seqLen, float32(cfg.Training.LR))
		if err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			t.Fatalf("step %d non-finite loss=%g", step, loss)
		}
	}
	last, err := trainer.EvaluateObjectiveGPU(batch, batchSize, seqLen)
	if err != nil {
		t.Fatal(err)
	}
	if !(last < first) {
		t.Fatalf("S4D classification loss did not decrease: first=%g last=%g", first, last)
	}
}

func repeatFloat32Train(n int, value float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = value
	}
	return out
}

func maxAbsDiffS4D(a, b []float32) float64 {
	var out float64
	for i := range a {
		diff := math.Abs(float64(a[i] - b[i]))
		if diff > out {
			out = diff
		}
	}
	return out
}
