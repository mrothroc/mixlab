//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"math"
	"math/rand"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

func TestMamba3CanonicalExternalPreNormGradMatchesExpanded(t *testing.T) {
	if !gpu.Available() {
		t.Skip("MLX backend not available")
	}
	const (
		B          = 2
		T          = 4
		D          = 8
		inner      = 8
		stateSize  = 4
		nGroups    = 2
		dtRank     = 2
		convKernel = 3
		scanChunk  = 3
	)

	expanded := arch.NewProgram(21)
	expanded.DeclareInput("dummy", arch.TensorFloat32, []int{1})
	expanded.DeclareOutput("loss", arch.TensorFloat32, []int{1})
	emitExpandedMamba3ExternalPreNormTestIR(expanded, B, T, inner, stateSize, nGroups, convKernel, scanChunk)
	fused := arch.NewProgram(21)
	fused.DeclareInput("dummy", arch.TensorFloat32, []int{1})
	fused.DeclareOutput("loss", arch.TensorFloat32, []int{1})
	blockInputs := make([]string, 21)
	for i := range blockInputs {
		blockInputs[i] = fmt.Sprintf("w%d", i)
	}
	fused.Mamba3CanonicalBlockExternalPreNorm(blockInputs, "out", B, T, true, scanChunk)
	addMeanLoss(fused, "out")

	expandedGPU, err := gpu.LowerIRProgram(expanded)
	if err != nil {
		t.Fatal(err)
	}
	defer expandedGPU.Destroy()
	fusedGPU, err := gpu.LowerIRProgram(fused)
	if err != nil {
		t.Fatal(err)
	}
	defer fusedGPU.Destroy()

	rng := rand.New(rand.NewSource(20260823))
	rawX := seededFloats(rng, B*T*D, 0.2)
	normalizedX := seededFloats(rng, B*T*D, 0.3)
	canonicalWeights, canonicalShapes := seededCanonicalBlockWeights(rng, D, inner, stateSize, nGroups, dtRank, convKernel)
	weights := make([][]float32, 0, 21)
	weights = append(weights, rawX, normalizedX)
	weights = append(weights, canonicalWeights[1:]...)
	shapes := make([][2]int, 0, 21)
	shapes = append(shapes, [2]int{B * T, D}, [2]int{B * T, D})
	shapes = append(shapes, canonicalShapes[1:]...)

	handles := make([]int64, len(weights))
	for i := range weights {
		handles[i], err = gpu.FromData(weights[i], shapes[i][0], shapes[i][1])
		if err != nil {
			t.Fatalf("FromData(%d): %v", i, err)
		}
		defer gpu.FreeHandle(handles[i])
	}
	inputs := []gpu.TensorInput{{Name: "dummy", DType: gpu.TensorFloat32, Shape: []int{1}, Data: []float32{0}}}
	expandedLoss, expandedGrads, err := gpu.EvalProgramGradientsForOutput(expandedGPU, handles, inputs, "loss")
	if err != nil {
		t.Fatalf("expanded gradients: %v", err)
	}
	fusedLoss, fusedGrads, err := gpu.EvalProgramGradientsForOutput(fusedGPU, handles, inputs, "loss")
	if err != nil {
		t.Fatalf("fused gradients: %v", err)
	}
	if diff := math.Abs(float64(expandedLoss - fusedLoss)); diff > 2e-5 {
		t.Fatalf("loss mismatch: expanded=%g fused=%g diff=%g", expandedLoss, fusedLoss, diff)
	}
	for i := range expandedGrads {
		maxRel, maxAbs := maxGradientError(fusedGrads[i], expandedGrads[i])
		if maxRel > 5e-3 && maxAbs > 5e-5 {
			t.Fatalf("w%d gradient mismatch: max_relative_error=%g max_absolute_error=%g", i, maxRel, maxAbs)
		}
	}
}

func emitExpandedMamba3ExternalPreNormTestIR(prog *arch.Program, B, T, inner, stateSize, nGroups, convKernel, scanChunk int) {
	prog.MatMul("w1", "w2", "x_proj")
	prog.DepthwiseConv1D("x_proj", "w3", "x_conv", B, T, inner, convKernel)
	prog.MatMul("x_conv", "w4", "dt_low")
	prog.MatMul("dt_low", "w5", "dt_raw")
	prog.Add("dt_raw", "w17", "dt")
	prog.MatMul("x_conv", "w6", "lambda_low")
	prog.MatMul("lambda_low", "w7", "lambda")
	prog.MatMul("x_conv", "w8", "theta_low")
	prog.MatMul("theta_low", "w9", "theta")
	prog.MatMul("x_conv", "w10", "b_proj")
	prog.Reshape("b_proj", []int{B * T * nGroups, stateSize}, "b_group")
	prog.RMSNorm("b_group", "w12", "b_norm", 1e-5)
	prog.Reshape("b_norm", []int{B * T, nGroups * stateSize}, "b_flat")
	prog.Add("b_flat", "w14", "b_biased")
	prog.MatMul("x_conv", "w11", "c_proj")
	prog.Reshape("c_proj", []int{B * T * nGroups, stateSize}, "c_group")
	prog.RMSNorm("c_group", "w13", "c_norm", 1e-5)
	prog.Reshape("c_norm", []int{B * T, nGroups * stateSize}, "c_flat")
	prog.Add("c_flat", "w15", "c_biased")
	prog.Mamba3SelectiveScanChunked("x_conv", "dt", "lambda", "theta", "w16", "b_biased", "c_biased", "y", B, T, inner, stateSize, nGroups, scanChunk)
	prog.RMSNorm("y", "w18", "y_norm", 1e-5)
	prog.MatMul("w1", "w19", "z")
	prog.SiLU("z", "z_act")
	prog.Mul("y_norm", "z_act", "y_gated")
	prog.MatMul("y_gated", "w20", "out_proj")
	prog.Add("w0", "out_proj", "out")
	addMeanLoss(prog, "out")
}

func TestModernMixerBatchNormContinuousClassificationTrainingAndCheckpoint(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	blocks := []struct {
		name string
		json string
	}{
		{"gated_deltanet", `{"type":"gated_deltanet","heads":2,"d_k":2,"d_v":4,"bidirectional":true}`},
		{"mamba3_canonical", `{"type":"mamba3-canonical","inner_dim":8,"state_size":4,"n_groups":2,"dt_rank":2,"use_conv":false,"bidirectional":true}`},
	}
	for _, tc := range blocks {
		t.Run(tc.name, func(t *testing.T) {
			raw := fmt.Sprintf(`{
				"name":"modern_mixer_batchnorm_smoke",
				"model_dim":8,
				"seq_len":8,
				"positional_embedding":"none",
				"norm_type":"batchnorm",
				"norm_placement":"pre",
				"batchnorm_momentum":0.1,
				"input_adapter":{"kind":"linear_frames","feature_dim":1,"bias":true,"norm":"none"},
				"blocks":[%s,{"type":"swiglu"}],
				"training":{
					"objective":"classification",
					"classification":{"num_labels":2,"pooling":"mean","classifier_dropout":0},
					"optimizer":"adamw",
					"batch_tokens":16,
					"steps":30,
					"lr":0.003,
					"grad_clip":1,
					"weight_decay":0,
					"seed":37
				}
			}`, tc.json)
			cfg, err := ParseArchConfig([]byte(raw), t.Name())
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

			frames := make([]float32, 16)
			for pos := 0; pos < 8; pos++ {
				noise := float32((pos%3)-1) * 0.03
				frames[pos] = -1 + noise
				frames[8+pos] = 1 + noise
			}
			batch := objectiveBatch{
				frames: frames, classificationLabels: []int32{0, 1},
				classificationMask: repeatFloat32Train(16, 1),
				classificationPos:  []int32{7, 7},
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
				t.Fatalf("classification loss did not decrease: first=%g last=%g", first, last)
			}

			shapes, err := computeWeightShapes(cfg)
			if err != nil {
				t.Fatal(err)
			}
			weights, err := trainer.ReadWeights()
			if err != nil {
				t.Fatal(err)
			}
			changedRunningMean := false
			for i, shape := range shapes {
				if !shape.IsBuffer || !strings.HasSuffix(shape.Name, "_running_mean") {
					continue
				}
				for _, value := range weights[i] {
					if math.Abs(float64(value)) > 1e-6 {
						changedRunningMean = true
					}
				}
			}
			if !changedRunningMean {
				t.Fatal("BatchNorm running means did not update")
			}

			path := filepath.Join(t.TempDir(), tc.name+".safetensors")
			if err := exportSafetensors(path, cfg, shapes, weights); err != nil {
				t.Fatal(err)
			}
			reloaded, err := loadSafetensorsWeights(path, shapes)
			if err != nil {
				t.Fatal(err)
			}
			for i, shape := range shapes {
				if shape.IsBuffer && maxAbsDiffS4D(weights[i], reloaded[i]) != 0 {
					t.Fatalf("%s BatchNorm buffer changed after checkpoint round-trip", shape.Name)
				}
			}
		})
	}
}
