//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/mrothroc/mixlab/gpu"
)

func TestGatedDeltaNetFLAGradParity(t *testing.T) {
	refPath := os.Getenv("FLA_GRAD_REFERENCE")
	if refPath == "" {
		t.Skip("set FLA_GRAD_REFERENCE=/path/to/reference.npz to enable FLA gradient parity test")
	}
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}

	ref, err := loadNPZ(refPath)
	if err != nil {
		t.Fatalf("loadNPZ: %v", err)
	}

	modelDim := mustNPZScalarInt64(t, ref, "model_dim")
	seqLen := mustNPZScalarInt64(t, ref, "seq_len")
	heads := mustNPZScalarInt64(t, ref, "heads")
	dk := mustNPZScalarInt64(t, ref, "d_k")
	dv := mustNPZScalarInt64(t, ref, "d_v")
	layers := optionalNPZScalarInt64(ref, "layers", 1)
	if layers != 1 {
		t.Fatalf("gradient parity expects layers=1, got %d", layers)
	}
	kvShare := mustNPZScalarInt64(t, ref, "kv_share") != 0
	input := mustNPZFloat32(t, ref, "input")

	shareJSON := "true"
	if !kvShare {
		shareJSON = "false"
	}
	cfgJSON := fmt.Sprintf(`{
		"name": "gdn_fla_grad_parity",
		"model_dim": %d,
		"vocab_size": %d,
		"seq_len": %d,
		"blocks": [
			{"type": "gated_deltanet", "heads": %d, "d_k": %d, "d_v": %d, "kv_share": %s}
		],
		"training": {"steps": 1, "lr": 1e-4, "seed": 7, "batch_tokens": %d, "grad_clip": 0, "weight_decay": 0}
	}`, modelDim, seqLen+1, seqLen, heads, dk, dv, shareJSON, seqLen)

	cfg, err := ParseArchConfig([]byte(cfgJSON), "gdn_fla_grad_parity")
	if err != nil {
		t.Fatalf("ParseArchConfig: %v", err)
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatalf("BuildIRProgramFromConfig: %v", err)
	}
	trainerIface, err := initGPUTrainer(prog, cfg, nil, nil)
	if err != nil {
		t.Fatalf("initGPUTrainer: %v", err)
	}
	trainer, ok := trainerIface.(*mlxGPUTrainer)
	if !ok {
		t.Fatalf("trainer type=%T, want *mlxGPUTrainer", trainerIface)
	}
	defer trainer.CloseTrainer()

	embed := make([]float32, (seqLen+1)*modelDim)
	for tkn := 0; tkn < seqLen; tkn++ {
		copy(embed[tkn*modelDim:(tkn+1)*modelDim], input[tkn*modelDim:(tkn+1)*modelDim])
	}
	if err := trainer.SetWeightGPU("embed", embed); err != nil {
		t.Fatalf("SetWeightGPU(embed): %v", err)
	}
	if err := trainer.SetWeightGPU("head", make([]float32, modelDim*(seqLen+1))); err != nil {
		t.Fatalf("SetWeightGPU(head): %v", err)
	}
	if err := trainer.SetWeightGPU("final_norm", mustNPZFloat32(t, ref, "final_norm_scale")); err != nil {
		t.Fatalf("SetWeightGPU(final_norm): %v", err)
	}
	for _, weight := range gdnParityWeightNames(kvShare) {
		if err := trainer.SetWeightGPU(weight, mustNPZFloat32(t, ref, weight)); err != nil {
			t.Fatalf("SetWeightGPU(%s): %v", weight, err)
		}
	}

	tokens := make([]int, seqLen)
	targets := make([]int, seqLen)
	for i := range tokens {
		tokens[i] = i
	}
	inputs, err := trainer.makeInputs(tokens, targets, 1, seqLen)
	if err != nil {
		t.Fatalf("makeInputs: %v", err)
	}
	gotLoss, err := gpu.TrainerComputeMeanSquareGrads(trainer.handle, inputs, "x_hidden")
	if err != nil {
		t.Fatalf("TrainerComputeMeanSquareGrads: %v", err)
	}
	wantLoss := mustNPZFloat32(t, ref, "loss")[0]
	if diff := math.Abs(float64(gotLoss - wantLoss)); diff > 1e-5 {
		t.Fatalf("loss diff=%g, got=%g want=%g", diff, gotLoss, wantLoss)
	}

	maxGradDiff := float32(0)
	weights := []struct {
		trainerName string
		refName     string
	}{
		{trainerName: "final_norm", refName: "final_norm_scale"},
	}
	for _, name := range gdnParityWeightNames(kvShare) {
		weights = append(weights, struct {
			trainerName string
			refName     string
		}{trainerName: name, refName: name})
	}
	for _, weight := range weights {
		idx := gradParityWeightIndex(t, trainer.shapes, weight.trainerName)
		got := make([]float32, len(mustNPZFloat32(t, ref, weight.refName+"_grad")))
		if err := gpu.TrainerReadGrad(trainer.handle, idx, got); err != nil {
			t.Fatalf("TrainerReadGrad(%s idx=%d): %v", weight.trainerName, idx, err)
		}
		want := mustNPZFloat32(t, ref, weight.refName+"_grad")
		diff := maxAbsDiff(got, want)
		t.Logf("%s grad max diff=%g", weight.refName, diff)
		if diff > maxGradDiff {
			maxGradDiff = diff
		}
		if diff > 1e-3 {
			t.Fatalf("%s grad max diff=%g, want <= 1e-3", weight.refName, diff)
		}
	}
	t.Logf("max grad diff=%g", maxGradDiff)
}

func gradParityWeightIndex(t *testing.T, shapes []WeightShape, name string) int {
	t.Helper()
	for i, shape := range shapes {
		if shape.Name == name {
			return i
		}
	}
	t.Fatalf("missing weight %q", name)
	return -1
}
