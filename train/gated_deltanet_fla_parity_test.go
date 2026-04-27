//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/gpu"
)

func TestGatedDeltaNetFLAParity(t *testing.T) {
	runGatedDeltaNetFLAParity(t, 1)
}

func TestGatedDeltaNetFLAParityMultiLayer(t *testing.T) {
	runGatedDeltaNetFLAParity(t, 4)
}

func runGatedDeltaNetFLAParity(t *testing.T, wantLayers int) {
	refPath := os.Getenv("FLA_REFERENCE")
	if refPath == "" {
		t.Skip("set FLA_REFERENCE=/path/to/reference.npz to enable FLA parity test")
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
	kvShare := mustNPZScalarInt64(t, ref, "kv_share") != 0
	input := mustNPZFloat32(t, ref, "input")
	expected := mustNPZFloat32(t, ref, "expected_x_hidden")
	if layers != wantLayers {
		t.Skipf("fixture layers=%d, want %d", layers, wantLayers)
	}

	shareJSON := "true"
	if !kvShare {
		shareJSON = "false"
	}
	blockJSON := fmt.Sprintf(`{"type": "gated_deltanet", "heads": %d, "d_k": %d, "d_v": %d, "kv_share": %s}`, heads, dk, dv, shareJSON)
	blocks := strings.Repeat(blockJSON+",", layers)
	blocks = strings.TrimSuffix(blocks, ",")
	cfgJSON := fmt.Sprintf(`{
		"name": "gdn_fla_parity",
		"model_dim": %d,
		"vocab_size": %d,
		"seq_len": %d,
		"blocks": [%s],
		"training": {"steps": 1, "lr": 1e-4, "seed": 7, "batch_tokens": %d}
	}`, modelDim, seqLen+1, seqLen, blocks, seqLen)

	cfg, err := ParseArchConfig([]byte(cfgJSON), "gdn_fla_parity")
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

	for blockIdx := 0; blockIdx < layers; blockIdx++ {
		prefix := fmt.Sprintf("block_%d_", blockIdx)
		for _, weight := range gdnParityWeightNames(kvShare) {
			tensorName := prefix + weight
			idx := nthWeightIndex(trainer.shapes, weight, blockIdx)
			if idx < 0 {
				t.Fatalf("missing weight occurrence %d for %q", blockIdx, weight)
			}
			if err := gpu.TrainerSetWeight(trainer.handle, idx, mustNPZFloat32(t, ref, tensorName)); err != nil {
				t.Fatalf("TrainerSetWeight(block=%d weight=%s idx=%d): %v", blockIdx, weight, idx, err)
			}
		}
	}

	tokens := make([]int, seqLen)
	targets := make([]int, seqLen)
	for i := range tokens {
		tokens[i] = i
	}
	if _, err := trainer.EvaluateGPU(tokens, targets, 1, seqLen); err != nil {
		t.Fatalf("EvaluateGPU: %v", err)
	}
	got, err := trainer.ReadOutput("x_hidden", []int{1, seqLen, modelDim})
	if err != nil {
		t.Fatalf("ReadOutput(x_hidden): %v", err)
	}

	if len(expected) == seqLen*modelDim {
		expected = append([]float32(nil), expected...)
	} else if len(expected) != len(got) {
		t.Fatalf("expected_x_hidden elems=%d, want %d", len(expected), len(got))
	}
	maxDiff := maxAbsDiff(got, expected)
	t.Logf("x_hidden max diff=%g", maxDiff)
	if maxDiff > 1e-3 {
		t.Fatalf("x_hidden max diff=%g, want <= 1e-3", maxDiff)
	}
}

func nthWeightIndex(shapes []WeightShape, name string, occurrence int) int {
	count := 0
	for i, shape := range shapes {
		if shape.Name != name {
			continue
		}
		if count == occurrence {
			return i
		}
		count++
	}
	return -1
}

func gdnParityWeightNames(kvShare bool) []string {
	names := []string{
		"norm_scale",
		"wq",
	}
	if kvShare {
		names = append(names, "w_kv")
	} else {
		names = append(names, "wk", "wv")
	}
	names = append(names,
		"q_conv",
		"k_conv",
		"v_conv",
		"w_a",
		"A_log",
		"dt_bias",
		"w_beta",
		"w_out_gate",
		"o_norm_scale",
		"wo",
	)
	return names
}

func mustNPZFloat32(t *testing.T, ref map[string]npzTensor, name string) []float32 {
	t.Helper()
	v, ok := ref[name]
	if !ok {
		t.Fatalf("missing tensor %q", name)
	}
	if len(v.F32) == 0 {
		t.Fatalf("tensor %q is not float32", name)
	}
	return v.F32
}

func mustNPZScalarInt64(t *testing.T, ref map[string]npzTensor, name string) int {
	t.Helper()
	v, ok := ref[name]
	if !ok {
		t.Fatalf("missing tensor %q", name)
	}
	if len(v.I64) != 1 {
		t.Fatalf("tensor %q has %d int64 values, want 1", name, len(v.I64))
	}
	return int(v.I64[0])
}

func optionalNPZScalarInt64(ref map[string]npzTensor, name string, fallback int) int {
	v, ok := ref[name]
	if !ok || len(v.I64) != 1 {
		return fallback
	}
	return int(v.I64[0])
}

func maxAbsDiff(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	maxDiff := float32(0)
	for i := 0; i < n; i++ {
		d := float32(math.Abs(float64(a[i] - b[i])))
		if d > maxDiff {
			maxDiff = d
		}
	}
	return maxDiff
}
