//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

// Each child owns a fresh MLX/cuBLAS/cuDNN/Metal cache. Running two policies
// inside one process would test cached algorithms rather than the requested
// setting. Metal children run by default; CUDA children need a CUDA host opt-in.
func TestGPUPrecisionSubprocess(t *testing.T) {
	backend := PrecisionBackend()
	if backend != "cuda" && backend != "metal" {
		t.Skip("GPU precision acceptance")
	}
	if child := os.Getenv("MIXLAB_TEST_TF32_CHILD"); child != "" {
		lockMLXThread(t)
		allow := child == "1"
		p, err := ConfigurePrecision("float32", &allow)
		if err != nil || p.TF32Allowed != allow {
			t.Fatalf("policy %+v: %v", p, err)
		}
		if !Available() {
			t.Fatalf("%s acceptance child has no GPU", backend)
		}
		for _, conv := range []bool{false, true} {
			for _, depth := range []int{1, 4} {
				t.Run(fmt.Sprintf("conv=%v/depth=%d", conv, depth), func(t *testing.T) { precisionNumericalProbe(t, conv, depth, allow) })
			}
		}
		precisionSpatialProbes(t, allow)
		other := !allow
		if _, err := ConfigurePrecision("float32", &other); err == nil {
			t.Fatal("changed initialized TF32 policy")
		}
		return
	}
	if backend == "cuda" && os.Getenv("MIXLAB_TEST_CUDA_PRECISION") != "1" {
		t.Skip("set MIXLAB_TEST_CUDA_PRECISION=1 on a CUDA host")
	}
	for _, value := range []string{"0", "1"} {
		cmd := exec.Command(os.Args[0], "-test.run=^TestGPUPrecisionSubprocess$", "-test.v")
		for _, e := range os.Environ() {
			if !strings.HasPrefix(e, "MLX_ENABLE_TF32=") && !strings.HasPrefix(e, "NVIDIA_TF32_OVERRIDE=") && !strings.HasPrefix(e, "MIXLAB_TEST_TF32_CHILD=") {
				cmd.Env = append(cmd.Env, e)
			}
		}
		cmd.Env = append(cmd.Env, "MIXLAB_TEST_TF32_CHILD="+value)
		out, err := cmd.CombinedOutput()
		t.Logf("TF32=%s\n%s", value, out)
		if err != nil {
			t.Fatalf("child: %v", err)
		}
	}
}

func precisionNumericalProbe(t *testing.T, conv bool, depth int, allow bool) {
	const rows, channels = 64, 32
	kernel := 1
	if conv {
		kernel = 3
	}
	x := make([]float32, rows*channels)
	w := make([]float32, channels*kernel*kernel*channels)
	b := make([]float32, channels)
	for i := range x {
		x[i] = float32(math.Sin(float64(i) * 1.371))
	}
	for i := range w {
		w[i] = float32(.04 * math.Cos(float64(i)*.7351))
	}
	for i := range b {
		b[i] = float32(.003 * math.Sin(float64(i)))
	}
	shape := []int{rows, channels}
	wshape := []int{channels, channels}
	if conv {
		shape = []int{1, 8, 8, channels}
		wshape = []int{channels, 3, 3, channels}
	}
	p := ir.NewProgram(3)
	last := "w0"
	for i := 0; i < depth; i++ {
		out := fmt.Sprintf("layer%d", i)
		if conv {
			p.AddOp(ir.OpConv2D, []string{last, "w1", "w2"}, []string{out}, nil, []int{3, 1, 1})
		} else {
			p.MatMul(last, "w1", out)
		}
		last = out
	}
	p.DeclareOutput(last, ir.TensorFloat32, shape)
	p.DeclareInput("unused", ir.TensorFloat32, []int{1})
	gp, err := LowerIRProgram(p)
	if err != nil {
		t.Fatal(err)
	}
	defer gp.Destroy()
	handles := make([]int64, 3)
	defer FreeHandles(handles)
	for i, v := range [][]float32{x, w, b} {
		s := shape
		if i == 1 {
			s = wshape
		}
		if i == 2 {
			s = []int{channels}
		}
		handles[i], err = FromDataShape(v, s)
		if err != nil {
			t.Fatal(err)
		}
	}
	trainer, err := CreateTrainer(gp, handles, TrainerOptimizerSpec{Weights: make([]WeightOptimizer, 3), Groups: []OptimizerGroup{{Kind: OptimizerAdamW, LR: 0, Beta1: .9, Beta2: .999, Epsilon: 1e-8}}})
	if err != nil {
		t.Fatal(err)
	}
	defer TrainerDestroy(trainer)
	in := []TensorInput{{Name: "unused", DType: TensorFloat32, Shape: []int{1}, Data: []float32{0}}}
	got, err := EvalProgramOutput(gp, handles, in, last)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := TrainerComputeMeanSquareGrads(trainer, in, last)
	if err != nil {
		t.Fatal(err)
	}
	want, grads := precisionOracle(x, w, b, conv, depth)
	limit := 2e-5
	if allow {
		limit = .02
	}
	check := func(label string, actual []float32, reference []float64) {
		maxErr, maxRef := 0., 0.
		for i, v := range actual {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("nonfinite %s", label)
			}
			maxErr = math.Max(maxErr, math.Abs(float64(v)-reference[i]))
			maxRef = math.Max(maxRef, math.Abs(reference[i]))
		}
		rel := maxErr / math.Max(maxRef, 1e-20)
		t.Logf("%s normalized_max_error=%g", label, rel)
		if rel > limit {
			t.Fatalf("%s error %g > %g", label, rel, limit)
		}
	}
	check("forward", got, want)
	wantLoss := 0.
	for _, v := range want {
		wantLoss += v * v / float64(len(want))
	}
	check("loss", []float32{loss}, []float64{wantLoss})
	actualGrads := make([][]float32, len(grads))
	for i, wantGrad := range grads {
		g := make([]float32, len(wantGrad))
		if err := TrainerReadGrad(trainer, i, g); err != nil {
			t.Fatal(err)
		}
		check(fmt.Sprintf("gradient%d", i), g, wantGrad)
		actualGrads[i] = g
	}
	if dir := os.Getenv("MIXLAB_TF32_PROBE_OUT"); dir != "" {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			t.Fatal(err)
		}
		blob, err := json.Marshal(map[string]any{"conv": conv, "depth": depth, "tf32_allowed": allow, "x": x, "w": w, "b": b, "output": got, "loss": loss, "gradients": actualGrads})
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("conv-%t-depth-%d-tf32-%t.json", conv, depth, allow)), blob, 0o600); err != nil {
			t.Fatal(err)
		}
	}
}

// Independent float64 dense contraction oracle, including accumulated gradients
// for repeated applications of the SAME weight (not independent layer copies).
func precisionOracle(x, w, b []float32, conv bool, depth int) ([]float64, [][]float64) {
	const c = 32
	convert := func(v []float32) []float64 {
		r := make([]float64, len(v))
		for i, x := range v {
			r[i] = float64(x)
		}
		return r
	}
	weights, bias := convert(w), convert(b)
	states := [][]float64{convert(x)}
	each := func(visit func(out, in, wi int)) {
		for row := 0; row < 64; row++ {
			for o := 0; o < c; o++ {
				if !conv {
					for i := 0; i < c; i++ {
						visit(row*c+o, row*c+i, i*c+o)
					}
					continue
				}
				for ky := 0; ky < 3; ky++ {
					for kx := 0; kx < 3; kx++ {
						y, z := row/8+ky-1, row%8+kx-1
						if y < 0 || y >= 8 || z < 0 || z >= 8 {
							continue
						}
						for i := 0; i < c; i++ {
							visit(row*c+o, (y*8+z)*c+i, ((o*3+ky)*3+kx)*c+i)
						}
					}
				}
			}
		}
	}
	for d := 0; d < depth; d++ {
		next := make([]float64, len(x))
		if conv {
			for i := range next {
				next[i] = bias[i%c]
			}
		}
		each(func(o, i, wi int) { next[o] += states[d][i] * weights[wi] })
		states = append(states, next)
	}
	grad := make([]float64, len(x))
	for i, v := range states[depth] {
		grad[i] = 2 * v / float64(len(x))
	}
	gw, gb := make([]float64, len(w)), make([]float64, len(b))
	for d := depth - 1; d >= 0; d-- {
		prev := make([]float64, len(x))
		each(func(o, i, wi int) { prev[i] += grad[o] * weights[wi]; gw[wi] += grad[o] * states[d][i] })
		if conv {
			for i, g := range grad {
				gb[i%c] += g
			}
		}
		grad = prev
	}
	return states[depth], [][]float64{grad, gw, gb}
}
