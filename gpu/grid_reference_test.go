//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"encoding/json"
	"fmt"
	ir "github.com/mrothroc/mixlab/arch"
	"math"
	"os"
	"testing"
)

type gridReferenceTensor struct {
	Shape []int
	Data  []float32
}
type gridReferenceCase struct {
	Name, Op                string
	Kernel, Stride, Padding int
	Weights, Grads          []gridReferenceTensor
	Output, Cotangent       gridReferenceTensor
}

func TestGridSpatialPyTorchForwardBackward(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX unavailable")
	}
	encoded, err := os.ReadFile("testdata/grid_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixtures struct{ Cases []gridReferenceCase }
	if err = json.Unmarshal(encoded, &fixtures); err != nil {
		t.Fatal(err)
	}
	fixtures.Cases = append(fixtures.Cases, gridPoolCPUFixtures()...)
	fixtures.Cases = append(fixtures.Cases, gridConvTransposeCPUFixtures()...)
	for _, f := range fixtures.Cases {
		t.Run(f.Name, func(t *testing.T) {
			p := ir.NewProgram(len(f.Weights))
			names := []string{"w0"}
			code := ir.OpMaxPool2D
			if f.Op != "max_pool2d" && f.Op != "layout" {
				names = append(names, "w1", "w2")
				code = ir.OpConv2D
				if f.Op == "conv_transpose2d" {
					code = ir.OpConvTranspose2D
				}
			}
			if f.Op == "layout" {
				p.ReLU("w0", "positive")
				p.StopGradient("w0", "detached")
				p.Concat("positive", "detached", 3, "joined")
				p.Transpose("joined", []int{0, 2, 1, 3}, "transposed")
				p.Slice("transposed", 1, 4, 1, 1, "output")
			} else {
				p.AddOp(code, names, []string{"output"}, nil, []int{f.Kernel, f.Stride, f.Padding})
			}
			p.DeclareOutput("output", ir.TensorFloat32, f.Output.Shape)
			p.DeclareInput("cotangent", ir.TensorFloat32, f.Cotangent.Shape)
			p.Mul("output", "cotangent", "weighted")
			p.DeclareOutput("weighted", ir.TensorFloat32, f.Output.Shape)
			gp, err := LowerIRProgram(p)
			if err != nil {
				t.Fatal(err)
			}
			defer gp.Destroy()
			handles := make([]int64, len(f.Weights))
			defer FreeHandles(handles)
			for n, w := range f.Weights {
				handles[n], err = FromDataShape(w.Data, w.Shape)
				if err != nil {
					t.Fatal(err)
				}
			}
			inputs := []TensorInput{{Name: "cotangent", DType: TensorFloat32, Shape: f.Cotangent.Shape, Data: f.Cotangent.Data}}
			got, err := EvalProgramOutput(gp, handles, inputs, "output")
			if err != nil {
				t.Fatal(err)
			}
			checkGridReference(t, "output", got, f.Output.Data)
			weights := make([]WeightOptimizer, len(handles))
			for n := range weights {
				weights[n] = WeightOptimizer{GroupIndex: 0}
			}
			trainer, err := CreateTrainer(gp, handles, TrainerOptimizerSpec{Groups: []OptimizerGroup{{Kind: OptimizerAdamW, LR: 0, Beta1: .9, Beta2: .999, Epsilon: 1e-8}}, Weights: weights})
			if err != nil {
				t.Fatal(err)
			}
			defer TrainerDestroy(trainer)
			if _, err = TrainerComputeMeanSquareGrads(trainer, inputs, "weighted"); err != nil {
				t.Fatal(err)
			}
			for n, w := range f.Grads {
				g := make([]float32, len(w.Data))
				if err = TrainerReadGrad(trainer, n, g); err != nil {
					t.Fatal(err)
				}
				checkGridReference(t, fmt.Sprintf("gradient w%d", n), g, w.Data)
			}
		})
	}
}

func checkGridReference(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s size %d != %d", name, len(got), len(want))
	}
	maxAbs, squaredDiff, squaredReference := 0., 0., 0.
	for n, v := range got {
		d := float64(v) - float64(want[n])
		maxAbs = math.Max(maxAbs, math.Abs(d))
		squaredDiff += d * d
		squaredReference += float64(want[n]) * float64(want[n])
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Abs(float64(v-want[n])) > 1e-5+1e-4*math.Abs(float64(want[n])) {
			t.Fatalf("%s[%d]=%g want %g", name, n, v, want[n])
		}
	}
	t.Logf("%s max_abs=%g relative_l2=%g", name, maxAbs, math.Sqrt(squaredDiff/math.Max(squaredReference, 1e-30)))
}
