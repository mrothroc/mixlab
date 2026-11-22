//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

func TestGridPolarPyTorchAdjoint(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX required")
	}
	raw, err := os.ReadFile("testdata/grid_polar_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixtures struct{ Cases []gridReferenceCase }
	if err = json.Unmarshal(raw, &fixtures); err != nil {
		t.Fatal(err)
	}
	runGridReferenceCases(t, fixtures.Cases)
}

func TestGridPolarNativeForwardBackward(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX required")
	}
	for _, inverse := range []bool{false, true} {
		t.Run(map[bool]string{false: "cartesian", true: "polar"}[inverse], func(t *testing.T) {
			b, h, w, k, j, c := 2, 4, 5, 8, 7, 2
			ih, iw, oh, ow, code := h, w, k, j, ir.OpCartesianToPolar
			if inverse {
				ih, iw, oh, ow, code = k, j, h, w, ir.OpPolarToCartesian
			}
			x := make([]float32, b*ih*iw*c)
			for i := range x {
				x[i] = float32(math.Sin(float64(i) * .17))
			}
			tx := []float32{1.25, 2.5, 0, 4}
			vectors := []float32{2.5, 17, 1.25, 4, 99, 0}
			p := ir.NewProgram(1)
			p.DeclareInput("v", ir.TensorFloat32, []int{b, 3})
			p.AddOp(code, []string{"w0", "v"}, []string{"y"}, []float32{1}, []int{k, j, h, w, 2, 0})
			p.DeclareOutput("y", ir.TensorFloat32, []int{b, oh, ow, c})
			gp, e := LowerIRProgram(p)
			if e != nil {
				t.Fatal(e)
			}
			defer gp.Destroy()
			wh, e := FromDataShape(x, []int{b, ih, iw, c})
			if e != nil {
				t.Fatal(e)
			}
			defer FreeHandles([]int64{wh})
			in := []TensorInput{{Name: "v", DType: TensorFloat32, Shape: []int{b, 3}, Data: vectors}}
			got, e := EvalProgramOutput(gp, []int64{wh}, in, "y")
			if e != nil {
				t.Fatal(e)
			}
			want := make([]float32, len(got))
			grad := make([]float32, len(x))
			for n := 0; n < b; n++ {
				for r := 0; r < oh; r++ {
					for col := 0; col < ow; col++ {
						row := float64(tx[2*n]) + float64(col)*math.Sin(2*math.Pi*float64(r)/float64(k))
						cc := float64(tx[2*n+1]) + float64(col)*math.Cos(2*math.Pi*float64(r)/float64(k))
						if inverse {
							dy, dx := float64(r)-float64(tx[2*n]), float64(col)-float64(tx[2*n+1])
							a := math.Atan2(dy, dx)
							if a < 0 {
								a += 2 * math.Pi
							}
							row = a * float64(k) / (2 * math.Pi)
							cc = math.Hypot(dy, dx)
						} else {
							row = math.Max(0, math.Min(float64(h-1), row))
							cc = math.Max(0, math.Min(float64(w-1), cc))
						}
						r0, c0 := int(math.Floor(row)), int(math.Floor(cc))
						r1, c1 := r0+1, c0+1
						if inverse {
							r1 %= k
						} else {
							r1 = min(r1, h-1)
							c1 = min(c1, w-1)
						}
						fr, fc := row-float64(r0), cc-float64(c0)
						idx := []int{r0*iw + c0, r0*iw + c1, r1*iw + c0, r1*iw + c1}
						a := []float64{(1 - fr) * (1 - fc), (1 - fr) * fc, fr * (1 - fc), fr * fc}
						for ch := 0; ch < c; ch++ {
							o := ((n*oh+r)*ow+col)*c + ch
							var sum float64
							for z := range idx {
								sum += float64(x[(n*ih*iw+idx[z])*c+ch]) * a[z]
							}
							want[o] = float32(sum)
							if inverse {
								for z := range idx {
									grad[(n*ih*iw+idx[z])*c+ch] += float32(2 * sum * a[z] / float64(len(want)))
								}
							}
						}
					}
				}
			}
			checkGridReference(t, "forward", got, want)
			tr, e := CreateTrainer(gp, []int64{wh}, TrainerOptimizerSpec{Groups: []OptimizerGroup{{Kind: OptimizerAdamW, LR: 0, Beta1: .9, Beta2: .999, Epsilon: 1e-8}}, Weights: []WeightOptimizer{{GroupIndex: 0}}})
			if e != nil {
				t.Fatal(e)
			}
			defer TrainerDestroy(tr)
			if _, e = TrainerComputeMeanSquareGrads(tr, in, "y"); e != nil {
				t.Fatal(e)
			}
			actual := make([]float32, len(x))
			if e = TrainerReadGrad(tr, 0, actual); e != nil {
				t.Fatal(e)
			}
			checkGridReference(t, "adjoint", actual, grad)
			for _, bad := range []float32{-1, 5, float32(math.NaN()), float32(math.Inf(1))} {
				vectors[5] = bad
				if _, e = EvalProgramOutput(gp, []int64{wh}, in, "y"); e == nil {
					t.Fatal("invalid transmitter accepted")
				}
				if _, e = TrainerComputeMeanSquareGrads(tr, in, "y"); e == nil {
					t.Fatal("invalid transmitter reached gradients")
				}
			}
		})
	}
}

func TestGridPolarInvalidIR(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX required")
	}
	for _, params := range [][]int{{8, 7, 4, 5, 0}, {0, 7, 4, 5, 0, 1}, {8, 6, 4, 5, 0, 1}, {8, 7, 4, 5, 0, 0}, {8, 7, 4, 5, 0, 2}, {8, 7, 4, 5, -1, 1}, {8, 7, math.MaxInt32, 5, 0, 1}} {
		p := ir.NewProgram(1)
		p.DeclareInput("v", ir.TensorFloat32, []int{1, 2})
		p.AddOp(ir.OpPolarToCartesian, []string{"w0", "v"}, []string{"y"}, []float32{1}, params)
		p.DeclareOutput("y", ir.TensorFloat32, []int{1, 4, 5, 1})
		gp, e := LowerIRProgram(p)
		if e != nil {
			t.Fatal(e)
		}
		wh, e := FromDataShape(make([]float32, 56), []int{1, 8, 7, 1})
		if e != nil {
			t.Fatal(e)
		}
		_, e = EvalProgramOutput(gp, []int64{wh}, []TensorInput{{Name: "v", DType: TensorFloat32, Shape: []int{1, 2}, Data: []float32{1, 2}}}, "y")
		FreeHandles([]int64{wh})
		gp.Destroy()
		if e == nil || strings.TrimSpace(e.Error()) == "" {
			t.Fatal("accepted", params)
		}
	}
}
