//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

// Larger, unequal channel counts and decoder kernels exercise cuDNN plans
// missed by the original 8x8, 32-channel forward-convolution precision probe.
func precisionSpatialProbes(t *testing.T, allow bool) {
	for _, c := range []struct {
		transpose                 bool
		h, ci, co, k, stride, pad int
	}{
		{false, 32, 2, 6, 3, 1, 1},
		{false, 32, 6, 40, 5, 1, 2},
		{false, 16, 40, 50, 5, 2, 2},
		{false, 16, 100, 100, 3, 1, 1},
		{true, 8, 100, 60, 4, 2, 1},
		{true, 16, 60, 40, 6, 2, 2},
	} {
		t.Run(fmt.Sprintf("spatial/transpose=%t/c%d-%d/k%d", c.transpose, c.ci, c.co, c.k), func(t *testing.T) {
			rng := rand.New(rand.NewSource(83))
			x := make([]float32, c.h*c.h*c.ci)
			w := make([]float32, c.co*c.k*c.k*c.ci)
			b := make([]float32, c.co)
			for i := range x {
				x[i] = float32(rng.NormFloat64())
			}
			for i := range w {
				w[i] = float32(rng.NormFloat64() / math.Sqrt(float64(c.k*c.k*c.ci)))
			}
			for i := range b {
				b[i] = float32(rng.NormFloat64() * .03)
			}
			oh := (c.h+2*c.pad-c.k)/c.stride + 1
			ws := []int{c.co, c.k, c.k, c.ci}
			op := ir.OpConv2D
			if c.transpose {
				oh = (c.h-1)*c.stride - 2*c.pad + c.k
				ws = []int{c.ci, c.k, c.k, c.co}
				op = ir.OpConvTranspose2D
			}
			want := make([]float64, oh*oh*c.co)
			for i := range want {
				want[i] = float64(b[i%c.co])
			}
			// Enumerate edges directly in float64; no MLX/PyTorch convolution or
			// shared im2col implementation participates in this oracle.
			for y := 0; y < oh; y++ {
				for z := 0; z < oh; z++ {
					for ky := 0; ky < c.k; ky++ {
						for kz := 0; kz < c.k; kz++ {
							iy, iz := y*c.stride-c.pad+ky, z*c.stride-c.pad+kz
							if c.transpose {
								iy, iz = y+c.pad-ky, z+c.pad-kz
								if iy%c.stride != 0 || iz%c.stride != 0 {
									continue
								}
								iy, iz = iy/c.stride, iz/c.stride
							}
							if iy < 0 || iy >= c.h || iz < 0 || iz >= c.h {
								continue
							}
							for ci := 0; ci < c.ci; ci++ {
								for co := 0; co < c.co; co++ {
									wi := ((co*c.k+ky)*c.k+kz)*c.ci + ci
									if c.transpose {
										wi = ((ci*c.k+ky)*c.k+kz)*c.co + co
									}
									want[(y*oh+z)*c.co+co] += float64(x[(iy*c.h+iz)*c.ci+ci]) * float64(w[wi])
								}
							}
						}
					}
				}
			}
			p := ir.NewProgram(3)
			p.AddOp(op, []string{"w0", "w1", "w2"}, []string{"output"}, nil, []int{c.k, c.stride, c.pad})
			p.DeclareInput("unused", ir.TensorFloat32, []int{1})
			p.DeclareOutput("output", ir.TensorFloat32, []int{1, oh, oh, c.co})
			gp, err := LowerIRProgram(p)
			if err != nil {
				t.Fatal(err)
			}
			defer gp.Destroy()
			handles := make([]int64, 3)
			defer FreeHandles(handles)
			for i, v := range [][]float32{x, w, b} {
				shape := [][]int{{1, c.h, c.h, c.ci}, ws, {c.co}}[i]
				handles[i], err = FromDataShape(v, shape)
				if err != nil {
					t.Fatal(err)
				}
			}
			inputs := []TensorInput{{Name: "unused", DType: TensorFloat32, Shape: []int{1}, Data: []float32{0}}}
			// Repeat to exercise the cached plan as well as its initial selection.
			for repeat := 0; repeat < 2; repeat++ {
				got, err := EvalProgramOutput(gp, handles, inputs, "output")
				if err != nil {
					t.Fatal(err)
				}
				maxErr, maxRef := 0., 0.
				if len(got) != len(want) {
					t.Fatal("output shape mismatch")
				}
				for i, v := range got {
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						t.Fatal("nonfinite output")
					}
					maxErr = math.Max(maxErr, math.Abs(float64(v)-want[i]))
					maxRef = math.Max(maxRef, math.Abs(want[i]))
				}
				limit := 2e-5
				if allow {
					limit = .02
				}
				rel := maxErr / math.Max(maxRef, 1e-20)
				t.Logf("repeat=%d max_abs=%g normalized_max=%g", repeat, maxErr, rel)
				if rel > limit {
					t.Errorf("spatial precision error %g > %g", rel, limit)
				}
			}
		})
	}
}
