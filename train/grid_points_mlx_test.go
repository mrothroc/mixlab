//go:build mlx && cgo && (darwin || linux)

package train

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"slices"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/gpu"
	"github.com/mrothroc/mixlab/internal/gridpoints"
)

// Fixed recipe: two distinct fields per transmitter case, six seeded waves,
// sum |amplitude*frequency| = 0.01. Noise uses the same fixed seed separately.
func gridPointsField(noise bool) []float32 {
	r := rand.New(rand.NewSource(20261125))
	values := make([]float32, 2*256*256)
	for row := 0; row < 2; row++ {
		var waves [6][4]float64
		for j := range waves {
			angle, frequency, phase := r.Float64()*2*math.Pi, .02+.08*r.Float64(), r.Float64()*2*math.Pi
			waves[j] = [4]float64{frequency * math.Sin(angle), frequency * math.Cos(angle), phase, .01 / (6 * frequency)}
		}
		for y := 0; y < 256; y++ {
			for x := 0; x < 256; x++ {
				v := 1.0
				for _, w := range waves {
					v += w[3] * math.Sin(w[0]*float64(y)+w[1]*float64(x)+w[2])
				}
				if noise {
					v = r.NormFloat64()
				}
				values[row*256*256+y*256+x] = float32(v)
			}
		}
	}
	return values
}

func TestGridPointsPolarD4Equivariance(t *testing.T) {
	if !runInStrictFP32Process(t) {
		return
	}
	if !mlxAvailable() {
		t.Skip("MLX required")
	}
	for _, k := range []int{512, 1024} {
		t.Run(fmt.Sprintf("K%d", k), func(t *testing.T) {
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()
			p := arch.NewProgram(1)
			p.DeclareInput("v", arch.TensorFloat32, []int{2, 3})
			p.AddOp(arch.OpCartesianToPolar, []string{"w0", "v"}, []string{"polar"}, []float32{1}, []int{k, 363, 256, 256, 2, 0})
			p.AddOp(arch.OpPolarToCartesian, []string{"polar", "v"}, []string{"roundtrip"}, []float32{1}, []int{k, 363, 256, 256, 2, 0})
			p.DeclareOutput("roundtrip", arch.TensorFloat32, []int{2, 256, 256, 1})
			program, err := gpu.LowerIRProgram(p)
			if err != nil {
				t.Fatal(err)
			}
			defer program.Destroy()
			eval := func(b data.GridBatch) []float32 {
				t.Helper()
				h, err := gpu.FromDataShape(b.Inputs, []int{2, 256, 256, 1})
				if err != nil {
					t.Fatal(err)
				}
				defer gpu.FreeHandles([]int64{h})
				out, err := gpu.EvalProgramOutput(program, []int64{h}, []gpu.TensorInput{{Name: "v", DType: gpu.TensorFloat32, Shape: []int{2, 3}, Data: b.Vectors}}, "roundtrip")
				if err != nil {
					t.Fatal(err)
				}
				return out
			}
			for _, point := range [][2]float32{{127, 127}, {0, 255}, {100.3, 57.7}} {
				for _, noise := range []bool{false, true} {
					b, err := data.NewGridBatch(data.GridGeometry{Height: 256, Width: 256, Channels: 1, TargetChannels: 1, VectorDim: 3}, 2)
					if err != nil {
						t.Fatal(err)
					}
					b.Count = 2
					b.Inputs = gridPointsField(noise)
					b.Vectors = []float32{point[1], 17, point[0], point[1], 29, point[0]}
					base := eval(b)
					for index := 0; index < 8; index++ {
						aug := b
						aug.Inputs = slices.Clone(b.Inputs)
						aug.Vectors = slices.Clone(b.Vectors)
						for row := 0; row < 2; row++ {
							for id := 0; ; id++ {
								s := fmt.Sprint(id)
								if data.GridDihedralIndex(42, 0, row, s) == index {
									aug.IDs[row] = s
									break
								}
							}
						}
						a := data.GridAugmenter{Points: gridpoints.Columns{{2, 0}}}
						if err := a.Apply(aug, 42, 0, 0); err != nil {
							t.Fatal(err)
						}
						want := slices.Clone(base)
						for row := 0; row < 2; row++ {
							if err := a.Transform(want[row*256*256:(row+1)*256*256], 256, 1, index); err != nil {
								t.Fatal(err)
							}
						}
						got := eval(aug)
						maxError := 0.0
						for i, v := range got {
							e := math.Abs(float64(v) - float64(want[i]))
							if math.IsNaN(e) || math.IsInf(e, 0) {
								t.Fatal("nonfinite roundtrip")
							}
							maxError = max(maxError, e)
						}
						t.Logf("K=%d B=2 tx=%v noise=%t D4=%d max_abs=%.9g", k, point, noise, index, maxError)
						if !noise && maxError > 1e-5 {
							t.Errorf("smooth-field D4 error %.9g exceeds 1e-5", maxError)
						}
					}
				}
			}
		})
	}
}
