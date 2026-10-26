//go:build mlx && cgo && (darwin || linux)

package gpu

import (
	"fmt"
	"math"
	"reflect"
	"testing"

	ir "github.com/mrothroc/mixlab/arch"
)

// Independent CPU permutation: each row is one output channel, with H,W,I
// columns. Unequal input/output channels expose accidental reshape-only paths.
func kernelMatrixCPU(x []float32, shape []int, view MuonMatrixView) []float32 {
	out := append([]float32(nil), x...)
	if view == MuonMatrixIHWO {
		ci, kh, kw, co := shape[0], shape[1], shape[2], shape[3]
		for i := 0; i < ci; i++ {
			for h := 0; h < kh; h++ {
				for w := 0; w < kw; w++ {
					for o := 0; o < co; o++ {
						out[((o*kh+h)*kw+w)*ci+i] = x[((i*kh+h)*kw+w)*co+o]
					}
				}
			}
		}
	}
	return out
}

func kernelMuonTrainer(t *testing.T, shape []int, w, driver []float32, view MuonMatrixView, group OptimizerGroup) (TrainerHandle, []TensorInput) {
	t.Helper()
	p := ir.NewProgram(1)
	p.DeclareInput("driver", ir.TensorFloat32, shape)
	p.Mul("w0", "driver", "product")
	p.Reshape("product", []int{len(w)}, "flat")
	p.MeanAxis("flat", 0, "loss")
	p.DeclareOutput("loss", ir.TensorFloat32, []int{1})
	gp, err := LowerIRProgram(p)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(gp.Destroy)
	h, err := FromDataShape(w, shape)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { FreeHandle(h) })
	tr, err := CreateTrainer(gp, []int64{h}, TrainerOptimizerSpec{Groups: []OptimizerGroup{group}, Weights: []WeightOptimizer{{GroupIndex: 0, MuonView: view}}, DefaultBaseLR: group.LR})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { TrainerDestroy(tr) })
	return tr, []TensorInput{{Name: "driver", DType: TensorFloat32, Shape: shape, Data: driver}}
}

func TestMuonKernelMatrixParityAndResume(t *testing.T) {
	lockMLXThread(t)
	if !Available() {
		t.Skip("MLX required")
	}
	for _, view := range []MuonMatrixView{MuonMatrixOHWI, MuonMatrixIHWO} {
		for _, normalization := range []MuonNormalization{MuonNormalizationNone, MuonNormalizationRowL2, MuonNormalizationNorMuon} {
			for _, variant := range []NewtonSchulzVariant{NewtonSchulzFixed, NewtonSchulzPolarExpress} {
				t.Run(fmt.Sprintf("view%d_norm%d_ns%d", view, normalization, variant), func(t *testing.T) {
					shape := []int{3, 2, 2, 2}
					if view == MuonMatrixIHWO {
						shape = []int{2, 2, 2, 3}
					}
					w, driver := make([]float32, 24), make([]float32, 24)
					for i := range w {
						w[i] = float32(math.Sin(float64(i+1))) * .2
						driver[i] = float32(math.Cos(float64(i*3 + 1)))
					}
					matrixShape, _ := view.Shape(shape)
					mw, md := kernelMatrixCPU(w, shape, view), kernelMatrixCPU(driver, shape, view)
					group := OptimizerGroup{Kind: OptimizerMuon, LR: .025, Beta1: .9, Beta2: .95, BackendSteps: 5, Nesterov: true, MuonNormalization: normalization, NewtonSchulzVariant: variant}
					kernel, in := kernelMuonTrainer(t, shape, w, driver, view, group)
					matrix, min := kernelMuonTrainer(t, matrixShape, mw, md, MuonMatrixIdentity, group)
					read := func(tr TrainerHandle) []float32 {
						v := make([]float32, 24)
						if err := TrainerReadWeight(tr, 0, v); err != nil {
							t.Fatal(err)
						}
						return v
					}
					var snapshot TrainerStateSnapshot
					var checkpoint []float32
					for step := 0; step < 4; step++ {
						for _, r := range []struct {
							tr TrainerHandle
							in []TensorInput
						}{{kernel, in}, {matrix, min}} {
							loss, err := TrainerStep(r.tr, r.in)
							if err != nil || math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
								t.Fatal(loss, err)
							}
						}
						got := kernelMatrixCPU(read(kernel), shape, view)
						requireCloseSlice(t, got, read(matrix), 2e-4)
						if step == 0 && variant == NewtonSchulzFixed {
							want := muonVariantOneStepOracle(mw, md, 3, 8, .025, .9, .95, 5, normalization)
							requireCloseSlice(t, got, want, 2e-4)
						}
						if step == 1 {
							var err error
							snapshot, err = TrainerStateSnapshotRead(kernel)
							if err != nil {
								t.Fatal(err)
							}
							checkpoint = read(kernel)
						}
					}
					resumed, rin := kernelMuonTrainer(t, shape, checkpoint, driver, view, group)
					if err := TrainerStateSnapshotRestore(resumed, snapshot); err != nil {
						t.Fatal(err)
					}
					for i := 0; i < 2; i++ {
						if _, err := TrainerStep(resumed, rin); err != nil {
							t.Fatal(err)
						}
					}
					if !reflect.DeepEqual(read(resumed), read(kernel)) {
						t.Fatal("resume changed weights")
					}
					a, err := TrainerStateSnapshotRead(resumed)
					if err != nil {
						t.Fatal(err)
					}
					b, err := TrainerStateSnapshotRead(kernel)
					if err != nil {
						t.Fatal(err)
					}
					if !reflect.DeepEqual(a, b) {
						t.Fatal("resume changed optimizer state")
					}
				})
			}
		}
	}
}
