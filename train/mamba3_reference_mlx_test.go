//go:build mlx && cgo && (darwin || linux)

package train

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

func TestMamba3FullBlockMatchesReferenceFixture(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	fixtures := loadMamba3FullBlockFixtures(t)
	for _, fx := range fixtures {
		t.Run(fx.Name, func(t *testing.T) {
			cfg, err := ParseArchConfig(fx.Config, fx.Name)
			if err != nil {
				t.Fatalf("ParseArchConfig: %v", err)
			}
			prog, err := arch.BuildData2VecTeacherIRProgramFromConfig(cfg, arch.ObjectiveMNTP)
			if err != nil {
				t.Fatalf("BuildData2VecTeacherIRProgramFromConfig: %v", err)
			}
			gpuProg, err := gpu.LowerIRProgram(prog)
			if err != nil {
				t.Fatalf("LowerIRProgram: %v", err)
			}
			defer gpuProg.Destroy()

			shapes, err := computeWeightShapes(cfg)
			if err != nil {
				t.Fatalf("computeWeightShapes: %v", err)
			}
			if prog.NumWeights > len(shapes) {
				t.Fatalf("program weights=%d exceeds collected weight shapes=%d", prog.NumWeights, len(shapes))
			}
			weights := fixtureWeightsForShapes(t, fx.Weights, shapes[:prog.NumWeights])
			handles, err := uploadWeightHandles(shapes[:prog.NumWeights], weights)
			if err != nil {
				t.Fatalf("uploadWeightHandles: %v", err)
			}
			defer gpu.FreeHandles(handles)

			inputs := []gpu.TensorInput{
				{Name: "tokens", DType: gpu.TensorInt32, Shape: []int{fx.Batch, fx.SeqLen}, Data: fx.Tokens},
				{Name: "targets", DType: gpu.TensorInt32, Shape: []int{fx.Batch * fx.SeqLen}, Data: fx.Targets},
			}
			if data2VecProgramDeclaresInput(prog, "loss_mask") {
				inputs = append(inputs, gpu.TensorInput{Name: "loss_mask", DType: gpu.TensorFloat32, Shape: []int{fx.Batch * fx.SeqLen}, Data: fx.LossMask})
			}

			got, err := gpu.EvalProgramOutput(gpuProg, handles, inputs, fx.Output)
			if err != nil {
				t.Fatalf("EvalProgramOutput(%s): %v", fx.Output, err)
			}
			if len(got) != len(fx.ExpectedHidden) {
				t.Fatalf("output elems=%d want %d", len(got), len(fx.ExpectedHidden))
			}
			if diff := maxAbsDiff(got, fx.ExpectedHidden); diff > 1e-3 {
				t.Fatalf("%s L_inf=%g, want <= 1e-3\ngot=%v\nwant=%v", fx.Output, diff, got, fx.ExpectedHidden)
			}
		})
	}
}

func loadMamba3FullBlockFixtures(t *testing.T) []recurrentFullBlockFixture {
	t.Helper()
	path := filepath.Join("testdata", "mamba3_full_block_reference.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture %s: %v", path, err)
	}
	var file recurrentFullBlockFixtureFile
	if err := json.Unmarshal(data, &file); err != nil {
		t.Fatalf("decode fixture %s: %v", path, err)
	}
	if file.Version != 1 {
		t.Fatalf("fixture version=%d want 1", file.Version)
	}
	if len(file.Fixtures) == 0 {
		t.Fatalf("fixture %s contains no cases", path)
	}
	return file.Fixtures
}
