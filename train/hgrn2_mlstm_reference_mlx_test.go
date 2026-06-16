//go:build mlx && cgo && (darwin || linux)

package train

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/gpu"
)

type recurrentFullBlockFixtureFile struct {
	Version  int                         `json:"version"`
	Fixtures []recurrentFullBlockFixture `json:"fixtures"`
}

type recurrentFullBlockFixture struct {
	Name           string          `json:"name"`
	Config         json.RawMessage `json:"config"`
	Output         string          `json:"output"`
	Batch          int             `json:"batch"`
	SeqLen         int             `json:"seq_len"`
	ModelDim       int             `json:"model_dim"`
	Tokens         []int32         `json:"tokens"`
	Targets        []int32         `json:"targets"`
	LossMask       []float32       `json:"loss_mask"`
	Weights        []fixtureWeight `json:"weights"`
	ExpectedHidden []float32       `json:"expected_hidden"`
}

type fixtureWeight struct {
	Name   string    `json:"name"`
	Shape  []int     `json:"shape"`
	Values []float32 `json:"values"`
}

func TestHGRN2MLSTMFullBlocksMatchReferenceFixtures(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	fixtures := loadRecurrentFullBlockFixtures(t)
	for _, fx := range fixtures {
		t.Run(fx.Name, func(t *testing.T) {
			cfg, err := ParseArchConfig(fx.Config, fx.Name)
			if err != nil {
				t.Fatalf("ParseArchConfig: %v", err)
			}
			if cfg.ModelDim != fx.ModelDim || cfg.SeqLen != fx.SeqLen {
				t.Fatalf("fixture dimensions model_dim=%d seq_len=%d, config has model_dim=%d seq_len=%d", fx.ModelDim, fx.SeqLen, cfg.ModelDim, cfg.SeqLen)
			}
			if len(fx.Tokens) != fx.Batch*fx.SeqLen || len(fx.Targets) != fx.Batch*fx.SeqLen {
				t.Fatalf("token/target size mismatch: tokens=%d targets=%d want %d", len(fx.Tokens), len(fx.Targets), fx.Batch*fx.SeqLen)
			}
			if len(fx.LossMask) != fx.Batch*fx.SeqLen {
				t.Fatalf("loss_mask size=%d want %d", len(fx.LossMask), fx.Batch*fx.SeqLen)
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
			if diff := maxAbsDiff(got, fx.ExpectedHidden); diff > 2e-4 {
				t.Fatalf("%s L_inf=%g, want <= 2e-4\ngot=%v\nwant=%v", fx.Output, diff, got, fx.ExpectedHidden)
			}
		})
	}
}

func loadRecurrentFullBlockFixtures(t *testing.T) []recurrentFullBlockFixture {
	t.Helper()
	path := filepath.Join("testdata", "hgrn2_mlstm_full_block_reference.json")
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

func fixtureWeightsForShapes(t *testing.T, fixtures []fixtureWeight, shapes []WeightShape) [][]float32 {
	t.Helper()
	if len(fixtures) != len(shapes) {
		t.Fatalf("fixture weight count=%d want %d", len(fixtures), len(shapes))
	}
	weights := make([][]float32, len(shapes))
	for i, shape := range shapes {
		fx := fixtures[i]
		if fx.Name != shape.Name {
			t.Fatalf("weight[%d] name=%q want %q", i, fx.Name, shape.Name)
		}
		if !reflect.DeepEqual(fx.Shape, shape.Shape) {
			t.Fatalf("weight[%d] %s shape=%v want %v", i, fx.Name, fx.Shape, shape.Shape)
		}
		want := 1
		for _, dim := range shape.Shape {
			want *= dim
		}
		if len(fx.Values) != want {
			t.Fatalf("weight[%d] %s values=%d want %d", i, fx.Name, len(fx.Values), want)
		}
		weights[i] = append([]float32(nil), fx.Values...)
	}
	return weights
}
