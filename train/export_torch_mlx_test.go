//go:build mlx && cgo && (darwin || linux)

package train

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
)

// Fixtures require both phases, both channel counts, and --samples 4.
func TestGridTorchExportReferenceParity(t *testing.T) {
	root, source := os.Getenv("GRID_EXPORT_REFERENCE_DIR"), os.Getenv("GRID_EXPORT_REFERENCE_SOURCE")
	if root == "" || source == "" {
		t.Skip("set GRID_EXPORT_REFERENCE_DIR and GRID_EXPORT_REFERENCE_SOURCE for pinned four-map export parity")
	}
	if !mlxAvailable() {
		t.Skip("MLX required")
	}
	for _, channels := range []int{2, 3} {
		t.Run(fmt.Sprintf("c%d", channels), func(t *testing.T) {
			var checkpoint string
			for stage, phase := range []string{"firstU", "secondU"} {
				func() {
					dir := filepath.Join(root, fmt.Sprint(channels), phase)
					config := filepath.Join(dir, fmt.Sprintf("model%d.json", stage+1))
					cfg, err := LoadArchConfig(config)
					if err != nil {
						t.Fatal(err)
					}
					p, err := arch.BuildGridIRProgram(cfg, true)
					if err != nil {
						t.Fatal(err)
					}
					shapes, err := computeWeightShapes(cfg)
					if err != nil {
						t.Fatal(err)
					}
					load := filepath.Join(dir, "weights.safetensors")
					if stage == 1 {
						load = checkpoint
					}
					weights, _, err := loadGridWeights(load, cfg, shapes, nil)
					if err != nil {
						t.Fatal(err)
					}
					tr, err := initGPUTrainer(p, cfg, weights, nil)
					if err != nil {
						t.Fatal(err)
					}
					defer tr.CloseTrainer()
					raw, err := os.ReadFile(filepath.Join(dir, "input.npy"))
					if err != nil {
						t.Fatal(err)
					}
					x, err := decodeNPY(raw)
					if err != nil {
						t.Fatal(err)
					}
					if len(x.Shape) != 4 || x.Shape[0] != 4 {
						t.Fatal("expected four maps")
					}
					const pixels = 256 * 256
					b := data.GridBatch{BatchSize: 1, Count: 1, Geometry: data.GridGeometry{Channels: channels, Height: 256, Width: 256, TargetChannels: 1}, Inputs: make([]float32, pixels*channels), Targets: make([]float32, pixels), LossMask: make([]float32, pixels)}
					for j := range b.LossMask {
						b.LossMask[j] = 1
					}
					for _, state := range []string{"initialized", "trained"} {
						if state == "trained" {
							for c := 0; c < channels; c++ {
								for j := 0; j < pixels; j++ {
									b.Inputs[j*channels+c] = x.F32[c*pixels+j]
								}
							}
							if err = submitPreparedStepGPU(tr, objectiveBatch{grid: &b}, 1, 0, 1e-4); err != nil {
								t.Fatal(err)
							}
							if _, err = tr.CollectLossGPU(); err != nil {
								t.Fatal(err)
							}
						}
						out := make([]float32, 0, 4*pixels)
						for row := 0; row < 4; row++ {
							for c := 0; c < channels; c++ {
								for j := 0; j < pixels; j++ {
									b.Inputs[j*channels+c] = x.F32[(row*channels+c)*pixels+j]
								}
							}
							if _, err = tr.(gpuObjectiveOutputEvaluator).EvaluateObjectiveGPUWithOutputs(objectiveBatch{grid: &b}, 1, 0, []string{"predictions"}); err != nil {
								t.Fatal(err)
							}
							y, e := readTrainerOutput(tr, "predictions", []int{1, 256, 256, 1})
							if e != nil {
								t.Fatal(e)
							}
							out = append(out, y...)
						}
						w, e := readTrainerWeights(tr)
						if e != nil {
							t.Fatal(e)
						}
						if state == "trained" {
							changed := false
							for j, shape := range shapes {
								same := reflect.DeepEqual(w[j], weights[j])
								if shape.Frozen && !same {
									t.Fatal("frozen tensor changed", shape.Name)
								}
								changed = changed || !same
							}
							if !changed {
								t.Fatal("reference weights did not train")
							}
						}
						checkpoint = filepath.Join(dir, state+".st")
						if err = exportSafetensors(checkpoint, cfg, shapes, w); err != nil {
							t.Fatal(err)
						}
						packageDir := filepath.Join(t.TempDir(), "package")
						if err = RunExportTorchState(ExportTorchStateOptions{ConfigPath: config, SafetensorsLoad: checkpoint, MapPath: filepath.Join(dir, "export-map.json"), OutputDir: packageDir}); err != nil {
							t.Fatal(err)
						}
						native := filepath.Join(dir, state+"-native.npy")
						if err = writeGridNPY(native, []int{4, 1, 256, 256}, out); err != nil {
							t.Fatal(err)
						}
						report := filepath.Join(dir, state+"-export-report.json")
						cmd := exec.Command("python3", "../scripts/check_grid_torch_export.py", "--source", source, "--package", packageDir, "--inputs", filepath.Join(dir, "input.npy"), "--native", native, "--stage", fmt.Sprint(stage+1), "--report", report, "--native-output", cfg.DenseRegression.Output)
						output, e := cmd.CombinedOutput()
						if e != nil {
							t.Fatalf("%v: %s", e, output)
						}
						t.Log(string(output))
						var recorded map[string]any
						blob, e := os.ReadFile(report)
						if e != nil {
							t.Fatal(e)
						}
						if e = json.Unmarshal(blob, &recorded); e != nil {
							t.Fatal(e)
						}
					}
				}()
			}
		})
	}
}
