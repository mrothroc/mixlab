package train

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/artifact/tensors"
	"github.com/mrothroc/mixlab/diloco/optimizer"
	"github.com/mrothroc/mixlab/workerjob"
)

func workerInitialConfigBytes(t *testing.T, cfg *ArchConfig) []byte {
	t.Helper()
	b, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func workerInitialArtifact(t *testing.T, cfg *ArchConfig) ([]byte, []optimizer.Tensor) {
	t.Helper()
	var out bytes.Buffer
	if err := RunWorkerInitialModel(bytes.NewReader(workerInitialConfigBytes(t, cfg)), &out); err != nil {
		t.Fatal(err)
	}
	values, err := tensors.Read(bytes.NewReader(out.Bytes()), uint64(out.Len()))
	if err != nil {
		t.Fatal(err)
	}
	return out.Bytes(), values
}

func TestWorkerInitialModelConfiguredInitializationAndLayout(t *testing.T) {
	for _, mode := range []string{"xavier_uniform", "normal", "gpt2", "gptbert", "pytorch_linear", "pytorch_linear_all"} {
		for _, tied := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/tied=%v", mode, tied), func(t *testing.T) {
				cfg, _, _, _ := learnerFixture(t, 2)
				cfg.TieEmbeddings = tied
				cfg.Blocks = append(cfg.Blocks, cfg.Blocks[0])
				cfg.Training.WeightInit, cfg.Training.WeightInitStd = mode, .037
				shapes, err := computeWeightShapes(cfg)
				if err != nil {
					t.Fatal(err)
				}
				want := initWeightData(shapes, cfg.Training.Seed, mode, .037)
				blob, got := workerInitialArtifact(t, cfg)
				again, _ := workerInitialArtifact(t, cfg)
				if !bytes.Equal(blob, again) {
					t.Fatal("configured initialization is not byte deterministic")
				}
				if len(got) != len(shapes) {
					t.Fatalf("tensor count=%d, want %d", len(got), len(shapes))
				}
				model, err := LocalLearnerModelFromWire(got)
				if err != nil {
					t.Fatal(err)
				}
				for i, v := range got {
					if v.Name != fmt.Sprintf("w%d_%s", i, shapes[i].Name) || !reflect.DeepEqual(v.Shape, shapes[i].Shape) || !reflect.DeepEqual(v.Values, want[i]) {
						t.Fatalf("tensor %d differs from configured native initialization", i)
					}
				}
				if _, err := localModelData(model, shapes); err != nil {
					t.Fatal("artifact is not installable:", err)
				}
				path := filepath.Join(t.TempDir(), "generated.safetensors")
				if err := os.WriteFile(path, blob, 0o600); err != nil {
					t.Fatal(err)
				}
				if loaded, err := loadSafetensorsWeights(path, shapes); err != nil || !reflect.DeepEqual(loaded, want) {
					t.Fatal("generated artifact is not native weights-only compatible", err)
				}
				plan, err := inspectWorkerConfig(workerInitialConfigBytes(t, cfg), strings.Repeat("a", 64))
				if err != nil {
					t.Fatal(err)
				}
				layout, err := workerjob.Digest(shapes)
				if err != nil || layout != plan.WeightLayoutHash {
					t.Fatal("generated layout differs from learner probe", err)
				}
				cfg.Training.Seed++
				other, _ := workerInitialArtifact(t, cfg)
				if bytes.Equal(blob, other) {
					t.Fatal("configured seed ignored")
				}
			})
		}
	}
}

func TestWorkerInitialModelRepeatedNamesRoundTrip(t *testing.T) {
	cfg, _, _, _ := learnerFixture(t, 2)
	cfg.Blocks = append(cfg.Blocks, cfg.Blocks[0])
	_, wire := workerInitialArtifact(t, cfg)
	model, err := LocalLearnerModelFromWire(wire)
	if err != nil {
		t.Fatal(err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := localModelData(model, shapes); err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	repeated := false
	for _, v := range model {
		repeated = repeated || seen[v.Name]
		seen[v.Name] = true
	}
	if !repeated || len(model) < 11 {
		t.Fatal("fixture does not cover duplicate metadata and multi-digit indices")
	}
	if got := LocalLearnerWireTensors(model); !reflect.DeepEqual(got, wire) {
		t.Fatal("wire conversion changed ordered payload")
	}
}

func TestWorkerInitialModelConfiguredStdAndF32Storage(t *testing.T) {
	cfg, _, _, _ := learnerFixture(t, 2)
	cfg.Training.WeightInit, cfg.Training.WeightInitStd = "normal", .01
	first, _ := workerInitialArtifact(t, cfg)
	cfg.Training.WeightInitStd = .04
	second, _ := workerInitialArtifact(t, cfg)
	if bytes.Equal(first, second) {
		t.Fatal("configured initialization std ignored")
	}
	cfg.Training.ComputeDType = "bf16"
	bf16, _ := workerInitialArtifact(t, cfg)
	if !bytes.Equal(second, bf16) {
		t.Fatal("compute precision changed the F32 master initialization")
	}
}

func TestWorkerInitialModelNativeCheckpointWeightsOnlyRestart(t *testing.T) {
	cfg, dataPath, id, _ := learnerFixture(t, 2)
	cfg.Blocks = append(cfg.Blocks, cfg.Blocks[0])
	_, initial := workerInitialArtifact(t, cfg)
	model, err := LocalLearnerModelFromWire(initial)
	if err != nil {
		t.Fatal(err)
	}
	skip := false
	source, err := newLocalLearnerSession(cfg, dataPath, id, model, fakeLocalFactory(2, &skip))
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	if _, err := source.Train(context.Background(), 2); err != nil {
		t.Fatal(err)
	}
	trained, err := source.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	model = trained.Model
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w, err := localModelData(model, shapes)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "native.safetensors")
	if err := exportSafetensors(path, cfg, shapes, w); err != nil {
		t.Fatal(err)
	}
	blob, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := tensors.Read(bytes.NewReader(blob), uint64(len(blob)))
	if err != nil {
		t.Fatal(err)
	}
	if len(decoded) != len(shapes) {
		t.Fatal("native checkpoint lost tensors")
	}
	for i, v := range decoded {
		if v.Name != fmt.Sprintf("w%d_%s", i, shapes[i].Name) || !reflect.DeepEqual(v.Shape, shapes[i].Shape) || !reflect.DeepEqual(v.Values, w[i]) {
			t.Fatalf("native payload ordering/layout changed at %d", i)
		}
	}
	raw, err := LocalLearnerModelFromWire(decoded)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := localModelData(raw, shapes); err != nil || !reflect.DeepEqual(got, w) {
		t.Fatal("normalized weights-only restart changed values", err)
	}
	id.RunID = "new-weights-only-run"
	restarted, err := newLocalLearnerSession(cfg, dataPath, id, raw, fakeLocalFactory(2, &skip))
	if err != nil {
		t.Fatal(err)
	}
	defer restarted.Close()
	fresh, err := restarted.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(fresh.Model, trained.Model) || fresh.Attempt != 0 || fresh.EffectiveTokens != 0 || fresh.EffectiveExamples != 0 || fresh.Optimizer.Optimizer.AttemptedSteps != 0 || fresh.Optimizer.Tensors[0].Data[0] != 0 {
		t.Fatal("weights-only restart did not retain weights with fresh private state")
	}
	if fresh.Sampler.Identity.RunID == trained.Sampler.Identity.RunID || reflect.DeepEqual(fresh.Optimizer, trained.Optimizer) {
		t.Fatal("weights-only restart reused old learner identity or moments")
	}
	if _, err := restarted.Train(context.Background(), 2); err != nil {
		t.Fatal("fresh schedule/sampler could not train:", err)
	}
}

func TestWorkerInitialModelShapeBounds(t *testing.T) {
	for name, shapes := range map[string][]WeightShape{
		"zero":      {{Shape: []int{0}}},
		"negative":  {{Shape: []int{-1}}},
		"empty":     {{Shape: nil}},
		"rank":      {{Shape: []int{1, 1, 1, 1, 1, 1, 1, 1, 1}}},
		"oversize":  {{Shape: []int{1<<27 + 1}}},
		"overflow":  {{Shape: []int{1 << 30, 1 << 30, 16}}},
		"aggregate": {{Shape: []int{1 << 27}}, {Shape: []int{1}}},
	} {
		t.Run(name, func(t *testing.T) {
			if err := validateWorkerInitialModelShapes(shapes); err == nil {
				t.Fatal("unbounded shape accepted")
			}
		})
	}
	if err := validateWorkerInitialModelShapes([]WeightShape{{Shape: []int{1 << 26}}, {Shape: []int{1 << 26}}}); err != nil {
		t.Fatal("exact cap rejected:", err)
	}
}

func TestWorkerInitialModelNativeCheckpointPaddedIngestion(t *testing.T) {
	cfg, _, _, _ := learnerFixture(t, 2)
	shapes := []WeightShape{{Name: "scalar", Shape: []int{1}}, {Name: "matrix", Shape: []int{2, 2}}, {Name: "scalar", Shape: []int{3}}}
	w := [][]float32{{1}, {2, 3, 4, 5}, {-1, .25, 7}}
	path := filepath.Join(t.TempDir(), "padded.safetensors")
	if err := exportSafetensors(path, cfg, shapes, w); err != nil {
		t.Fatal(err)
	}
	got, err := loadSafetensorsWeights(path, shapes)
	if err != nil || !reflect.DeepEqual(got, w) {
		t.Fatal("native scalar checkpoint invalid", err)
	}
	// Exercise the actual native writer's interior and final padding directly.
	blob, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := tensors.Read(bytes.NewReader(blob), uint64(len(blob)))
	if err != nil {
		t.Fatal(err)
	}
	model, err := LocalLearnerModelFromWire(decoded)
	if err != nil {
		t.Fatal(err)
	}
	if actual, err := localModelData(model, shapes); err != nil || !reflect.DeepEqual(actual, w) {
		t.Fatal("native ingestion changed scalar or matrix payload", err)
	}
	var canonical, expected bytes.Buffer
	if err := tensors.Write(&canonical, decoded); err != nil {
		t.Fatal(err)
	}
	if err := tensors.Write(&expected, LocalLearnerWireTensors(localModel(w, shapes))); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(canonical.Bytes(), expected.Bytes()) {
		t.Fatal("native ingestion did not normalize to the same canonical artifact")
	}
}

func TestWorkerInitialModelWireRejectsNoncanonicalAndMalformedTensors(t *testing.T) {
	cfg, _, _, model := learnerFixture(t, 2)
	for name, change := range map[string]func([]optimizer.Tensor) []optimizer.Tensor{
		"raw_name":     func(v []optimizer.Tensor) []optimizer.Tensor { v[0].Name = model[0].Name; return v },
		"wrong_index":  func(v []optimizer.Tensor) []optimizer.Tensor { v[0].Name = "w1_" + model[0].Name; return v },
		"leading_zero": func(v []optimizer.Tensor) []optimizer.Tensor { v[0].Name = "w00_" + model[0].Name; return v },
		"missing_name": func(v []optimizer.Tensor) []optimizer.Tensor { v[0].Name = "w0_"; return v },
		"reorder":      func(v []optimizer.Tensor) []optimizer.Tensor { v[0], v[1] = v[1], v[0]; return v },
		"duplicate":    func(v []optimizer.Tensor) []optimizer.Tensor { v[1].Name = v[0].Name; return v },
		"shape":        func(v []optimizer.Tensor) []optimizer.Tensor { v[0].Shape[0]++; return v },
		"nan":          func(v []optimizer.Tensor) []optimizer.Tensor { v[0].Values[0] = float32(math.NaN()); return v },
		"empty":        func([]optimizer.Tensor) []optimizer.Tensor { return nil },
	} {
		t.Run(name, func(t *testing.T) {
			v := change(LocalLearnerWireTensors(cloneLearnerValue(model)))
			if result, err := LocalLearnerModelFromWire(v); err == nil || result != nil {
				t.Fatal("invalid wire tensor returned a partial model", err)
			}
		})
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	wrong := LocalLearnerWireTensors(model)
	wrong[0].Name = "w0_wrong_metadata_name"
	normalized, err := LocalLearnerModelFromWire(wrong)
	if err != nil {
		t.Fatal("canonical wire name should reach model layout validation", err)
	}
	if _, err := localModelData(normalized, shapes); err == nil {
		t.Fatal("normalization bypassed the numerical layout contract")
	}
	// Strip only the container prefix, even if a raw metadata name looks indexed.
	model[0].Name = "w7_inner_name"
	wire := LocalLearnerWireTensors(model)
	got, err := LocalLearnerModelFromWire(wire)
	if err != nil || !reflect.DeepEqual(got, model) {
		t.Fatal("conversion changed raw metadata", err)
	}
	if model[0].Name != "w7_inner_name" || wire[0].Name != "w0_w7_inner_name" {
		t.Fatal("conversion mutated caller names")
	}
}

type workerInitialFailIO struct{ err error }

func (f workerInitialFailIO) Read([]byte) (int, error)  { return 0, f.err }
func (f workerInitialFailIO) Write([]byte) (int, error) { return 0, f.err }

func TestWorkerInitialModelRejectsInvalidInputAndPropagatesIO(t *testing.T) {
	for _, b := range []string{"", "{}", `{"model_dim":8,"model_dim":16}`, strings.Repeat(" ", 256<<10+1), managedTinyConfig} {
		var out bytes.Buffer
		if err := RunWorkerInitialModel(strings.NewReader(b), &out); err == nil || out.Len() != 0 {
			t.Fatalf("invalid config produced artifact: err=%v bytes=%d", err, out.Len())
		}
	}
	want := errors.New("injected I/O failure")
	if err := RunWorkerInitialModel(workerInitialFailIO{want}, io.Discard); !errors.Is(err, want) {
		t.Fatal("input failure lost:", err)
	}
	cfg, _, _, _ := learnerFixture(t, 2)
	if err := RunWorkerInitialModel(bytes.NewReader(workerInitialConfigBytes(t, cfg)), workerInitialFailIO{want}); !errors.Is(err, want) {
		t.Fatal("output failure lost:", err)
	}
}
