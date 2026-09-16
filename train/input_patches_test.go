package train

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
)

func patchModelJSON(pool, coords, pos string) string {
	return fmt.Sprintf(`{
 "name":"patch_classifier","model_dim":8,"seq_len":6,"max_positions":7,
 "positional_embedding":%q,"norm_type":"layernorm","norm_affine":true,
 "input_adapter":{"kind":"linear_patches","image":{"height":4,"width":6,"channels":2},"patch":2,"coords":%q,"norm":"layernorm","augment":{"hflip":true,"random_crop_pad":2,"pad_value":[-1,-2]}},
 "blocks":[{"type":"plain","heads":2,"attention_mask":"bidirectional","ffn_activation":"gelu","ffn_pre_norm":true,"ffn_bias":true}],
 "training":{"objective":"classification","classification":{"num_labels":3,"pooling":%q,"classifier_dropout":0},
 "optimizer":"adamw","steps":30,"batch_tokens":12,"lr":0.001,"grad_clip":1,"weight_decay":0,"seed":23}
}`, pos, coords, pool)
}

func patchRawBatch() trainBatch {
	frames := make([]float32, 96)
	for i := range frames {
		frames[i] = float32((i*7)%31-15) / 15
	}
	return trainBatch{frames: frames, labels: []int32{1, 2}, validMask: []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}
}

func TestLinearPatchesRuntimeAndExportContract(t *testing.T) {
	c, err := ParseArchConfig([]byte(patchModelJSON("cls", "learned_xy", "none")), t.Name())
	if err != nil {
		t.Fatal(err)
	}
	if err = validateHFExportConfig(c); err != nil {
		t.Fatal(err)
	}
	shapes, err := computeWeightShapes(c)
	if err != nil {
		t.Fatal(err)
	}
	mapping, err := buildHFWeightMap(c, shapes)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"input_adapter_coord_x", "input_adapter_coord_y"} {
		i := weightShapeIndex(shapes, name)
		if i < 0 || mapping[i].HF != name {
			t.Fatalf("missing mapping for %s", name)
		}
	}
	if effectiveLoaderOptions(c).PatchTransform.Training || !trainingLoaderOptions(c).PatchTransform.Training {
		t.Fatal("augmentation mode not isolated")
	}
	raw := patchRawBatch()
	if _, err := prepareObjectiveBatch(c, raw, 0, arch.ObjectiveClassification); err != nil {
		t.Fatal(err)
	}
	raw.validMask[5] = 0
	if _, err := prepareObjectiveBatch(c, raw, 0, arch.ObjectiveClassification); err == nil {
		t.Fatal("accepted partial patch image")
	}
	c.InputAdapter.Coords = "none"
	w, err := computeWeightShapes(c)
	if err != nil {
		t.Fatal(err)
	}
	a := initWeightData(w, 23, "normal", .02)
	cc := *c
	cc.InputAdapter = &arch.InputAdapterSpec{Kind: arch.InputAdapterLinearFrames, FeatureDim: 8, Norm: "layernorm"}
	fw, err := computeWeightShapes(&cc)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a, initWeightData(fw, 23, "normal", .02)) {
		t.Fatal("coords none changed initialized weights")
	}
}

func TestPrepareLinearPatchesGeometry(t *testing.T) {
	if err := exec.Command("python3", "-c", "import numpy").Run(); err != nil {
		t.Skip("numpy unavailable")
	}
	dir := t.TempDir()
	input := filepath.Join(dir, "features.npy")
	labels := filepath.Join(dir, "labels.tsv")
	config := filepath.Join(dir, "model.json")
	if err := os.WriteFile(config, []byte(patchModelJSON("cls", "learned_xy", "none")), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(labels, []byte("0\t0\n1\t1\n2\t0\n3\t1\n"), 0600); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name  string
		t, f  int
		short bool
		want  string
	}{
		{"valid", 6, 8, false, ""}, {"feature", 6, 7, false, "patch geometry"}, {"sequence", 5, 8, false, "patch geometry"}, {"partial", 6, 8, true, "full-length"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			writeNPYFloat32(t, input, []int{4, tt.t, tt.f}, make([]float32, 4*tt.t*tt.f))
			opts := PrepareOptions{ConfigPath: config, Input: input, Output: filepath.Join(dir, tt.name), InputFormat: "continuous", LabelFile: labels, ContinuousModality: "image"}
			if tt.short {
				opts.LengthFile = filepath.Join(dir, "lengths.tsv")
				if err := os.WriteFile(opts.LengthFile, []byte("0\t5\n1\t6\n2\t6\n3\t6\n"), 0600); err != nil {
					t.Fatal(err)
				}
			}
			err := runPrepare(opts)
			if tt.want != "" {
				if err == nil || !strings.Contains(err.Error(), tt.want) {
					t.Fatalf("got %v want %s", err, tt.want)
				}
				if _, e := os.Stat(filepath.Join(opts.Output, data.DatasetManifestFilename)); !os.IsNotExist(e) {
					t.Fatal("invalid geometry wrote artifacts")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			cfg, e := LoadArchConfigQuiet(config)
			if e != nil {
				t.Fatal(e)
			}
			loader, e := data.NewLoaderWithOptions(filepath.Join(opts.Output, "train_*.bin"), 23, trainingLoaderOptions(cfg))
			if e != nil {
				t.Fatal(e)
			}
			for i := 0; i < 4; i++ {
				if _, e = loader.NextBatchDetailed(12, 6); e != nil {
					t.Fatal(e)
				}
			}
			resumed, e := data.NewLoaderWithOptions(filepath.Join(opts.Output, "train_*.bin"), 23, trainingLoaderOptions(cfg))
			if e != nil {
				t.Fatal(e)
			}
			if e = replayTrainingLoader(resumed, 4, 12, 6); e != nil {
				t.Fatal(e)
			}
			a, e := loader.NextBatchDetailed(12, 6)
			if e != nil {
				t.Fatal(e)
			}
			b, e := resumed.NextBatchDetailed(12, 6)
			if e != nil {
				t.Fatal(e)
			}
			if !reflect.DeepEqual(a, b) {
				t.Fatal("resume loader changed augmentation")
			}
		})
	}
}
