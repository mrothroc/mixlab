package arch

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func patchesConfig() ArchConfig {
	c := linearFramesTestConfig()
	c.SeqLen = 6
	c.Training.BatchTokens = 12
	c.InputAdapter = &InputAdapterSpec{Kind: InputAdapterLinearPatches, Image: &PatchImageSpec{Height: 4, Width: 6, Channels: 2}, Patch: 2, Norm: "layernorm"}
	return c
}

func TestLinearPatchesConfigValidation(t *testing.T) {
	c := parseInputAdapterTestConfig(t, patchesConfig())
	if c.InputFeatureDim() != 8 || c.InputAdapter.Coords != "none" || !c.ContinuousInputEnabled() {
		t.Fatalf("adapter=%+v", c.InputAdapter)
	}
	for _, tt := range []struct {
		name string
		edit func(*ArchConfig)
		want string
	}{
		{"image", func(c *ArchConfig) { c.InputAdapter.Image = nil }, "positive"},
		{"patch", func(c *ArchConfig) { c.InputAdapter.Patch = 0 }, "positive"},
		{"divisibility", func(c *ArchConfig) { c.InputAdapter.Image.Width = 5 }, "divisible"},
		{"sequence", func(c *ArchConfig) { c.SeqLen = 4 }, "seq_len=6"},
		{"feature", func(c *ArchConfig) { c.InputAdapter.FeatureDim = 3 }, "feature_dim=8"},
		{"coords", func(c *ArchConfig) { c.InputAdapter.Coords = "rotary" }, "coords"},
		{"buckets", func(c *ArchConfig) { c.Training.LengthBuckets = []int{3, 6} }, "fixed geometry"},
		{"padding", func(c *ArchConfig) { c.InputAdapter.Augment = &PatchAugmentSpec{RandomCropPad: -1} }, "random_crop_pad"},
		{"channels", func(c *ArchConfig) { c.InputAdapter.Augment = &PatchAugmentSpec{PadValue: []float32{1}} }, "pad_value"},
		{"wrong adapter", func(c *ArchConfig) { c.InputAdapter.Kind = InputAdapterLinearFrames }, "require"},
		{"overflow", func(c *ArchConfig) { c.InputAdapter.Image.Width = 1 << 32 }, "int32"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c := patchesConfig()
			tt.edit(&c)
			raw, _ := json.Marshal(c)
			_, err := ParseArchConfig(raw, tt.name)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("got %v want %s", err, tt.want)
			}
		})
	}
}

func TestLinearPatchesLayoutIRAndCounts(t *testing.T) {
	for _, pool := range []string{"mean", "cls"} {
		for _, pos := range []string{"none", "rope", "learned_absolute"} {
			t.Run(pool+"/"+pos, func(t *testing.T) {
				base := patchesConfig()
				base.PositionalEmbedding = pos
				base.Training.Classification.Pooling = pool
				c := parseInputAdapterTestConfig(t, base)
				frames := base
				frames.InputAdapter = &InputAdapterSpec{Kind: InputAdapterLinearFrames, FeatureDim: 8, Norm: "layernorm"}
				f := parseInputAdapterTestConfig(t, frames)
				cw, err := CollectWeightShapesFromConfig(c)
				if err != nil {
					t.Fatal(err)
				}
				fw, err := CollectWeightShapesFromConfig(f)
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(cw, fw) {
					t.Fatal("coords:none changed layout")
				}
				cp, err := BuildIRProgramFromConfig(c)
				if err != nil {
					t.Fatal(err)
				}
				fp, err := BuildIRProgramFromConfig(f)
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(cp, fp) {
					t.Fatal("coords:none changed IR")
				}
				base.InputAdapter.Coords = "learned_xy"
				xy := parseInputAdapterTestConfig(t, base)
				xw, err := CollectWeightShapesFromConfig(xy)
				if err != nil {
					t.Fatal(err)
				}
				xp, err := BuildIRProgramFromConfig(xy)
				if err != nil {
					t.Fatal(err)
				}
				if len(xw) != len(cw)+2 || xp.NumWeights != len(xw) {
					t.Fatal("wrong XY weight count")
				}
				if delta := countWeightMetaElements(xw) - countWeightMetaElements(cw); delta != 5*int64(c.ModelDim) {
					t.Fatalf("delta=%d", delta)
				}
				norm, xyIdx, cls, posIdx := -1, -1, -1, -1
				for i, op := range xp.Ops {
					for _, out := range op.Outputs {
						switch out {
						case "x_frame_norm":
							norm = i
						case "patch_positioned":
							xyIdx = i
						case "cls_input":
							cls = i
						case "x_embed_pos":
							posIdx = i
						}
					}
				}
				if norm < 0 || xyIdx <= norm || (cls >= 0 && cls <= xyIdx) || (posIdx >= 0 && posIdx <= xyIdx) {
					t.Fatalf("bad emission order norm=%d xy=%d cls=%d pos=%d", norm, xyIdx, cls, posIdx)
				}
				if EstimateFLOPs(xy).ForwardFLOPs <= EstimateFLOPs(c).ForwardFLOPs {
					t.Fatal("missing XY FLOPs")
				}
			})
		}
	}
}
