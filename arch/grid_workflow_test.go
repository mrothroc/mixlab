package arch

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestGridWorkflowValidation(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit func(*ArchConfig)
		want string
	}{
		{"valid", func(c *ArchConfig) {
			c.Training.GridAugmentation = &GridAugmentationSpec{Dihedral: true}
			c.Training.InitFrom = "stage1.safetensors"
			c.Training.Freeze = []string{"network.*bias"}
			c.Training.InitAllowMissing = []string{"network.*weight"}
		}, ""},
		{"rectangle", func(c *ArchConfig) {
			c.InputAdapter.Width = 7
			c.Training.GridAugmentation = &GridAugmentationSpec{Dihedral: true}
		}, "square"},
		{"rectangle-disabled", func(c *ArchConfig) { c.InputAdapter.Width = 7; c.Training.GridAugmentation = &GridAugmentationSpec{} }, ""},
		{"bad-glob", func(c *ArchConfig) { c.Training.Freeze = []string{"["} }, "invalid weight pattern"},
		{"unmatched", func(c *ArchConfig) { c.Training.Freeze = []string{"other.*"} }, "matches no"},
		{"all-frozen", func(c *ArchConfig) { c.Training.Freeze = []string{"*"} }, "no trainable"},
		{"missing-pattern", func(c *ArchConfig) { c.Training.InitAllowMissing = []string{"other.*"} }, "matches no"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := gridTestConfig(t)
			tc.edit(cfg)
			raw, err := json.Marshal(cfg)
			if err != nil {
				t.Fatal(err)
			}
			_, err = ParseArchConfig(raw, tc.name)
			if tc.want == "" && err != nil || tc.want != "" && (err == nil || !strings.Contains(err.Error(), tc.want)) {
				t.Fatalf("error=%v want=%s", err, tc.want)
			}
		})
	}
	for _, field := range []string{`"grid_augmentation": {"dihedral":false}`, `"freeze":["*"]`, `"init_from":"weights"`, `"init_allow_missing":["*"]`} {
		_, err := ParseArchConfig([]byte(`{"model_dim":8,"vocab_size":16,"seq_len":4,"blocks":[{"type":"plain","heads":2}],"training":{`+field+`}}`), "nongrid")
		if err == nil || !strings.Contains(err.Error(), "require dense_regression") {
			t.Fatal(err)
		}
	}
}

func TestGridFreezeUsesStableMetadataOnly(t *testing.T) {
	cfg := gridTestConfig(t)
	before, err := BuildGridIRProgram(cfg, true)
	if err != nil {
		t.Fatal(err)
	}
	cfg.Training.Freeze = []string{"network.*bias", "network.conv.bias"}
	after, metas, err := buildGridGraph(cfg, true)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(before, after) {
		t.Fatal("freeze changed forward graph")
	}
	if metas[0].Frozen || !metas[1].Frozen {
		t.Fatal(metas)
	}
	if _, err = MatchGridWeightPatterns([]string{"network.conv.?eight"}, []string{"network.conv.weight"}); err != nil {
		t.Fatal(err)
	}
}
