package arch

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func TestGridMuonMetadataAndValidation(t *testing.T) {
	b, err := os.ReadFile("../examples/grid_muon_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	for _, optimizer := range []string{"muon", "muon_eq_r", "normuon"} {
		t.Run(optimizer, func(t *testing.T) {
			cfg, err := ParseArchConfig(b, "grid muon")
			if err != nil {
				t.Fatal(err)
			}
			cfg.Training.Optimizer = optimizer
			encoded, _ := json.Marshal(cfg)
			cfg, err = ParseArchConfig(encoded, "roundtrip")
			if err != nil {
				t.Fatal(err)
			}
			metas, err := CollectWeightShapesFromConfig(cfg)
			if err != nil {
				t.Fatal(err)
			}
			for i, m := range metas {
				want := ""
				if i%2 == 0 {
					want = KernelLayoutOHWI
				}
				if i == 4 {
					want = KernelLayoutIHWO
				}
				if m.KernelLayout != want || m.MuonExcluded != (i == 0 || i == 6) {
					t.Fatalf("metadata[%d]=%+v", i, m)
				}
			}
		})
	}
	for _, tc := range []struct {
		name string
		edit func(*ArchConfig)
		want string
	}{
		{"unknown", func(c *ArchConfig) { c.Training.MuonExclude = []string{"missing"} }, "not a convolution"},
		{"duplicate", func(c *ArchConfig) { c.Training.MuonExclude = []string{"stem", "stem"} }, "duplicate"},
		{"weight not node", func(c *ArchConfig) { c.Training.MuonExclude = []string{"stem.weight"} }, "not a convolution"},
		{"non muon", func(c *ArchConfig) { c.Training.Optimizer = "adamw" }, "requires a grid"},
		{"shared exclusion", func(c *ArchConfig) {
			op := c.Blocks[0].Ops[0]
			op.Output = "shared"
			c.Blocks[0].Ops = append(c.Blocks[0].Ops, op)
		}, "conflicting"},
		{"shared layout", func(c *ArchConfig) {
			op := c.Blocks[0].Ops[0]
			op.Output = "shared"
			op.Op = "conv_transpose2d"
			op.Inputs = []string{"stem", "stem.weight"}
			c.Blocks[0].Ops = append(c.Blocks[0].Ops, op)
			c.Training.MuonExclude = append(c.Training.MuonExclude, "shared")
		}, "conflicting"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg, err := ParseArchConfig(b, "grid muon")
			if err != nil {
				t.Fatal(err)
			}
			tc.edit(cfg)
			encoded, _ := json.Marshal(cfg)
			_, err = ParseArchConfig(encoded, "invalid")
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, want %s", err, tc.want)
			}
		})
	}
}

func TestGridMuonSharedKernelConsistentExclusion(t *testing.T) {
	cfg := gridTestConfig(t)
	cfg.Training.Optimizer = "muon"
	op := cfg.Blocks[0].Ops[0]
	op.Output = "shared"
	cfg.Blocks[0].Ops = append(cfg.Blocks[0].Ops, op)
	cfg.Training.MuonExclude = []string{"prediction", "shared"}
	b, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ParseArchConfig(b, "shared"); err != nil {
		t.Fatal(err)
	}
}

func TestGridAdaptiveOptimizerDefaultsUnchanged(t *testing.T) {
	for _, optimizer := range []string{"adamw", "lamb"} {
		cfg := gridTestConfig(t)
		cfg.Training.Optimizer = optimizer
		b, err := json.Marshal(cfg)
		if err != nil {
			t.Fatal(err)
		}
		cfg, err = ParseArchConfig(b, "adaptive")
		if err != nil {
			t.Fatal(err)
		}
		if cfg.Training.MuonMomentum != 0 || cfg.Training.MuonBackendSteps != 0 || cfg.Training.NewtonSchulzVariant != "" || cfg.Training.MuonNesterov != nil {
			t.Fatal("inert Muon defaults changed adaptive grid checkpoint identity")
		}
	}
}

func TestMuonExclusionsRejectSequenceModels(t *testing.T) {
	b, err := os.ReadFile("../examples/plain_3L.json")
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := ParseArchConfig(b, "sequence")
	if err != nil {
		t.Fatal(err)
	}
	cfg.Training.MuonExclude = []string{"not_a_grid_node"}
	b, err = json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ParseArchConfig(b, "sequence"); err == nil || !strings.Contains(err.Error(), "requires a grid") {
		t.Fatalf("unexpected exclusion policy: %v", err)
	}
}
