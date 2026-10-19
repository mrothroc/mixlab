package main

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/mrothroc/mixlab/arch"
)

func TestBaselineRequiresExplicitInputs(t *testing.T) {
	for _, args := range [][3]string{{"", "train", "val"}, {"config", "", "val"}, {"config", "train", ""}} {
		if err := baseline(context.Background(), args[0], args[1], args[2], 8, nil); err == nil {
			t.Fatal("missing input accepted")
		}
	}
	if err := baseline(context.Background(), "config", "train", "val", 0, nil); err == nil {
		t.Fatal("zero validation batches accepted")
	}
}

func TestRegisteredConfigs(t *testing.T) {
	for _, seed := range []int{17, 29, 43} {
		t.Run(fmt.Sprint(seed), func(t *testing.T) {
			name := fmt.Sprintf("seed%d.json", seed)
			raw, err := os.ReadFile(name)
			if err != nil {
				t.Fatal(err)
			}
			cfg, err := arch.ParseArchConfig(raw, name)
			if err != nil {
				t.Fatal(err)
			}
			if cfg.Training.Seed != int64(seed) || cfg.Training.Steps != 400 || cfg.Training.BatchTokens != 2048 || cfg.SeqLen != 128 || cfg.VocabSize != 8192 || cfg.ModelDim != 256 {
				t.Fatal("registered workload changed")
			}
			if cfg.Training.Distributed == nil || cfg.Training.Distributed.InnerSteps != 100 {
				t.Fatal("registered inner horizon changed")
			}
		})
	}
}
