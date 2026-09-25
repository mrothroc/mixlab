package arch

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestDistributedPublicConfig(t *testing.T) {
	base := `{"model_dim":16,"seq_len":4,"vocab_size":32,"blocks":[{"type":"plain","heads":2}],"training":{"optimizer":"adamw","batch_tokens":8%s}}`
	parse := func(extra string) (*ArchConfig, error) {
		return ParseArchConfig([]byte(strings.Replace(base, "%s", extra, 1)), "ddp")
	}
	ordinary, err := parse("")
	if err != nil {
		t.Fatal(err)
	}
	if ordinary.Training.Distributed != nil {
		t.Fatal("default enabled DDP")
	}
	cfg, err := parse(`,"distributed":{"mode":"ddp"}`)
	if err != nil {
		t.Fatal(err)
	}
	if *cfg.Training.Distributed != (DistributedSpec{Mode: "ddp", Backend: "auto", GradientAccumulationSteps: 1, GradientBucketBytes: 32 << 20}) {
		t.Fatal(cfg.Training.Distributed)
	}
	p0, err := BuildIRProgramFromConfig(ordinary)
	if err != nil {
		t.Fatal(err)
	}
	p1, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(p0, p1) {
		t.Fatal("DDP config changed model graph")
	}
	blob, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	round, err := ParseArchConfig(blob, "roundtrip")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(round.Training.Distributed, cfg.Training.Distributed) {
		t.Fatal("roundtrip changed config")
	}
	for _, body := range []string{`{}`, `{"mode":"diloco"}`, `{"mode":"ddp","unknown":1}`, `{"mode":"ddp","gradient_accumulation_steps":0}`, `{"mode":"ddp","gradient_bucket_bytes":-1}`, `{"mode":"ddp","backend":"mpi"}`} {
		if _, err := parse(`,"distributed":` + body); err == nil {
			t.Fatalf("accepted %s", body)
		}
	}
}

func TestDistributedSupportPolicy(t *testing.T) {
	cfg, err := ParseArchConfig([]byte(`{"model_dim":16,"seq_len":4,"vocab_size":32,"blocks":[{"type":"plain","heads":2}],"training":{"optimizer":"adamw","batch_tokens":8,"distributed":{"mode":"ddp"}}}`), "ddp")
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name   string
		mutate func(*ArchConfig)
	}{
		{"objective", func(c *ArchConfig) { c.Training.Objective = ObjectiveMLM }},
		{"optimizer", func(c *ArchConfig) { c.Training.Optimizer = "muon" }},
		{"qat", func(c *ArchConfig) { c.Training.QAT = "int8" }},
		{"swa", func(c *ArchConfig) { c.Training.SWAStart = 2 }},
		{"z_loss", func(c *ArchConfig) { c.Training.ZLoss = 0.1 }},
		{"batchnorm", func(c *ArchConfig) { c.NormType = NormTypeBatchNorm }},
		{"sequence", func(c *ArchConfig) { c.Training.SeqLenSchedule = [][]int{{0, 4}} }},
		{"chunk", func(c *ArchConfig) { c.Training.ShuffleChunkTokens = 5 }},
		{"ttt", func(c *ArchConfig) { c.Training.TTTSteps = 1 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			clone := *cfg
			tc.mutate(&clone)
			if ValidateDistributedConfig(&clone) == nil {
				t.Fatal("accepted unsupported config")
			}
		})
	}
}
