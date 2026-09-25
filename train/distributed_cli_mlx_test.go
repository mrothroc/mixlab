//go:build mlx && cgo && darwin

package train

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/gpu"
)

func TestDistributedCLIRejectsSingleton(t *testing.T) {
	cli := os.Getenv("MIXLAB_DDP_CLI")
	if cli == "" {
		t.Skip("set MIXLAB_DDP_CLI")
	}
	config, err := filepath.Abs("../examples/distributed_causal.json")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	hostfile := filepath.Join(t.TempDir(), "ring.json")
	if err := os.WriteFile(hostfile, []byte(`[["127.0.0.1:29460"]]`), 0600); err != nil {
		t.Fatal(err)
	}
	cmd := exec.CommandContext(ctx, cli, "-mode", "arch", "-config", config, "-train", "unused.bin")
	cmd.Env = append(os.Environ(), "MLX_HOSTFILE="+hostfile, "MLX_RANK=0")
	output, err := cmd.CombinedOutput()
	// MLX versions either expose size one or fail to establish a singleton
	// strict ring. Both must fail closed before constructing the data loader.
	rejected := strings.Contains(string(output), "world size in [2,4096], got 1") || strings.Contains(string(output), "DDP strict ring startup: context deadline exceeded")
	if ctx.Err() != nil || err == nil || !rejected {
		t.Fatalf("singleton was not rejected before data loading: %v\n%s", err, output)
	}
}

func TestDistributedBootstrapRejectsChangedMember(t *testing.T) {
	runTwoRankCase(t, "bootstrap_changed_member", func(t *testing.T, rank int) {
		members := []distributed.DDPGroupMember{{MemberID: "old/rank/0", Rank: 0}, {MemberID: "old/rank/1", Rank: 1}}
		expected, err := distributed.NewDDPGroupMembership("saved-run", "ddp", 0, "ring", members)
		if err != nil {
			t.Fatal(err)
		}
		group, err := gpu.BootstrapGroupRuntime(context.Background(), "ring", func(rank int) (string, error) { return fmt.Sprintf("new/rank/%d", rank), nil }, &expected)
		if group != nil {
			group.Close()
		}
		if err == nil || !strings.Contains(err.Error(), "topology mismatch") {
			t.Fatalf("changed member accepted: %v", err)
		}
	})
}

// This gate runs the released entrypoint, not the internal hardware worker.
// Supply a freshly built MLX CLI through MIXLAB_DDP_CLI.
func TestDistributedCLITrainingAndResume(t *testing.T) {
	cli := os.Getenv("MIXLAB_DDP_CLI")
	if cli == "" {
		t.Skip("set MIXLAB_DDP_CLI to a freshly built mixlab executable")
	}
	if !gpu.Available() {
		t.Skip("MLX GPU unavailable")
	}
	for _, records := range []bool{false, true} {
		t.Run(fmt.Sprintf("records_%t", records), func(t *testing.T) {
			testDistributedCLITrainingAndResume(t, cli, records)
		})
	}
}

func testDistributedCLITrainingAndResume(t *testing.T, cli string, records bool) {
	dir := t.TempDir()
	config := `{"name":"ddp_cli","model_dim":16,"seq_len":4,"vocab_size":32,"dropout":0.1,"blocks":[{"type":"plain","heads":2},{"type":"swiglu"}],"training":{"optimizer":"adamw","steps":4,"batch_tokens":8,"seed":73,"lr":0.001,"warmup_steps":0,"weight_decay":0,"distributed":{"mode":"ddp","gradient_accumulation_steps":4,"gradient_bucket_bytes":1024}}}`
	configPath := filepath.Join(dir, "model.json")
	if err := os.WriteFile(configPath, []byte(config), 0600); err != nil {
		t.Fatal(err)
	}
	tokens := make([]uint16, 257)
	for i := range tokens {
		tokens[i] = uint16(1 + i%30)
	}
	writeData := func(dir string) {
		if !records {
			writeTestShard(t, dir, "train_00.bin", tokens)
			writeTestShard(t, dir, "val_00.bin", tokens)
			return
		}
		recs := [][]uint16{{4}, {5, 6}, {7}, {8, 9}, {10}, {11, 12}, {13}, {14, 15}}
		writeNucleotideMLXSequenceShard(t, filepath.Join(dir, "train_00.bin"), recs)
		writeNucleotideMLXSequenceShard(t, filepath.Join(dir, "val_00.bin"), recs)
		m := data.DatasetManifest{Format: data.DatasetManifestFormat, Version: data.DatasetManifestVersion,
			Representation: data.DatasetRepresentationDiscreteTokens, Modality: "text", VocabSize: 32,
			TokenDType: data.DatasetTokenDTypeUint16, ShardFormat: data.DatasetShardFormatSequenceV1,
			SequenceLayout: data.DatasetSequenceLayoutOneRecordRow, RecordSeqLen: 4,
			SpecialTokenIDs: map[string]int{"pad": 0, "bos": 1, "eos": 2},
			Artifacts:       data.DatasetManifestArtifacts{Tokenizer: "tokenizer.json"},
			Splits:          map[string]data.DatasetSplit{"train": {Pattern: "train_*.bin", Tokens: 12, Shards: 1, Sequences: 8, MaxSequenceTokens: 2}, "val": {Pattern: "val_*.bin", Tokens: 12, Shards: 1, Sequences: 8, MaxSequenceTokens: 2}}}
		blob, err := json.Marshal(m)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, data.DatasetManifestFilename), blob, 0600); err != nil {
			t.Fatal(err)
		}
	}
	writeData(dir)
	launch := func(configPath, pattern, checkpoint, out, resume string, wantErr string) string {
		t.Helper()
		args := []string{"--hosts", "127.0.0.1", "--repeat-hosts", "2", "--backend", "ring", "--starting-port", fmt.Sprint(reserveRingPortPair(t)), "--", "env", cli, "-mode", "arch", "-config", configPath, "-train", pattern, "-checkpoint-dir", checkpoint, "-checkpoint-every", "2", "-safetensors", out, "-log-every", "1", "-val-every", "1", "-telemetry-out", out + ".jsonl"}
		if resume != "" {
			args = append(args, "-resume", resume)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, "mlx.launch", args...)
		blob, err := cmd.CombinedOutput()
		output := string(blob)
		if ctx.Err() != nil {
			t.Fatalf("CLI timeout: %v\n%s", ctx.Err(), output)
		}
		if wantErr != "" {
			if !strings.Contains(output, wantErr) {
				t.Fatalf("expected %q, got %v\n%s", wantErr, err, output)
			}
			return output
		}
		if err != nil {
			t.Fatalf("CLI failed: %v\n%s", err, output)
		}
		return output
	}
	checkpoint := filepath.Join(dir, "checkpoints")
	first := filepath.Join(dir, "first.safetensors")
	output := launch(configPath, filepath.Join(dir, "train_*.bin"), checkpoint, first, "", "")
	if strings.Count(output, "DDP backend=") != 1 || strings.Count(output, "attempt 4/4") != 1 {
		t.Fatalf("non-root progress or missing steps:\n%s", output)
	}
	m, err := readDistributedResumeManifest(filepath.Join(checkpoint, distributedResumeManifestFilename(4)))
	if err != nil {
		t.Fatal(err)
	}
	wantTokens := uint64(256)
	if records {
		wantTokens = 160
	}
	if m.GlobalCommittedStep != 4 || m.Sampler.LocalMicrostepsConsumed != 16 || m.Sampler.Counter == nil || m.EffectiveGlobalTokens != wantTokens {
		t.Fatalf("bad counters %+v", m)
	}
	saved2 := filepath.Join(checkpoint, distributedResumeManifestFilename(2))
	copyDir := t.TempDir()
	writeData(copyDir)
	second := filepath.Join(dir, "second.safetensors")
	launch(configPath, filepath.Join(copyDir, "train_*.bin"), filepath.Join(dir, "resumed"), second, saved2, "")
	cfg, err := ParseArchConfig([]byte(config), "cli")
	if err != nil {
		t.Fatal(err)
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		t.Fatal(err)
	}
	w1, err := loadSafetensorsWeights(first, shapes)
	if err != nil {
		t.Fatal(err)
	}
	w2, err := loadSafetensorsWeights(second, shapes)
	if err != nil {
		t.Fatal(err)
	}
	if diff := maxWeightDifference(w1, w2); diff > 1e-6 {
		t.Fatalf("CLI resume max parameter difference %g", diff)
	}
	m2, err := readDistributedResumeManifest(filepath.Join(dir, "resumed", distributedResumeManifestFilename(4)))
	if err != nil {
		t.Fatal(err)
	}
	if m.Topology.LaunchAttemptID == m2.Topology.LaunchAttemptID || m.Topology.MembersHash != m2.Topology.MembersHash || m.Topology.RunID != m2.Topology.RunID {
		t.Fatal("relaunch identity contract violated")
	}
	one, err := loadDistributedResumeState(m)
	if err != nil {
		t.Fatal(err)
	}
	two, err := loadDistributedResumeState(m2)
	if err != nil {
		t.Fatal(err)
	}
	for i := range one.Trainer.Tensors {
		if maxWeightDifference([][]float32{one.Trainer.Tensors[i].Data}, [][]float32{two.Trainer.Tensors[i].Data}) > 1e-6 {
			t.Fatal("resumed moments differ")
		}
	}
	bad := strings.Replace(config, `"gradient_accumulation_steps":4`, `"gradient_accumulation_steps":2`, 1)
	badPath := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(badPath, []byte(bad), 0600); err != nil {
		t.Fatal(err)
	}
	launch(badPath, filepath.Join(dir, "train_*.bin"), checkpoint, second, saved2, "batch topology mismatch")
	stopPath := filepath.Join(dir, "stop.json")
	stopConfig := strings.Replace(config, `"optimizer":"adamw"`, `"optimizer":"adamw","target_val_loss":100`, 1)
	if err := os.WriteFile(stopPath, []byte(stopConfig), 0600); err != nil {
		t.Fatal(err)
	}
	stopDir := filepath.Join(dir, "early-stop")
	stopped := launch(stopPath, filepath.Join(dir, "train_*.bin"), stopDir, filepath.Join(dir, "stopped.safetensors"), "", "")
	if strings.Count(stopped, "attempt 1/4") != 1 || strings.Contains(stopped, "attempt 2/4") {
		t.Fatalf("rank-zero stop failed:\n%s", stopped)
	}
	if _, err := readDistributedResumeManifest(filepath.Join(stopDir, distributedResumeManifestFilename(1))); err != nil {
		t.Fatal(err)
	}
	// Check telemetry belongs to rank zero and records optimizer attempts, not microsteps.
	for _, path := range []string{first, second} {
		lines, err := os.ReadFile(path + ".jsonl")
		if err != nil {
			t.Fatal(err)
		}
		wantLines := 4
		if path == second {
			wantLines = 2
		}
		if len(strings.Split(strings.TrimSpace(string(lines)), "\n")) != wantLines {
			t.Fatalf("telemetry: %s", lines)
		}
		for _, line := range strings.Split(strings.TrimSpace(string(lines)), "\n") {
			var s telemetrySnapshot
			if json.Unmarshal([]byte(line), &s) != nil || s.Distributed == nil || s.Distributed.Microsteps != uint64(s.Step*4) || s.Distributed.OptimizerAttempts != uint64(s.Step) || s.Distributed.EffectiveTokensPerUpdate != wantTokens/4 {
				t.Fatalf("bad telemetry: %s", line)
			}
		}
	}
}

func TestDistributedProductionSamplerGlobalBatchParity(t *testing.T) {
	runTwoRankCase(t, "production_sampler_parity", func(t *testing.T, rank int) {
		group, view := newTwoRankRuntime(t, rank)
		cfg := mustParseDistributedAccumulationConfig(t, 8)
		program, err := BuildIRProgramFromConfig(cfg)
		if err != nil {
			t.Fatal(err)
		}
		trainer, err := initGPUTrainerWithDistributedContext(program, cfg, nil, nil, &DistributedTrainerContext{GroupRuntime: group, LocalView: view, AccumulationSteps: 4})
		if err != nil {
			t.Fatal(err)
		}
		defer trainer.CloseTrainer()
		dir := t.TempDir()
		tokens := make([]uint16, 257)
		for i := range tokens {
			tokens[i] = uint16(1 + i%30)
		}
		writeTestShard(t, dir, "train.bin", tokens)
		loaders := make([]*data.DistributedLoader, 2)
		for r := range loaders {
			loaders[r], err = data.NewDistributedLoader(filepath.Join(dir, "*.bin"), cfg.Training.Seed, 2, r, 4, 4, 32, "members")
			if err != nil {
				t.Fatal(err)
			}
		}
		global := *cfg
		global.Training.BatchTokens = 64
		gp, err := BuildIRProgramFromConfig(&global)
		if err != nil {
			t.Fatal(err)
		}
		ref, err := initGPUTrainer(gp, &global, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		defer ref.CloseTrainer()
		for step := 0; step < 3; step++ {
			var allX, allY []int
			for micro := 0; micro < 4; micro++ {
				for r, l := range loaders {
					b, e := l.NextBatch(8)
					if e != nil {
						t.Fatal(e)
					}
					allX = append(allX, b.X...)
					allY = append(allY, b.Y...)
					if r == rank {
						if e = trainer.SubmitStepGPU(b.X, b.Y, 2, 4, float32(cfg.Training.LR)); e != nil {
							t.Fatal(e)
						}
						if _, e = trainer.CollectLossGPU(); e != nil {
							t.Fatal(e)
						}
					}
				}
			}
			if _, err := ref.TrainStepGPU(allX, allY, 16, 4, float32(cfg.Training.LR)); err != nil {
				t.Fatal(err)
			}
		}
		a, err := readTrainerWeights(trainer)
		if err != nil {
			t.Fatal(err)
		}
		b, err := readTrainerWeights(ref)
		if err != nil {
			t.Fatal(err)
		}
		if diff := maxWeightDifference(a, b); diff > 1e-5 {
			t.Fatalf("global batch parity diff %g", diff)
		}
	})
}
