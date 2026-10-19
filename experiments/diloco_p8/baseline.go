// The P8 baseline uses the production numerical learner without a coordinator.
package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/artifact/tensors"
	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/train"
	"github.com/mrothroc/mixlab/workerjob"
)

func main() {
	started := time.Now()
	config := flag.String("config", "", "DiLoCo config shared with the pair")
	pattern := flag.String("train", "", "training shard glob")
	validation := flag.String("validation", "", "validation shard glob")
	batches := flag.Uint64("validation-batches", 4, "fixed validation batches")
	limit := flag.Duration("timeout", 30*time.Minute, "bounded experiment runtime")
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	ctx, cancel := context.WithTimeout(ctx, *limit)
	defer cancel()
	emit := func(v map[string]any) error {
		v["elapsed_seconds"] = time.Since(started).Seconds()
		v["observed_at"] = time.Now().UTC().Format(time.RFC3339Nano)
		return json.NewEncoder(os.Stdout).Encode(v)
	}
	if err := baseline(ctx, *config, *pattern, *validation, *batches, emit); err != nil {
		_ = emit(map[string]any{"event": "failed", "error": err.Error()})
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func baseline(ctx context.Context, path, pattern, validation string, batches uint64, emit func(map[string]any) error) error {
	if path == "" || pattern == "" || validation == "" || batches == 0 {
		return fmt.Errorf("config, train, validation and positive validation-batches required")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	cfg, err := arch.ParseArchConfig(raw, path)
	if err != nil {
		return err
	}
	if cfg.Training.Distributed == nil || cfg.Training.Distributed.Mode != "diloco" {
		return fmt.Errorf("DiLoCo config required")
	}
	trainHash, err := data.DistributedDatasetIdentity(pattern)
	if err != nil {
		return err
	}
	valHash, err := data.DistributedDatasetIdentity(validation)
	if err != nil {
		return err
	}
	var buf bytes.Buffer
	if err := train.RunWorkerInitialModel(bytes.NewReader(raw), &buf); err != nil {
		return err
	}
	initialHash := sha256.Sum256(buf.Bytes())
	configHash := sha256.Sum256(raw)
	wire, err := tensors.Read(bytes.NewReader(buf.Bytes()), uint64(buf.Len()))
	if err != nil {
		return err
	}
	model, err := train.LocalLearnerModelFromWire(wire)
	if err != nil {
		return err
	}
	id := data.LearnerIdentity{RunID: "p8-baseline", WorkerID: "single", PartitionID: "partition-0", Count: 1, Seed: cfg.Training.Seed}
	session, err := train.NewLocalLearnerSession(cfg, pattern, id, model)
	if err != nil {
		return err
	}
	defer session.Close()
	if err := emit(map[string]any{"event": "ready", "config_sha256": hex.EncodeToString(configHash[:]), "initial_sha256": hex.EncodeToString(initialHash[:]), "train_hash": trainHash, "validation_hash": valHash, "seed": cfg.Training.Seed}); err != nil {
		return err
	}
	h := cfg.Training.Distributed.InnerSteps
	var tokens uint64
	var trainingSeconds float64
	for step := 0; step < cfg.Training.Steps; {
		begin := time.Now()
		n := h
		if step == 0 {
			result, err := session.Train(ctx, 1)
			if err != nil {
				return err
			}
			tokens += result.EffectiveTokens
			if err := emit(map[string]any{"event": "first_step", "seconds": time.Since(begin).Seconds(), "loss": result.Loss}); err != nil {
				return err
			}
			n--
		}
		if n > 0 {
			result, err := session.Train(ctx, n)
			if err != nil {
				return err
			}
			tokens += result.EffectiveTokens
		}
		trainingSeconds += time.Since(begin).Seconds()
		step += h
		begin = time.Now()
		snapshot, err := session.Snapshot()
		if err != nil {
			return err
		}
		snapshotSeconds := time.Since(begin).Seconds()
		begin = time.Now()
		loss, count, err := train.EvaluateLocalLearnerModel(ctx, cfg, snapshot.Model, workerjob.LearnerValidation{Pattern: validation, Hash: valHash, Split: "validation", BatchTokens: uint64(cfg.Training.BatchTokens), SequenceLength: uint64(cfg.SeqLen), Batches: batches}, snapshot.Precision)
		if err != nil {
			return err
		}
		if err := emit(map[string]any{"event": "evaluation", "step": step, "loss": loss, "validation_tokens": count, "effective_tokens": tokens, "training_seconds": trainingSeconds, "snapshot_seconds": snapshotSeconds, "evaluation_seconds": time.Since(begin).Seconds(), "precision": snapshot.Precision}); err != nil {
			return err
		}
	}
	return emit(map[string]any{"event": "complete", "effective_tokens": tokens, "training_seconds": trainingSeconds})
}
