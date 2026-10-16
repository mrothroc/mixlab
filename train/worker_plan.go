package train

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/gpu"
	"github.com/mrothroc/mixlab/internal/strictjson"
	"github.com/mrothroc/mixlab/workerjob"
	"github.com/mrothroc/mixlab/workerprobe"
)

// RunWorkerPlan reads only bounded config bytes from stdin. It never probes a
// device, reads datasets, loads weights, or opens a network/control endpoint.
func RunWorkerPlan(in io.Reader, out io.Writer) error {
	return runWorkerPlan(in, out, false)
}

// RunWorkerLearnerProbe additionally resolves device precision; it still never
// reads datasets, model weights, or any credential/network endpoint.
func RunWorkerLearnerProbe(in io.Reader, out io.Writer) error {
	return runWorkerPlan(in, out, true)
}

func runWorkerPlan(in io.Reader, out io.Writer, device bool) error {
	b, err := io.ReadAll(io.LimitReader(in, (256<<10)+1))
	if err != nil {
		return err
	}
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	build, err := workerjob.FileDigest(exe)
	if err != nil {
		return err
	}
	p, err := inspectWorkerConfig(b, build)
	if err != nil {
		return err
	}
	if device {
		if p.Learner == nil {
			return fmt.Errorf("learner probe requires DiLoCo config")
		}
		cfg, err := arch.ParseArchConfig(b, "learner precision probe")
		if err != nil {
			return err
		}
		policy, err := trainingPrecision(cfg)
		if err != nil {
			return err
		}
		p.Learner.Precision = &policy
		if err := p.Validate(); err != nil {
			return err
		}
	}
	return json.NewEncoder(out).Encode(p)
}

func checkManagedLayout(a workerjob.Assignment, shapes []WeightShape, spec gpu.TrainerOptimizerSpec) error {
	l, err := workerjob.Digest(shapes)
	if err != nil {
		return err
	}
	o, err := optimizerSpecHash(spec)
	if err != nil {
		return err
	}
	if l != a.WeightLayoutSHA256 || o != a.OptimizerSHA256 {
		return fmt.Errorf("managed weight layout or optimizer differs from admission")
	}
	return nil
}

func inspectWorkerConfig(b []byte, build string) (workerprobe.Plan, error) {
	var out workerprobe.Plan
	if len(b) == 0 || len(b) > 256<<10 {
		return out, fmt.Errorf("bounded worker config required")
	}
	if err := strictjson.Validate(b, 64); err != nil {
		return out, err
	}
	var compact bytes.Buffer
	if err := json.Compact(&compact, b); err != nil {
		return out, err
	}
	cfg, err := arch.ParseArchConfig(compact.Bytes(), "managed numerical plan")
	if err != nil {
		return out, err
	}
	if err := arch.ValidateDistributedConfig(cfg); err != nil {
		return out, err
	}
	d := cfg.Training.Distributed
	if d == nil || (d.Mode != "diloco" && d.Backend != "ring") || cfg.CharVocabSize != 0 {
		return out, fmt.Errorf("managed plan requires ring DDP or DiLoCo without external char artifacts")
	}
	program, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		return out, err
	}
	for _, op := range program.Ops {
		if op.Code == arch.OpRandomNormal {
			return out, fmt.Errorf("DDP rejects unkeyed RandomNormal")
		}
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		return out, err
	}
	for _, s := range shapes {
		if s.IsBuffer {
			return out, fmt.Errorf("DDP rejects mutable buffer %s", s.Name)
		}
	}
	spec, err := buildTrainerOptimizerSpec(cfg, shapes)
	if err != nil {
		return out, err
	}
	spec.ComputeDType, err = gpuComputeDTypeForTraining(cfg)
	if err != nil {
		return out, err
	}
	digest := sha256.Sum256(compact.Bytes())
	out = workerprobe.Plan{Version: workerprobe.PlanVersion, BuildID: build, ConfigHash: hex.EncodeToString(digest[:]), DType: "fp32"}
	if d.Mode == "diloco" {
		out.Learner = &workerprobe.LearnerPlan{Outer: run.Outer{Kind: d.OuterOptimizer, LR: d.OuterLR, Momentum: *d.OuterMomentum, Aggregation: d.Aggregation, InnerSteps: uint64(d.InnerSteps), TotalSteps: uint64(cfg.Training.Steps), Accumulation: uint64(d.GradientAccumulationSteps)}, BatchTokens: cfg.Training.BatchTokens, SequenceLength: cfg.SeqLen}
	}
	if cfg.Training.EffectiveComputeDType() == "bf16" {
		out.DType = "bf16"
	}
	out.ProgramHash, err = workerjob.Digest(program)
	if err != nil {
		return out, err
	}
	out.WeightLayoutHash, err = workerjob.Digest(shapes)
	if err != nil {
		return out, err
	}
	out.OptimizerHash, err = optimizerSpecHash(spec)
	if err != nil {
		return out, err
	}
	return out, out.Validate()
}
