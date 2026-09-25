//go:build mlx && cgo && (darwin || linux)

package train

import (
	"context"
	"fmt"
	"math"
	"os"
	"runtime"
	"time"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/gpu"
)

// runDistributedTrain owns optimizer attempts, not gradient synchronization.
// The C++ trainer remains the only owner of collective update transactions.
func runDistributedTrain(cfg *ArchConfig, pattern string, opts TrainOptions) (TrainResult, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := arch.ValidateDistributedConfig(cfg); err != nil {
		return TrainResult{}, err
	}
	if err := validateRunTrainOptions(opts); err != nil {
		return TrainResult{}, err
	}
	if pattern == "" {
		return TrainResult{}, fmt.Errorf("DDP requires -train")
	}
	if opts.OptimizerOverride != nil || opts.DoFullEval || opts.SWAStartOverride != nil || opts.SWADecayOverride != nil || opts.SWAIntervalOverride != nil {
		return TrainResult{}, fmt.Errorf("R1 DDP does not support optimizer overrides, SWA overrides, or in-run full eval; evaluate the student artifact separately")
	}
	backend, err := distributedBackend(cfg.Training.Distributed.Backend, runtime.GOOS)
	if err != nil {
		return TrainResult{}, err
	}
	var expected *distributed.DDPGroupMembership
	if opts.Resume != "" {
		m, err := resolveDistributedResumeManifest(opts.Resume)
		if err != nil {
			return TrainResult{}, err
		}
		membership, err := distributed.NewDDPGroupMembership(m.Topology.RunID, m.Topology.GroupID, m.Topology.MembershipGeneration, m.Topology.Backend, m.Topology.OrderedMembers)
		if err != nil {
			return TrainResult{}, err
		}
		expected = &membership
	}
	// The executable adapter supplies stable host identity; no hostfile or rank
	// environment is interpreted by the trainer or sampler.
	host, err := os.Hostname()
	if err != nil {
		return TrainResult{}, err
	}
	group, err := gpu.BootstrapGroupRuntime(context.Background(), backend, func(rank int) (string, error) {
		return fmt.Sprintf("%s/rank/%d", host, rank), nil
	}, expected)
	if err != nil {
		return TrainResult{}, err
	}
	defer group.Close()
	loader, err := data.NewDistributedLoader(pattern, cfg.Training.Seed, group.WorldSize(), group.Rank(), cfg.SeqLen, effectiveShuffleChunkTokens(cfg), cfg.VocabSize, group.LocalView().Membership.MembersHash)
	if err != nil {
		return TrainResult{}, err
	}
	manifest, _, err := validateDatasetManifestForConfig(cfg, pattern)
	if err != nil {
		return TrainResult{}, err
	}
	if manifest != nil && manifest.ShardFormat == data.DatasetShardFormatSequenceV1 {
		cfg.Training.DatasetRecordFraming = true
		cfg.Training.DatasetPADID = manifest.SpecialTokenIDs["pad"]
		cfg.Training.DatasetBOSID = manifest.SpecialTokenIDs["bos"]
		cfg.Training.DatasetEOSID = manifest.SpecialTokenIDs["eos"]
	}
	if cfg.CharVocabSize > 0 {
		if _, err := configureCharFeaturesForTraining(cfg, pattern); err != nil {
			return TrainResult{}, err
		}
	}
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		return TrainResult{}, err
	}
	for _, op := range prog.Ops {
		if op.Code == arch.OpRandomNormal {
			return TrainResult{}, fmt.Errorf("DDP rejects unkeyed RandomNormal")
		}
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		return TrainResult{}, err
	}
	for _, s := range shapes {
		if s.IsBuffer {
			return TrainResult{}, fmt.Errorf("DDP does not support mutable buffer %s", s.Name)
		}
	}
	ctx := &DistributedTrainerContext{GroupRuntime: group, LocalView: group.LocalView(), GradientBucketBytes: uint64(cfg.Training.Distributed.GradientBucketBytes), AccumulationSteps: cfg.Training.Distributed.GradientAccumulationSteps, DatasetHash: loader.State().DatasetID}
	spec, err := buildTrainerOptimizerSpec(cfg, shapes)
	if err != nil {
		return TrainResult{}, err
	}
	spec.ComputeDType, err = gpuComputeDTypeForTraining(cfg)
	if err != nil {
		return TrainResult{}, err
	}
	sched, steps := buildTrainingScheduler(cfg.Training)
	start := 0
	early := newEarlyStopState(cfg.Training.EarlyStop)
	var resume *DistributedResumePlan
	var weights [][]float32
	if opts.Resume != "" {
		plan, err := PrepareDistributedResume(opts.Resume, cfg, prog, shapes, spec, ctx, cfg.Training.BatchTokens)
		if err != nil {
			return TrainResult{}, err
		}
		resume = &plan
		if plan.loaded.Manifest.Sampler.Counter == nil {
			return TrainResult{}, fmt.Errorf("checkpoint has no direct sampler state; replay-only core checkpoints cannot resume production DDP")
		}
		if err = loader.Restore(*plan.loaded.Manifest.Sampler.Counter); err != nil {
			return TrainResult{}, err
		}
		if plan.StartOptimizerAttempt >= uint64(steps) {
			return TrainResult{}, fmt.Errorf("resume attempt %d is at/after configured steps %d", plan.StartOptimizerAttempt, steps)
		}
		start = int(plan.StartOptimizerAttempt)
		sched, steps, err = schedulerForResume(plan.loaded.Manifest.Schedule, steps)
		if err != nil {
			return TrainResult{}, err
		}
		if err = early.restoreResumeSnapshot(plan.loaded.Manifest.EarlyStop); err != nil {
			return TrainResult{}, err
		}
		weights, err = LoadDistributedResumeModelWeights(plan, shapes)
		if err != nil {
			return TrainResult{}, err
		}
	} else if opts.SafetensorsLoad != "" {
		weights, err = loadDistributedWeightsOnlyWarmStart(opts.SafetensorsLoad, shapes)
		if err != nil {
			return TrainResult{}, err
		}
	}
	// Control flow must agree even when per-host artifact paths differ.
	controlHash, err := hashJSON64(struct {
		Start, Steps, CheckpointEvery, LogEvery, ValEvery int
		Resume, Export                                    bool
	}{start, steps, opts.CheckpointEvery, opts.LogEvery, opts.ValEvery, resume != nil, opts.SafetensorsPath != ""})
	if err != nil {
		return TrainResult{}, err
	}
	if err = group.ValidateInitializationAgreement([]gpu.InitializationAgreementField{{Name: "orchestration", Value: controlHash}}); err != nil {
		return TrainResult{}, err
	}
	trainer, err := initGPUTrainerWithDistributedContext(prog, cfg, weights, nil, ctx)
	if err != nil {
		return TrainResult{}, err
	}
	defer trainer.CloseTrainer()
	if resume != nil {
		if err = RestoreDistributedResumableTrainerState(trainer, *resume); err != nil {
			return TrainResult{}, err
		}
	}
	if err = trainer.(gpuTrainingStepSetter).SetTrainingStepGPU(start * ctx.AccumulationSteps); err != nil {
		return TrainResult{}, err
	}
	var telemetry *telemetryRuntime
	var val *data.ValSet
	var progress *trainingProgressFile
	if group.Rank() == 0 {
		telemetry, err = newTelemetryRuntime(opts.PProfAddr, opts.TelemetryOut)
		if err != nil {
			return TrainResult{}, err
		}
		defer telemetry.Close()
		progress, err = newTrainingProgressFile()
		if err != nil {
			return TrainResult{}, err
		}
		val, _, err = loadTrainingValidationSet(cfg, pattern, cfg.Name, cfg.Training.Seed, cfg.Training.BatchTokens, cfg.SeqLen)
		if err != nil {
			return TrainResult{}, err
		}
		fmt.Printf("[%s] DDP backend=%s world=%d rank=0 local_batch_tokens=%d accumulation=%d global_batch_tokens=%d dataset=%s start_attempt=%d\n", cfg.Name, backend, group.WorldSize(), cfg.Training.BatchTokens, ctx.AccumulationSteps, cfg.Training.BatchTokens*ctx.AccumulationSteps*group.WorldSize(), loader.State().DatasetID, start)
	}
	return runDDPAttempts(cfg, pattern, opts, trainer, ctx, loader, prog, shapes, sched, steps, start, early, val, telemetry, progress)
}

func runDDPAttempts(cfg *ArchConfig, pattern string, opts TrainOptions, trainer GPUTrainer, ctx *DistributedTrainerContext, loader *data.DistributedLoader, prog *arch.Program, shapes []WeightShape, sched trainingScheduler, steps, start int, early *earlyStopState, val *data.ValSet, telemetry *telemetryRuntime, progress *trainingProgressFile) (TrainResult, error) {
	group := ctx.GroupRuntime
	root := group.Rank() == 0
	batchTokens, seqLen := cfg.Training.BatchTokens, cfg.SeqLen
	batchSize := batchTokens / seqLen
	logEvery := effectiveTrainEvery(opts.LogEvery, "MIXLAB_LOG_EVERY", 100)
	valEvery := effectiveTrainEvery(opts.ValEvery, "MIXLAB_VAL_EVERY", 100)
	// Environment-derived cadence must also agree before entering the loop.
	if err := group.ValidateInitializationAgreement([]gpu.InitializationAgreementField{{Name: "log_every", Value: uint64(logEvery)}, {Name: "val_every", Value: uint64(valEvery)}}); err != nil {
		return TrainResult{}, err
	}
	result := TrainResult{Name: cfg.Name, LastUnmaskedLoss: math.NaN(), LastValLoss: math.NaN()}
	began := time.Now()
	var last objectiveBatch
	var globalTokens, globalExamples uint64
	if start > 0 {
		m, err := resolveDistributedResumeManifest(opts.Resume)
		if err != nil {
			return result, err
		}
		globalTokens, globalExamples = m.EffectiveGlobalTokens, m.EffectiveGlobalExamples
	}
	for step := start; step < steps; step++ {
		lr := sched.At(step)
		numerator, denominator := 0.0, 0.0
		for micro := 0; micro < ctx.AccumulationSteps; micro++ {
			b, err := loader.NextBatch(batchTokens)
			if err != nil {
				return result, err
			}
			prepared, err := prepareObjectiveBatch(cfg, trainBatchFromDataBatch(b, nil), step, arch.ObjectiveCausal)
			if err != nil {
				return result, err
			}
			last = prepared
			if err = submitPreparedStepGPU(trainer, prepared, batchSize, seqLen, lr); err != nil {
				return result, fmt.Errorf("DDP attempt %d microstep %d: %w", step, micro, err)
			}
			v, err := trainer.CollectLossGPU()
			if err != nil {
				return result, err
			}
			numerator += float64(v) * float64(prepared.lossNormalizer)
			denominator += float64(prepared.lossNormalizer)
		}
		stats, err := readOptimizerStats(trainer)
		if err != nil {
			return result, err
		}
		if stats.AttemptedSteps != uint64(step+1) {
			return result, fmt.Errorf("DDP optimizer attempt counter %d, expected %d", stats.AttemptedSteps, step+1)
		}
		if stats.ConsecutiveSkipped >= 3 {
			return result, fmt.Errorf("DDP stopped after %d consecutive skipped updates", stats.ConsecutiveSkipped)
		}
		meanNumerator, err := distributedMeanLoss(group, numerator)
		if err != nil {
			return result, err
		}
		meanDenominator, err := distributedMeanLoss(group, denominator)
		if err != nil {
			return result, err
		}
		loss := meanNumerator / max(1, meanDenominator)
		globalTokens += uint64(meanDenominator * float64(group.WorldSize()))
		globalExamples += uint64(ctx.AccumulationSteps * batchSize * group.WorldSize())
		if math.IsNaN(loss) || math.IsInf(loss, 0) {
			return result, fmt.Errorf("DDP non-finite loss at attempt %d", step+1)
		}
		if step == start {
			result.FirstLoss = loss
		}
		result.LastLoss = loss
		stop := false
		var rootErr error
		if root && shouldRunTrainingValidationStep(cfg.Training, step, steps, valEvery) && val != nil {
			result.LastValLoss, rootErr = meanValidationLoss(val, trainer, batchSize, seqLen)
			if rootErr == nil && (math.IsNaN(result.LastValLoss) || math.IsInf(result.LastValLoss, 0)) {
				rootErr = fmt.Errorf("non-finite validation loss")
			}
			result.HasValLoss = rootErr == nil
			if rootErr == nil {
				stop, _ = early.observe(step, result.LastValLoss)
				stop = stop || (cfg.Training.TargetValLoss > 0 && result.LastValLoss <= cfg.Training.TargetValLoss)
			}
		}
		logStep := step == start || step == steps-1 || stop || (logEvery > 0 && (step+1)%logEvery == 0)
		if root && rootErr == nil {
			rootErr = progress.update(step+1, stats.CommittedSteps)
			if logStep && rootErr == nil {
				metrics, e := trainer.(*mlxGPUTrainer).DistributedStepTelemetryGPU()
				rootErr = e
				if e == nil {
					// Runtime timings are attempt-local; these counters span resumes.
					metrics.Microsteps = uint64(step+1) * uint64(ctx.AccumulationSteps)
					metrics.OptimizerAttempts = stats.AttemptedSteps
					metrics.EffectiveGlobalTokens = globalTokens
					validTokens := uint64(meanDenominator * float64(group.WorldSize()))
					if metrics.EffectiveTokensPerUpdate > 0 {
						metrics.GlobalTokensPerSec *= float64(validTokens) / float64(metrics.EffectiveTokensPerUpdate)
					}
					metrics.EffectiveTokensPerUpdate = validTokens
					telemetry.state.update(telemetryUpdate{Model: cfg.Name, Step: step + 1, TotalSteps: steps, Loss: loss, HasLoss: true, ValLoss: result.LastValLoss, HasValLoss: result.HasValLoss, LR: lr, Objective: arch.ObjectiveCausal, SeqLen: seqLen, BatchTokens: batchTokens, Elapsed: time.Since(began), TokensPerSec: metrics.GlobalTokensPerSec, Distributed: metrics, OptimizerSteps: stats.CommittedSteps, SkippedOptimizerSteps: stats.SkippedSteps, ConsecutiveSkipped: stats.ConsecutiveSkipped, OptimizerStepSkipped: stats.LastStepSkipped})
					fmt.Printf("[%s] attempt %d/%d loss=%.6f lr=%g committed=%d skipped=%d\n", cfg.Name, step+1, steps, loss, lr, stats.CommittedSteps, stats.SkippedSteps)
					fmt.Println(formatTelemetryLine(telemetry.state.snapshot(false)))
					rootErr = telemetry.writeSnapshot(false)
				}
			}
		}
		stop, err = distributedRootDecision(group, stop, rootErr)
		if err != nil {
			return result, err
		}
		if opts.CheckpointEvery > 0 && ((step+1)%opts.CheckpointEvery == 0 || step == steps-1 || stop) {
			schedule, err := resumeScheduleFrom(cfg.Training, sched, steps)
			if err != nil {
				return result, err
			}
			counter := loader.State()
			_, _, err = writeDistributedResumableCheckpoint(cfg, trainer, shapes, opts.CheckpointDir, distributedResumableCheckpointContext{Control: group, TrainPattern: pattern, DatasetHash: ctx.DatasetHash, Program: prog, Schedule: schedule, EarlyStop: early, LocalBatchTokens: batchTokens, AccumulationSteps: ctx.AccumulationSteps, Sampler: distributedResumeSamplerState{Counter: &counter, Epoch: int(counter.Epoch), LocalMicrostepsConsumed: uint64((step + 1) * ctx.AccumulationSteps)}, EffectiveGlobalTokens: globalTokens, EffectiveGlobalExamples: globalExamples})
			if err != nil {
				return result, err
			}
		}
		if stop {
			break
		}
	}
	var rootErr error
	if root {
		v, err := evaluateObjectiveTrainingLossGPU(trainer, last, batchSize, seqLen)
		rootErr = err
		result.LastUnmaskedLoss = float64(v)
		if rootErr == nil {
			_, rootErr = exportTrainingSafetensorsArtifacts(cfg, trainer, shapes, opts, nil)
		}
	}
	if _, err := distributedRootDecision(group, false, rootErr); err != nil {
		return result, err
	}
	result.Elapsed = time.Since(began)
	result.Delta = result.FirstLoss - result.LastLoss
	if root {
		fmt.Println(result.FormatSummary())
	}
	return result, nil
}
