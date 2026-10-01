package train

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
)

func checkGridGeometry(cfg *ArchConfig, g data.GridGeometry, targets bool) error {
	i := cfg.InputAdapter
	if g.Channels != i.Channels || g.Height != i.Height || g.Width != i.Width || (targets && g.TargetChannels != cfg.DenseRegression.TargetChannels) {
		return fmt.Errorf("grid dataset geometry %+v does not match model", g)
	}
	return nil
}

// The grid driver owns record iteration and pixel metrics; optimizer execution,
// numerical protection, scheduling and checkpoint serialization remain shared.
func runGridTrain(cfg *ArchConfig, manifest string, opts TrainOptions) (TrainResult, error) {
	result := TrainResult{Name: cfg.Name, LastUnmaskedLoss: math.NaN()}
	if err := validateRunTrainOptions(opts); err != nil {
		return result, err
	}
	initPath, err := gridWarmStartPath(cfg, opts)
	if err != nil {
		return result, err
	}
	if opts.DoFullEval || opts.managed != nil || opts.SWAStartOverride != nil || opts.SWADecayOverride != nil || opts.SWAIntervalOverride != nil || (opts.Quantize != "" && opts.Quantize != "none") {
		return result, fmt.Errorf("dense_regression does not yet support full BPB eval, managed training, SWA, or quantization")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	ds, err := data.OpenGridDataset(manifest, "train")
	if err != nil {
		return result, err
	}
	defer func() { _ = ds.Close() }()
	if err = checkGridGeometry(cfg, ds.Geometry, true); err != nil {
		return result, err
	}
	m, err := data.LoadDatasetManifest(manifest)
	if err != nil {
		return result, err
	}
	var val *data.GridDataset
	if _, ok := m.Splits["val"]; ok {
		val, err = data.OpenGridDataset(manifest, "val")
		if err != nil {
			return result, err
		}
		defer func() { _ = val.Close() }()
		if err = checkGridGeometry(cfg, val.Geometry, true); err != nil {
			return result, err
		}
	}
	if val == nil && (cfg.Training.EarlyStop != nil || cfg.Training.TargetValLoss > 0) {
		return result, fmt.Errorf("dense_regression early stopping requires a val split")
	}
	prog, err := arch.BuildGridIRProgram(cfg, true)
	if err != nil {
		return result, err
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		return result, err
	}
	if _, err = configureMLXMemoryLimits(cfg.Name); err != nil {
		return result, err
	}
	stopper := newEarlyStopState(cfg.Training.EarlyStop)
	setup, err := prepareResumeRun(cfg, manifest, opts.Resume, stopper)
	if err != nil {
		return result, err
	}
	var weights [][]float32
	if setup.Loaded != nil {
		weights, _, err = loadGridWeights(setup.Loaded.ModelPath, cfg, shapes, nil)
		if err != nil {
			return result, err
		}
	} else if initPath != "" {
		var report gridLoadReport
		weights, report, err = loadGridWeights(initPath, cfg, shapes, cfg.Training.InitAllowMissing)
		if err != nil {
			return result, err
		}
		fmt.Printf("  [%s] warm start: loaded=%v new=%v unexpected=[]\n", cfg.Name, report.Loaded, report.New)
	}
	trainer, err := initGPUTrainer(prog, cfg, weights, opts.OptimizerOverride)
	if err != nil {
		return result, err
	}
	defer trainer.CloseTrainer()
	if setup.Loaded != nil {
		if err = restoreResumableTrainerState(trainer, *setup.Loaded); err != nil {
			return result, err
		}
	}
	progress, err := newTrainingProgressFile()
	if err != nil {
		return result, err
	}
	sched, steps := setup.Scheduler, setup.Steps
	batchSize := cfg.Training.BatchSize
	logEvery := opts.LogEvery
	if logEvery <= 0 {
		logEvery = 10
	}
	memLogEvery := effectiveTrainEvery(0, mlxMemLogEveryEnv, 0)
	clearCacheEvery := effectiveTrainEvery(0, mlxClearCacheEveryEnv, 0)
	valEvery := opts.ValEvery
	if valEvery <= 0 {
		valEvery = cfg.Training.ValEverySteps
	}
	if valEvery <= 0 {
		valEvery = 100
	}
	best := math.Inf(1)
	sampler, err := data.NewGridSampler(ds.Len(), cfg.Training.Seed)
	if err != nil {
		return result, err
	}
	stopped := false
	if setup.Loaded != nil {
		best, stopped, err = restoreGridRunState(setup.Loaded.Manifest, sampler, shapes, batchSize, valEvery, &result)
		if err != nil {
			return result, err
		}
	}
	var datasetHash string
	if opts.CheckpointDir != "" {
		if setup.Loaded != nil {
			datasetHash = setup.Loaded.Manifest.DatasetHash
		} else {
			datasetHash, err = data.GridDatasetIdentity(manifest)
			if err != nil {
				return result, err
			}
		}
	}
	bestArtifact, err := newGridBestArtifact(cfg, opts.CheckpointDir, datasetHash, setup.Loaded != nil)
	if err != nil {
		return result, err
	}
	save := func(path string) error {
		w, e := readTrainerWeights(trainer)
		if e != nil {
			return e
		}
		if e = os.MkdirAll(filepath.Dir(path), 0755); e != nil {
			return e
		}
		return exportSafetensors(path, cfg, shapes, w)
	}
	validate := func(step int) (bool, error) {
		if val == nil {
			return false, nil
		}
		start := time.Now()
		metrics, e := evaluateGridDataset(cfg, trainer, val)
		if e != nil {
			return false, e
		}
		result.HasValLoss = true
		result.LastValLoss = metrics.MaskedMSE()
		fmt.Printf("  [%s] validation step=%d %s duration=%s\n", cfg.Name, step, metrics.String(cfg.DenseRegression.EffectiveMetricScale()), time.Since(start).Round(time.Millisecond))
		if metrics.MaskedRMSE() < best {
			best = metrics.MaskedRMSE()
			if e = bestArtifact.save(cfg, trainer, shapes, best); e != nil {
				return false, e
			}
		}
		stop, reason := stopper.observe(step, metrics.MaskedRMSE())
		if stop {
			fmt.Printf("  [%s] early stop: %s\n", cfg.Name, reason)
		}
		return stop || (cfg.Training.TargetValLoss > 0 && metrics.MaskedRMSE() <= cfg.Training.TargetValLoss), nil
	}
	// Validate the whole split before any update or best-checkpoint selection.
	if setup.Loaded == nil {
		stopped, err = validate(0)
		if err != nil {
			return result, err
		}
	}
	var augmenter data.GridAugmenter
	start := time.Now()
	var compute time.Duration
	records := 0
	fmt.Printf("  [%s] dense regression: records=%d batch_size=%d grid=%dx%dx%d epochs are complete shuffled passes\n", cfg.Name, ds.Len(), batchSize, ds.Geometry.Channels, ds.Geometry.Height, ds.Geometry.Width)
	var frozen []string
	for _, shape := range shapes {
		if shape.Frozen {
			frozen = append(frozen, shape.Name)
		}
	}
	fmt.Printf("  [%s] trainable_weights=%d frozen_or_unreachable=%d names=%v\n", cfg.Name, len(gridTrainableNames(shapes)), len(frozen), frozen)
	for step := setup.StartStep; step < steps && !stopped; step++ {
		indices, epoch, occurrence, e := sampler.Next(batchSize)
		if e != nil {
			return result, e
		}
		b, e := ds.ReadBatch(indices, batchSize)
		if e != nil {
			return result, e
		}
		if a := cfg.Training.GridAugmentation; a != nil && a.Dihedral {
			if e = augmenter.Apply(b, cfg.Training.Seed, epoch, occurrence); e != nil {
				return result, e
			}
		}
		tick := time.Now()
		if e = submitPreparedStepGPU(trainer, objectiveBatch{grid: &b}, batchSize, 0, sched.At(step)); e != nil {
			return result, e
		}
		loss, e := trainer.CollectLossGPU()
		if e != nil {
			return result, fmt.Errorf("grid step %d: %w", step, e)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			return result, fmt.Errorf("grid step %d: non-finite loss", step)
		}
		if step > setup.StartStep {
			compute += time.Since(tick)
			records += b.Count
		}
		if step == 0 {
			result.FirstLoss = float64(loss)
		}
		result.LastLoss = float64(loss)
		stats, e := readOptimizerStats(trainer)
		if e != nil {
			return result, e
		}
		if e = progress.update(step, stats.CommittedSteps); e != nil {
			return result, e
		}
		if stats.LastStepSkipped {
			fmt.Printf("  [%s] step=%d optimizer update skipped: total=%d consecutive=%d\n", cfg.Name, step+1, stats.SkippedSteps, stats.ConsecutiveSkipped)
		}
		if step == 0 || (step+1)%logEvery == 0 || step+1 == steps {
			rate := 0.0
			if compute > 0 {
				rate = float64(records) / compute.Seconds()
			}
			fmt.Printf("  [%s] step=%d/%d epoch=%d loss=%.7g lr=%g records/s=%.2f pixels/s=%.0f\n", cfg.Name, step+1, steps, epoch, float64(loss), sched.At(step), rate, rate*float64(ds.Geometry.Height*ds.Geometry.Width))
			if opts.telemetry != nil {
				opts.telemetry.state.update(telemetryUpdate{
					Model: cfg.Name, Step: step + 1, TotalSteps: steps, Loss: float64(loss), HasLoss: true,
					ValLoss: result.LastValLoss, HasValLoss: result.HasValLoss, LR: sched.At(step),
					Objective: arch.ObjectiveDenseRegression, Elapsed: time.Since(start),
					Extra:          map[string]float64{"records_per_sec": rate, "pixels_per_sec": rate * float64(ds.Geometry.Height*ds.Geometry.Width), "batch_size": float64(batchSize)},
					OptimizerSteps: stats.CommittedSteps, SkippedOptimizerSteps: stats.SkippedSteps,
					ConsecutiveSkipped: stats.ConsecutiveSkipped, OptimizerStepSkipped: stats.LastStepSkipped,
				})
				if e = opts.telemetry.writeSnapshot(false); e != nil {
					return result, e
				}
			}
			handleMLXMemoryControls(cfg.Name, step, memLogEvery, clearCacheEvery, opts.telemetry)
		}
		if (step+1)%valEvery == 0 || step+1 == steps {
			stopped, e = validate(step + 1)
			if e != nil {
				return result, e
			}
		}
		if opts.CheckpointDir != "" && opts.CheckpointEvery > 0 && (step+1)%opts.CheckpointEvery == 0 {
			schedule, e := resumeScheduleFrom(cfg.Training, sched, steps)
			if e != nil {
				return result, e
			}
			ctx := resumableCheckpointContext{TrainPattern: manifest, Schedule: schedule, EarlyStop: stopper, DatasetHash: datasetHash, Grid: snapshotGridRun(sampler, shapes, best, result, valEvery, stopped)}
			if _, _, e = writeResumableCheckpoint(cfg, trainer, shapes, opts.CheckpointDir, step+1, ctx); e != nil {
				return result, e
			}
		}
	}
	if opts.SafetensorsPath != "" {
		if err = save(opts.SafetensorsPath); err != nil {
			return result, err
		}
	}
	result.Delta = result.FirstLoss - result.LastLoss
	result.Elapsed = time.Since(start)
	return result, nil
}
