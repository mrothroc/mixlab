package train

import (
	"fmt"

	"github.com/mrothroc/mixlab/data"
)

type resumeRunSetup struct {
	Scheduler          trainingScheduler
	Steps              int
	StartStep          int
	CheckpointSchedule resumeSchedule
	Loaded             *resumeLoadedState
}

func prepareResumeRun(cfg *ArchConfig, trainPattern, resumePath string, earlyStop *earlyStopState) (resumeRunSetup, error) {
	scheduler, steps := buildTrainingScheduler(cfg.Training)
	checkpointSchedule, err := resumeScheduleFrom(cfg.Training, scheduler, steps)
	if err != nil {
		return resumeRunSetup{}, fmt.Errorf("snapshot training schedule: %w", err)
	}
	setup := resumeRunSetup{Scheduler: scheduler, Steps: steps, CheckpointSchedule: checkpointSchedule}
	if resumePath == "" {
		return setup, nil
	}
	manifest, err := resolveResumeManifest(resumePath)
	if err != nil {
		return resumeRunSetup{}, err
	}
	currentConfigHash, err := resumeConfigHash(cfg)
	if err != nil {
		return resumeRunSetup{}, fmt.Errorf("hash current config: %w", err)
	}
	if currentConfigHash != manifest.ConfigHash {
		return resumeRunSetup{}, fmt.Errorf("training config does not match resumable checkpoint (checkpoint=%s current=%s)", manifest.ConfigHash, currentConfigHash)
	}
	currentDatasetHash, err := trainingDatasetHash(trainPattern)
	if err != nil {
		return resumeRunSetup{}, fmt.Errorf("hash current training dataset: %w", err)
	}
	if currentDatasetHash != manifest.DatasetHash {
		return resumeRunSetup{}, fmt.Errorf("training dataset does not match resumable checkpoint (checkpoint=%s current=%s)", manifest.DatasetHash, currentDatasetHash)
	}
	scheduler, steps, err = schedulerForResume(manifest.Schedule, cfg.Training.TotalSteps())
	if err != nil {
		return resumeRunSetup{}, err
	}
	if manifest.GlobalStep >= steps {
		return resumeRunSetup{}, fmt.Errorf("checkpoint is at step %d, at or after configured training steps %d; raise training.steps to extend", manifest.GlobalStep, steps)
	}
	loaded, err := loadResumeState(manifest)
	if err != nil {
		return resumeRunSetup{}, fmt.Errorf("load resumable checkpoint: %w", err)
	}
	if !allFiniteState(loaded.Trainer) {
		return resumeRunSetup{}, fmt.Errorf("resumable checkpoint optimizer state contains non-finite values")
	}
	if err := earlyStop.restoreResumeSnapshot(manifest.EarlyStop); err != nil {
		return resumeRunSetup{}, err
	}
	setup.Scheduler = scheduler
	setup.Steps = steps
	setup.StartStep = manifest.GlobalStep
	setup.CheckpointSchedule = manifest.Schedule
	setup.Loaded = &loaded
	return setup, nil
}

func replayTrainingLoader(loader *data.Loader, steps, batchTokens, seqLen int) error {
	for step := 0; step < steps; step++ {
		if _, err := loader.NextBatchDetailed(batchTokens, seqLen); err != nil {
			return fmt.Errorf("replay loader batch %d/%d: %w", step+1, steps, err)
		}
	}
	return nil
}
