package train

import (
	"fmt"
	"path/filepath"
)

type resumableCheckpointContext struct {
	TrainPattern string
	Schedule     resumeSchedule
	SWA          [][]float32
	Data2Vec     *data2VecTeacher
	EarlyStop    *earlyStopState
}

func writeResumableCheckpoint(
	cfg *ArchConfig,
	trainer any,
	shapes []WeightShape,
	dir string,
	step int,
	ctx resumableCheckpointContext,
) (safetensorsArtifacts, string, error) {
	stateReader, ok := trainer.(gpuTrainerStateReader)
	if !ok {
		return safetensorsArtifacts{}, "", fmt.Errorf("trainer does not support resumable optimizer-state checkpoints")
	}
	specReader, ok := trainer.(gpuOptimizerSpecReader)
	if !ok {
		return safetensorsArtifacts{}, "", fmt.Errorf("trainer does not expose its optimizer specification")
	}
	snapshot, err := stateReader.ReadTrainerState()
	if err != nil {
		return safetensorsArtifacts{}, "", fmt.Errorf("read optimizer state: %w", err)
	}
	if int(snapshot.Optimizer.AttemptedSteps) != step {
		return safetensorsArtifacts{}, "", fmt.Errorf("optimizer attempted steps=%d do not match checkpoint step=%d", snapshot.Optimizer.AttemptedSteps, step)
	}
	if !allFiniteState(snapshot) {
		return safetensorsArtifacts{}, "", fmt.Errorf("optimizer state contains non-finite values")
	}

	configHash, err := resumeConfigHash(cfg)
	if err != nil {
		return safetensorsArtifacts{}, "", fmt.Errorf("hash config: %w", err)
	}
	optimizerHash, err := optimizerSpecHash(specReader.OptimizerSpec())
	if err != nil {
		return safetensorsArtifacts{}, "", fmt.Errorf("hash optimizer plan: %w", err)
	}
	datasetHash, err := trainingDatasetHash(ctx.TrainPattern)
	if err != nil {
		return safetensorsArtifacts{}, "", fmt.Errorf("hash training dataset: %w", err)
	}

	artifacts, err := writeCheckpoint(cfg, trainer, shapes, dir, step, ctx.SWA)
	if err != nil {
		return safetensorsArtifacts{}, "", err
	}
	manifestPath := filepath.Join(dir, resumeManifestFilename(step))
	statePath := filepath.Join(dir, resumeStateFilename(step))
	manifest := resumeManifest{
		Format:        resumeCheckpointFormat,
		GlobalStep:    step,
		ModelFile:     filepath.Base(artifacts.FinalPath),
		StateFile:     filepath.Base(statePath),
		ConfigHash:    configHash,
		OptimizerHash: optimizerHash,
		DatasetHash:   datasetHash,
		TrainPattern:  ctx.TrainPattern,
		Schedule:      ctx.Schedule,
		Optimizer:     snapshot.Optimizer,
		EarlyStop:     ctx.EarlyStop.resumeSnapshot(),
	}
	if artifacts.SWAPath != "" {
		manifest.SWAFile = filepath.Base(artifacts.SWAPath)
	}

	tensors := make([]namedFloatTensor, 0, len(snapshot.Tensors)+len(ctx.SWA))
	for _, state := range snapshot.Tensors {
		name := optimizerStateTensorName(state.Kind, state.WeightIndex)
		shape := append([]int(nil), state.Shape...)
		tensors = append(tensors, namedFloatTensor{Name: name, Shape: shape, Data: state.Data})
		manifest.OptimizerTensors = append(manifest.OptimizerTensors, resumeTensorRef{
			Name: name, Kind: state.Kind, WeightIndex: state.WeightIndex, Shape: shape,
		})
	}
	if err := appendWeightTensors("swa", ctx.SWA, shapes, &tensors, &manifest.SWATensors); err != nil {
		return safetensorsArtifacts{}, "", err
	}
	if ctx.Data2Vec != nil {
		if err := appendWeightTensors("data2vec", ctx.Data2Vec.emaWeights, ctx.Data2Vec.shapes, &tensors, &manifest.Data2VecTensors); err != nil {
			return safetensorsArtifacts{}, "", err
		}
	}
	if len(tensors) == 0 {
		// Keep future stateless optimizer bundles valid safetensors artifacts.
		tensors = append(tensors, namedFloatTensor{Name: "resume_state", Shape: []int{1}, Data: []float32{float32(step)}})
	}
	if err := writeNamedFloatSafetensorsAtomic(statePath, tensors, map[string]string{
		"format": resumeCheckpointFormat,
		"step":   fmt.Sprintf("%d", step),
	}); err != nil {
		return safetensorsArtifacts{}, "", fmt.Errorf("write training state: %w", err)
	}
	manifest.CheckpointSizeBytes = checkpointBundleSize(manifestPath, manifest)
	if err := atomicWriteJSON(manifestPath, manifest); err != nil {
		return safetensorsArtifacts{}, "", fmt.Errorf("publish resume manifest: %w", err)
	}
	return artifacts, manifestPath, nil
}

func restoreResumableTrainerState(trainer any, loaded resumeLoadedState) error {
	restorer, ok := trainer.(gpuTrainerStateRestorer)
	if !ok {
		return fmt.Errorf("trainer does not support resumable optimizer-state restore")
	}
	specReader, ok := trainer.(gpuOptimizerSpecReader)
	if !ok {
		return fmt.Errorf("trainer does not expose its optimizer specification")
	}
	hash, err := optimizerSpecHash(specReader.OptimizerSpec())
	if err != nil {
		return err
	}
	if hash != loaded.Manifest.OptimizerHash {
		return fmt.Errorf("optimizer configuration does not match checkpoint (checkpoint=%s current=%s)", loaded.Manifest.OptimizerHash, hash)
	}
	if err := restorer.RestoreTrainerState(loaded.Trainer); err != nil {
		return fmt.Errorf("restore optimizer state: %w", err)
	}
	return nil
}
