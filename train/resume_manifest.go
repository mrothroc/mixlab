package train

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/mrothroc/mixlab/gpu"
)

const resumeCheckpointFormat = "mixlab_resume_v1"

type resumeTensorRef struct {
	Name        string                 `json:"name"`
	Kind        gpu.OptimizerStateKind `json:"kind,omitempty"`
	WeightIndex int                    `json:"weight_index"`
	Shape       []int                  `json:"shape"`
}

type resumeSchedule struct {
	Kind               string          `json:"kind"`
	OriginalTotalSteps int             `json:"original_total_steps"`
	Standard           *LRSchedule     `json:"standard,omitempty"`
	Phases             []TrainingPhase `json:"phases,omitempty"`
	WarmdownSteps      int             `json:"warmdown_steps,omitempty"`
	MinLRFraction      float32         `json:"min_lr_fraction,omitempty"`
	ExtensionPolicy    string          `json:"extension_policy"`
}

type resumeEarlyStop struct {
	Enabled  bool    `json:"enabled"`
	Best     float64 `json:"best,omitempty"`
	HaveBest bool    `json:"have_best,omitempty"`
	Stale    int     `json:"stale,omitempty"`
}

type resumeManifest struct {
	Format              string                    `json:"format"`
	GlobalStep          int                       `json:"global_step"`
	ModelFile           string                    `json:"model_file"`
	SWAFile             string                    `json:"swa_file,omitempty"`
	StateFile           string                    `json:"state_file"`
	ConfigHash          string                    `json:"config_hash"`
	OptimizerHash       string                    `json:"optimizer_hash"`
	DatasetHash         string                    `json:"dataset_hash"`
	TrainPattern        string                    `json:"train_pattern"`
	Schedule            resumeSchedule            `json:"schedule"`
	Optimizer           gpu.TrainerOptimizerStats `json:"optimizer"`
	OptimizerTensors    []resumeTensorRef         `json:"optimizer_tensors"`
	SWATensors          []resumeTensorRef         `json:"swa_tensors,omitempty"`
	Data2VecTensors     []resumeTensorRef         `json:"data2vec_tensors,omitempty"`
	EarlyStop           resumeEarlyStop           `json:"early_stop"`
	CheckpointSizeBytes int64                     `json:"checkpoint_size_bytes,omitempty"`
	ManifestPath        string                    `json:"-"`
}

type resumeLoadedState struct {
	Manifest  resumeManifest
	ModelPath string
	Trainer   gpu.TrainerStateSnapshot
	SWA       [][]float32
	Data2Vec  [][]float32
}

type resumedScheduler struct {
	saved resumeSchedule
}

func (s resumedScheduler) At(step int) float32 {
	if s.saved.Standard != nil {
		if step <= s.saved.OriginalTotalSteps {
			return s.saved.Standard.At(step)
		}
		return s.saved.Standard.At(s.saved.OriginalTotalSteps)
	}
	return 0
}

func resumeScheduleFrom(spec TrainingSpec, scheduler trainingScheduler, total int) (resumeSchedule, error) {
	out := resumeSchedule{
		OriginalTotalSteps: total,
		ExtensionPolicy:    "original_then_floor",
	}
	switch s := scheduler.(type) {
	case LRSchedule:
		copy := s
		out.Kind = "cosine"
		out.Standard = &copy
	case phaseSchedule:
		out.Kind = "phases"
		out.Phases = append([]TrainingPhase(nil), spec.Phases...)
		out.WarmdownSteps = spec.WarmdownSteps
		out.MinLRFraction = spec.MinLRFraction
	default:
		return resumeSchedule{}, fmt.Errorf("unsupported scheduler type %T", scheduler)
	}
	return out, nil
}

func schedulerForResume(saved resumeSchedule, configuredSteps int) (trainingScheduler, int, error) {
	if saved.OriginalTotalSteps <= 0 {
		return nil, 0, fmt.Errorf("checkpoint has invalid original total steps %d", saved.OriginalTotalSteps)
	}
	if configuredSteps < saved.OriginalTotalSteps {
		return nil, 0, fmt.Errorf("configured training steps %d are below checkpoint schedule horizon %d", configuredSteps, saved.OriginalTotalSteps)
	}
	if saved.Kind == "phases" && configuredSteps != saved.OriginalTotalSteps {
		return nil, 0, fmt.Errorf("phase-schedule extension is not supported in resumable checkpoint v1; append a separate standard training run or keep total steps at %d", saved.OriginalTotalSteps)
	}
	if saved.Kind == "cosine" && saved.Standard == nil {
		return nil, 0, fmt.Errorf("checkpoint cosine schedule is missing parameters")
	}
	if saved.Kind != "cosine" && saved.Kind != "phases" {
		return nil, 0, fmt.Errorf("checkpoint has unsupported schedule kind %q", saved.Kind)
	}
	if saved.Kind == "phases" {
		return newPhaseSchedule(saved.Phases, saved.WarmdownSteps, saved.MinLRFraction), configuredSteps, nil
	}
	return resumedScheduler{saved: saved}, configuredSteps, nil
}

func resumeConfigHash(cfg *ArchConfig) (string, error) {
	if cfg == nil {
		return "", fmt.Errorf("nil config")
	}
	clone := *cfg
	clone.Training = cfg.Training
	if len(clone.Training.Phases) == 0 {
		clone.Training.Steps = 0
	}
	blob, err := json.Marshal(clone)
	if err != nil {
		return "", err
	}
	return sha256Hex(blob), nil
}

func optimizerSpecHash(spec gpu.TrainerOptimizerSpec) (string, error) {
	blob, err := json.Marshal(spec)
	if err != nil {
		return "", err
	}
	return sha256Hex(blob), nil
}

func trainingDatasetHash(pattern string) (string, error) {
	files, err := filepath.Glob(pattern)
	if err != nil {
		return "", err
	}
	if len(files) == 0 {
		return "", fmt.Errorf("no shard files matched %q", pattern)
	}
	sort.Strings(files)
	type fileIdentity struct {
		Path    string `json:"path"`
		Size    int64  `json:"size"`
		ModTime int64  `json:"mod_time_unix_nano"`
	}
	ids := make([]fileIdentity, 0, len(files))
	for _, path := range files {
		info, err := os.Stat(path)
		if err != nil {
			return "", err
		}
		abs, err := filepath.Abs(path)
		if err != nil {
			return "", err
		}
		ids = append(ids, fileIdentity{Path: filepath.Clean(abs), Size: info.Size(), ModTime: info.ModTime().UnixNano()})
	}
	blob, err := json.Marshal(ids)
	if err != nil {
		return "", err
	}
	return sha256Hex(blob), nil
}

func sha256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func resumeManifestFilename(step int) string {
	return fmt.Sprintf("step_%06d.resume.json", step)
}

func resumeStateFilename(step int) string {
	return fmt.Sprintf("step_%06d.state.safetensors", step)
}

func resolveResumeManifest(path string) (resumeManifest, error) {
	info, err := os.Stat(path)
	if err != nil {
		return resumeManifest{}, fmt.Errorf("inspect resume path %q: %w", path, err)
	}
	if info.IsDir() {
		matches, err := filepath.Glob(filepath.Join(path, "step_*.resume.json"))
		if err != nil {
			return resumeManifest{}, err
		}
		var candidates []resumeManifest
		for _, match := range matches {
			manifest, err := readResumeManifest(match)
			if err == nil && resumeManifestFilesExist(manifest) {
				candidates = append(candidates, manifest)
			}
		}
		if len(candidates) == 0 {
			return resumeManifest{}, fmt.Errorf("resume directory %q contains no complete %s checkpoints", path, resumeCheckpointFormat)
		}
		sort.Slice(candidates, func(i, j int) bool { return candidates[i].GlobalStep > candidates[j].GlobalStep })
		return candidates[0], nil
	}
	if strings.HasSuffix(path, ".resume.json") {
		manifest, err := readResumeManifest(path)
		if err != nil {
			return resumeManifest{}, err
		}
		if !resumeManifestFilesExist(manifest) {
			return resumeManifest{}, fmt.Errorf("resume manifest %q references missing companion files", path)
		}
		return manifest, nil
	}
	dir := filepath.Dir(path)
	matches, err := filepath.Glob(filepath.Join(dir, "step_*.resume.json"))
	if err != nil {
		return resumeManifest{}, err
	}
	base := filepath.Base(path)
	for _, match := range matches {
		manifest, err := readResumeManifest(match)
		if err == nil && (manifest.ModelFile == base || manifest.StateFile == base) && resumeManifestFilesExist(manifest) {
			return manifest, nil
		}
	}
	return resumeManifest{}, fmt.Errorf("resume file %q is not referenced by a complete resume manifest", path)
}

func readResumeManifest(path string) (resumeManifest, error) {
	blob, err := os.ReadFile(path)
	if err != nil {
		return resumeManifest{}, err
	}
	var manifest resumeManifest
	if err := json.Unmarshal(blob, &manifest); err != nil {
		return resumeManifest{}, fmt.Errorf("parse resume manifest %q: %w", path, err)
	}
	if manifest.Format != resumeCheckpointFormat {
		return resumeManifest{}, fmt.Errorf("resume manifest %q has format %q, want %q", path, manifest.Format, resumeCheckpointFormat)
	}
	if manifest.GlobalStep <= 0 || manifest.ModelFile == "" || manifest.StateFile == "" {
		return resumeManifest{}, fmt.Errorf("resume manifest %q is incomplete", path)
	}
	if manifest.Optimizer.AttemptedSteps != uint64(manifest.GlobalStep) {
		return resumeManifest{}, fmt.Errorf(
			"resume manifest %q has optimizer attempted_steps=%d, want global_step=%d",
			path, manifest.Optimizer.AttemptedSteps, manifest.GlobalStep,
		)
	}
	if manifest.Optimizer.CommittedSteps > manifest.Optimizer.AttemptedSteps ||
		manifest.Optimizer.SkippedSteps != manifest.Optimizer.AttemptedSteps-manifest.Optimizer.CommittedSteps {
		return resumeManifest{}, fmt.Errorf(
			"resume manifest %q has inconsistent optimizer counters: committed=%d skipped=%d attempted=%d",
			path, manifest.Optimizer.CommittedSteps, manifest.Optimizer.SkippedSteps, manifest.Optimizer.AttemptedSteps,
		)
	}
	manifest.ManifestPath = path
	return manifest, nil
}

func resumeManifestFilesExist(manifest resumeManifest) bool {
	dir := filepath.Dir(manifest.ManifestPath)
	for _, name := range []string{manifest.ModelFile, manifest.StateFile} {
		if name == "" {
			return false
		}
		if info, err := os.Stat(filepath.Join(dir, name)); err != nil || info.IsDir() {
			return false
		}
	}
	return true
}
