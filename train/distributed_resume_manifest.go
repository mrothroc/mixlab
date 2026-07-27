package train

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/gpu"
)

const distributedResumeCheckpointFormat = "mixlab_distributed_resume_v1"

type distributedResumeTopology struct {
	RunID                string                       `json:"run_id"`
	GroupID              string                       `json:"group_id"`
	MembershipGeneration uint64                       `json:"membership_generation"`
	Backend              string                       `json:"backend"`
	OrderedMembers       []distributed.DDPGroupMember `json:"ordered_members"`
	MembersHash          string                       `json:"members_hash"`
	LaunchAttemptID      string                       `json:"launch_attempt_id"`
}

type distributedResumeSamplerState struct {
	Epoch                   int    `json:"epoch"`
	LocalBatchCursor        uint64 `json:"local_batch_cursor"`
	LocalMicrostepsConsumed uint64 `json:"local_microsteps_consumed"`
}

type distributedResumeManifest struct {
	Format                   string                        `json:"format"`
	GlobalOptimizerAttempt   uint64                        `json:"global_optimizer_attempt"`
	GlobalCommittedStep      uint64                        `json:"global_committed_step"`
	ModelFile                string                        `json:"model_file"`
	StateFile                string                        `json:"state_file"`
	Topology                 distributedResumeTopology     `json:"topology"`
	LocalBatchTokens         int                           `json:"local_batch_tokens"`
	AccumulationSteps        int                           `json:"accumulation_steps"`
	EffectiveGlobalTokens    uint64                        `json:"effective_global_tokens"`
	EffectiveGlobalExamples  uint64                        `json:"effective_global_examples"`
	ConfigHash               string                        `json:"config_hash"`
	ProgramHash              string                        `json:"program_hash"`
	WeightLayoutHash         string                        `json:"weight_layout_hash"`
	OptimizerHash            string                        `json:"optimizer_hash"`
	DatasetHash              string                        `json:"dataset_hash"`
	TrainPattern             string                        `json:"train_pattern"`
	Sampler                  distributedResumeSamplerState `json:"sampler"`
	Schedule                 resumeSchedule                `json:"schedule"`
	Optimizer                gpu.TrainerOptimizerStats     `json:"optimizer"`
	OptimizerTensors         []resumeTensorRef             `json:"optimizer_tensors"`
	EarlyStop                resumeEarlyStop               `json:"early_stop"`
	ExactSupportedWorldSizes []int                         `json:"exact_supported_world_sizes"`
	CheckpointSizeBytes      int64                         `json:"checkpoint_size_bytes,omitempty"`
	ManifestPath             string                        `json:"-"`
}

type distributedResumeLoadedState struct {
	Manifest  distributedResumeManifest
	ModelPath string
	Trainer   gpu.TrainerStateSnapshot
}

func distributedResumeManifestFilename(step uint64) string {
	return fmt.Sprintf("step_%06d.distributed.resume.json", step)
}

func distributedResumeStateFilename(step uint64) string {
	return fmt.Sprintf("step_%06d.distributed.state.safetensors", step)
}

func distributedResumeModelFilename(step uint64) string {
	return fmt.Sprintf("step_%06d.distributed.st", step)
}

func hashJSONHex(value any) (string, error) {
	blob, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	return sha256Hex(blob), nil
}

func resolveDistributedResumeManifest(path string) (distributedResumeManifest, error) {
	info, err := os.Stat(path)
	if err != nil {
		return distributedResumeManifest{}, fmt.Errorf("inspect distributed resume path %q: %w", path, err)
	}
	if info.IsDir() {
		matches, err := filepath.Glob(filepath.Join(path, "step_*.distributed.resume.json"))
		if err != nil {
			return distributedResumeManifest{}, err
		}
		candidates := make([]distributedResumeManifest, 0, len(matches))
		for _, match := range matches {
			manifest, readErr := readDistributedResumeManifest(match)
			if readErr == nil && distributedResumeManifestFilesExist(manifest) {
				candidates = append(candidates, manifest)
			}
		}
		if len(candidates) == 0 {
			return distributedResumeManifest{}, fmt.Errorf(
				"distributed resume directory %q contains no complete %s checkpoints",
				path,
				distributedResumeCheckpointFormat,
			)
		}
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].GlobalOptimizerAttempt > candidates[j].GlobalOptimizerAttempt
		})
		return candidates[0], nil
	}
	if !strings.HasSuffix(path, ".distributed.resume.json") {
		return distributedResumeManifest{}, fmt.Errorf(
			"distributed resume path %q must be a directory or .distributed.resume.json manifest",
			path,
		)
	}
	manifest, err := readDistributedResumeManifest(path)
	if err != nil {
		return distributedResumeManifest{}, err
	}
	if !distributedResumeManifestFilesExist(manifest) {
		return distributedResumeManifest{}, fmt.Errorf(
			"distributed resume manifest %q references missing companion files",
			path,
		)
	}
	return manifest, nil
}

func readDistributedResumeManifest(path string) (distributedResumeManifest, error) {
	blob, err := os.ReadFile(path)
	if err != nil {
		return distributedResumeManifest{}, err
	}
	var manifest distributedResumeManifest
	if err := json.Unmarshal(blob, &manifest); err != nil {
		return distributedResumeManifest{}, fmt.Errorf(
			"parse distributed resume manifest %q: %w",
			path,
			err,
		)
	}
	if err := validateDistributedResumeManifest(manifest); err != nil {
		return distributedResumeManifest{}, fmt.Errorf(
			"distributed resume manifest %q: %w",
			path,
			err,
		)
	}
	manifest.ManifestPath = path
	return manifest, nil
}

func validateDistributedResumeManifest(manifest distributedResumeManifest) error {
	if manifest.Format != distributedResumeCheckpointFormat {
		return fmt.Errorf(
			"format %q, want %q",
			manifest.Format,
			distributedResumeCheckpointFormat,
		)
	}
	if manifest.GlobalOptimizerAttempt == 0 ||
		manifest.ModelFile == "" ||
		manifest.StateFile == "" {
		return fmt.Errorf("checkpoint is incomplete")
	}
	if manifest.Optimizer.AttemptedSteps != manifest.GlobalOptimizerAttempt ||
		manifest.Optimizer.CommittedSteps != manifest.GlobalCommittedStep {
		return fmt.Errorf("optimizer counters do not match global counters")
	}
	if manifest.Optimizer.CommittedSteps > manifest.Optimizer.AttemptedSteps ||
		manifest.Optimizer.SkippedSteps !=
			manifest.Optimizer.AttemptedSteps-manifest.Optimizer.CommittedSteps {
		return fmt.Errorf("optimizer counters are inconsistent")
	}
	membership, err := distributed.NewDDPGroupMembership(
		manifest.Topology.RunID,
		manifest.Topology.GroupID,
		manifest.Topology.MembershipGeneration,
		manifest.Topology.Backend,
		manifest.Topology.OrderedMembers,
	)
	if err != nil {
		return fmt.Errorf("topology: %w", err)
	}
	if membership.MembersHash != manifest.Topology.MembersHash {
		return fmt.Errorf(
			"topology members hash=%q, want %q",
			manifest.Topology.MembersHash,
			membership.MembersHash,
		)
	}
	if manifest.Topology.LaunchAttemptID == "" {
		return fmt.Errorf("topology launch_attempt_id is required")
	}
	if manifest.LocalBatchTokens <= 0 || manifest.AccumulationSteps <= 0 {
		return fmt.Errorf("local batch tokens and accumulation steps must be positive")
	}
	if manifest.Sampler.LocalMicrostepsConsumed !=
		manifest.GlobalOptimizerAttempt*uint64(manifest.AccumulationSteps) {
		return fmt.Errorf(
			"sampler microsteps=%d, want optimizer attempts %d * accumulation %d",
			manifest.Sampler.LocalMicrostepsConsumed,
			manifest.GlobalOptimizerAttempt,
			manifest.AccumulationSteps,
		)
	}
	for name, value := range map[string]string{
		"config_hash":        manifest.ConfigHash,
		"program_hash":       manifest.ProgramHash,
		"weight_layout_hash": manifest.WeightLayoutHash,
		"optimizer_hash":     manifest.OptimizerHash,
		"dataset_hash":       manifest.DatasetHash,
	} {
		if value == "" {
			return fmt.Errorf("%s is required", name)
		}
	}
	if len(manifest.ExactSupportedWorldSizes) == 0 {
		return fmt.Errorf("exact_supported_world_sizes is required")
	}
	worldSupported := false
	for index, worldSize := range manifest.ExactSupportedWorldSizes {
		if worldSize <= 0 {
			return fmt.Errorf("exact_supported_world_sizes contains %d", worldSize)
		}
		if index > 0 && worldSize <= manifest.ExactSupportedWorldSizes[index-1] {
			return fmt.Errorf("exact_supported_world_sizes must be sorted and unique")
		}
		if worldSize == membership.WorldSize() {
			worldSupported = true
		}
	}
	if !worldSupported {
		return fmt.Errorf(
			"checkpoint world size %d is not exact-resume supported",
			membership.WorldSize(),
		)
	}
	return nil
}

func distributedResumeManifestFilesExist(manifest distributedResumeManifest) bool {
	dir := filepath.Dir(manifest.ManifestPath)
	for _, name := range []string{manifest.ModelFile, manifest.StateFile} {
		if name == "" {
			return false
		}
		info, err := os.Stat(filepath.Join(dir, name))
		if err != nil || info.IsDir() {
			return false
		}
	}
	return true
}
