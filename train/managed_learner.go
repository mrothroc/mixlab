package train

import (
	"fmt"

	"github.com/mrothroc/mixlab/arch"
	"github.com/mrothroc/mixlab/data"
	"github.com/mrothroc/mixlab/workerjob"
)

// PrepareLocalLearner validates the numerical and local dataset contracts after
// local IPC authentication, before constructing a device engine or consuming data.
// Learner application, hosting and remote authentication are not owned here.
func PrepareLocalLearner(a workerjob.LearnerAssignment) (*ArchConfig, data.LearnerIdentity, error) {
	var id data.LearnerIdentity
	if err := a.Validate(); err != nil {
		return nil, id, err
	}
	if a.Cohort == nil {
		return nil, id, fmt.Errorf("managed learner requires the complete committed cohort")
	}
	cfg, err := arch.ParseArchConfig(a.Config, "managed learner assignment")
	if err != nil {
		return nil, id, err
	}
	if err = arch.ValidateDistributedConfig(cfg); err != nil {
		return nil, id, err
	}
	if cfg.Training.Distributed == nil || cfg.Training.Distributed.Mode != "diloco" {
		return nil, id, fmt.Errorf("managed learner requires DiLoCo config")
	}
	dataset, err := data.DistributedDatasetIdentity(a.TrainPattern)
	if err != nil {
		return nil, id, err
	}
	if dataset != a.Contract.DatasetHash {
		return nil, id, fmt.Errorf("learner dataset digest mismatch")
	}
	if err = configureDatasetForTraining(cfg, a.TrainPattern, cfg.Name); err != nil {
		return nil, id, err
	}
	if cfg.CharVocabSize > 0 {
		if _, err = configureCharFeaturesForTraining(cfg, a.TrainPattern); err != nil {
			return nil, id, err
		}
	}
	p, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		return nil, id, err
	}
	for _, op := range p.Ops {
		if op.Code == arch.OpRandomNormal {
			return nil, id, fmt.Errorf("learner rejects unkeyed random_normal")
		}
	}
	ph, err := workerjob.Digest(p)
	if err != nil {
		return nil, id, err
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		return nil, id, err
	}
	lh, err := workerjob.Digest(shapes)
	if err != nil {
		return nil, id, err
	}
	spec, err := buildTrainerOptimizerSpec(cfg, shapes)
	if err != nil {
		return nil, id, err
	}
	spec.ComputeDType, err = gpuComputeDTypeForTraining(cfg)
	if err != nil {
		return nil, id, err
	}
	oh, err := optimizerSpecHash(spec)
	if err != nil {
		return nil, id, err
	}
	if ph != a.Contract.ProgramHash || lh != a.Contract.WeightLayoutHash || oh != a.Contract.InnerOptimizerHash {
		return nil, id, fmt.Errorf("learner graph/layout/optimizer differs from admission")
	}
	id = data.LearnerIdentity{RunID: a.RunID, WorkerID: a.Member.WorkerID, PartitionID: a.Member.DatasetPartitionID, Count: len(a.Cohort.OrderedLearners), Seed: cfg.Training.Seed}
	for i, m := range a.Cohort.OrderedLearners {
		if m.WorkerID == a.Member.WorkerID {
			id.Index = i
		}
	}
	return cfg, id, id.Validate()
}
