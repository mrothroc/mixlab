package train

import (
	mixdist "github.com/mrothroc/mixlab/distributed"
	"github.com/mrothroc/mixlab/gpu"
)

// DistributedTrainerContext carries the immutable group identity and the
// local DDP execution settings shared by setup, resume, and the MLX trainer.
type DistributedTrainerContext struct {
	GroupRuntime        *gpu.GroupRuntime
	LocalView           mixdist.LocalGroupView
	GradientBucketBytes uint64
	AccumulationSteps   int
	DatasetHash         string
	ScheduledPhase      string
}
