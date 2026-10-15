//go:build !mlx || !cgo || (!darwin && !linux)

package train

import (
	"fmt"
	"github.com/mrothroc/mixlab/arch"
)

func makeLocalLearnerFactory(*ArchConfig, *arch.Program, uint64) (localLearnerFactory, error) {
	return nil, fmt.Errorf("local learner requires MLX")
}
