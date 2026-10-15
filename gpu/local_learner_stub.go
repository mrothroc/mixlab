//go:build !mlx || !cgo || (!darwin && !linux)

package gpu

import "fmt"

func TrainerSetLocalAccumulation(TrainerHandle, int) error {
	return fmt.Errorf("local learner requires MLX")
}
func TrainerInstallWeights(TrainerHandle, []int64) error {
	return fmt.Errorf("local learner requires MLX")
}
