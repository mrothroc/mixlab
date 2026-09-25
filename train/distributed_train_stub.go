//go:build !mlx || !cgo || (!darwin && !linux)

package train

import "fmt"

func runDistributedTrain(cfg *ArchConfig, pattern string, opts TrainOptions) (TrainResult, error) {
	return TrainResult{}, fmt.Errorf("DDP training requires MLX; rebuild with CGO_ENABLED=1 go build -tags mlx ./cmd/mixlab")
}
