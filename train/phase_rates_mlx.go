//go:build mlx && cgo && (darwin || linux)

package train

func (t *mlxGPUTrainer) effectiveLearningRates(lr float32) string {
	return effectiveGroupRates(t.optimizerSpec, lr)
}
