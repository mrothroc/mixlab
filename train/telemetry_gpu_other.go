//go:build !darwin && !linux

package train

func sampleGPUUtilPercent() *float64 {
	return nil
}
