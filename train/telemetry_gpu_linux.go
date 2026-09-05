//go:build linux

package train

import (
	"context"
	"time"
)

func sampleGPUUtilPercent() *float64 {
	ctx, cancel := context.WithTimeout(context.Background(), 750*time.Millisecond)
	defer cancel()
	return sampleNvidiaSMIGPUUtilPercent(ctx)
}
