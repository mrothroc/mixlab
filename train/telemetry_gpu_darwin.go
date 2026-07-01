//go:build darwin

package train

import (
	"context"
	"os/exec"
	"time"
)

func sampleGPUUtilPercent() *float64 {
	ctx, cancel := context.WithTimeout(context.Background(), 750*time.Millisecond)
	defer cancel()
	out, err := exec.CommandContext(ctx, "ioreg", "-r", "-c", "IOAccelerator", "-d1").Output()
	if err != nil {
		return nil
	}
	return parseIORegGPUUtilPercent(string(out))
}
