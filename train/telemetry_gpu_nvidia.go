package train

import (
	"context"
	"encoding/csv"
	"io"
	"math"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

func sampleNvidiaSMIGPUUtilPercent(ctx context.Context) *float64 {
	cmd := exec.CommandContext(ctx, "nvidia-smi",
		"--query-gpu=utilization.gpu", "--format=csv,noheader,nounits")
	// Bound pipe cleanup too if a wrapper leaves a child holding stdout open.
	cmd.WaitDelay = 100 * time.Millisecond
	out, err := cmd.Output()
	if err != nil {
		return nil
	}
	return parseNvidiaSMIGPUUtilPercent(string(out))
}

func parseNvidiaSMIGPUUtilPercent(out string) *float64 {
	reader := csv.NewReader(strings.NewReader(out))
	reader.FieldsPerRecord = 1
	reader.TrimLeadingSpace = true
	var maxUtil float64
	found := false
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil
		}
		value, err := strconv.ParseFloat(strings.TrimSpace(record[0]), 64)
		if err != nil || math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || value > 100 {
			continue
		}
		if !found || value > maxUtil {
			maxUtil = value
			found = true
		}
	}
	if !found {
		return nil
	}
	return &maxUtil
}
