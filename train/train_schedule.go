package train

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/mrothroc/mixlab/arch"
)

func formatMLMMaskUnitSchedule(schedule []arch.MLMMaskUnitScheduleEntry) string {
	parts := make([]string, 0, len(schedule))
	for _, entry := range schedule {
		parts = append(parts, fmt.Sprintf("step%d=%s", entry.Step, entry.Unit))
	}
	return strings.Join(parts, ", ")
}

func qatModeForStep(spec TrainingSpec, step int) string {
	mode := strings.TrimSpace(strings.ToLower(spec.QAT))
	if mode == "" {
		mode = "none"
	}
	if spec.QATStart > 0 && step < spec.QATStart {
		return "none"
	}
	return mode
}

func shouldWriteCheckpoint(step, every int) bool {
	return every > 0 && (step+1)%every == 0
}

func effectiveTrainEvery(option int, envName string, fallback int) int {
	if option > 0 {
		return option
	}
	if raw := strings.TrimSpace(os.Getenv(envName)); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
			return parsed
		}
	}
	return fallback
}

func envTruthy(name string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(name))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func isPowerOfTwo(n int) bool {
	return n > 0 && n&(n-1) == 0
}

func shouldLogTrainingStep(step, totalSteps, every int) bool {
	if totalSteps <= 0 {
		return false
	}
	if step == 0 || step == totalSteps-1 {
		return true
	}
	if every <= 0 {
		every = 100
	}
	if step < every && isPowerOfTwo(step) {
		return true
	}
	return step%every == 0
}

func shouldRunValidationStep(step, totalSteps, every int) bool {
	if totalSteps <= 0 {
		return false
	}
	if step == 0 || step == totalSteps-1 {
		return true
	}
	if every <= 0 {
		every = 100
	}
	return step%every == 0
}
