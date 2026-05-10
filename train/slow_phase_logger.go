package train

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

var (
	slowTrainingPhaseThresholdOnce  sync.Once
	slowTrainingPhaseThresholdValue time.Duration
)

func slowTrainingPhaseThreshold() time.Duration {
	slowTrainingPhaseThresholdOnce.Do(func() {
		if !envTruthy("MIXLAB_TRAINING_SLOW_PHASE_LOGS") {
			return
		}
		if envTruthy("MIXLAB_DISABLE_TRAINING_SLOW_PHASE_LOGS") {
			return
		}
		const fallback = 30 * time.Second
		raw := strings.TrimSpace(os.Getenv("MIXLAB_TRAINING_SLOW_PHASE_SECONDS"))
		if raw == "" {
			slowTrainingPhaseThresholdValue = fallback
			return
		}
		seconds, err := strconv.ParseFloat(raw, 64)
		if err != nil || seconds <= 0 {
			slowTrainingPhaseThresholdValue = fallback
			return
		}
		slowTrainingPhaseThresholdValue = time.Duration(seconds * float64(time.Second))
	})
	return slowTrainingPhaseThresholdValue
}

// startSlowTrainingPhaseLogger arms a watchdog that prints periodic
// "slow training phase" warnings when MIXLAB_TRAINING_SLOW_PHASE_LOGS=1 and
// the phase exceeds the configured threshold (default 30s, override via
// MIXLAB_TRAINING_SLOW_PHASE_SECONDS). Returns a stop function the caller
// invokes when the phase completes.
func startSlowTrainingPhaseLogger(runName string, step int, phase string) func() {
	threshold := slowTrainingPhaseThreshold()
	if threshold <= 0 {
		return func() {}
	}
	start := time.Now()
	done := make(chan struct{})
	timer := time.AfterFunc(threshold, func() {
		ticker := time.NewTicker(threshold)
		defer ticker.Stop()
		for {
			fmt.Printf("  [%s] slow training phase step=%d phase=%s elapsed=%s\n",
				runName, step, phase, time.Since(start).Round(time.Second))
			select {
			case <-done:
				return
			case <-ticker.C:
			}
		}
	})
	return func() {
		if timer.Stop() {
			close(done)
			return
		}
		close(done)
	}
}
