package train

import (
	"fmt"
	"time"
)

// trainingProgress measures completed work against wall time, so asynchronous
// submit/collect timing cannot inflate throughput. The first iteration, including
// its validation and logging, ends before this clock starts.
type trainingProgress struct {
	start  time.Time
	steps  int
	tokens int64
}

func (p *trainingProgress) afterStep(now time.Time) {
	if p.start.IsZero() {
		p.start = now
	}
}

func (p *trainingProgress) collected(now time.Time, tokens int) (time.Duration, int, float64) {
	if p.start.IsZero() {
		return 0, 0, 0
	}
	p.steps++
	p.tokens += int64(tokens)
	elapsed := now.Sub(p.start)
	if elapsed <= 0 {
		return 0, p.steps, 0
	}
	return elapsed, p.steps, float64(p.tokens) / elapsed.Seconds()
}

func formatTrainingThroughput(tokensPerSec float64) string {
	if tokensPerSec <= 0 {
		return "n/a"
	}
	return fmt.Sprintf("%.0f", tokensPerSec)
}
