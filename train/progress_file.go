package train

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// A local, opt-in liveness channel for an out-of-process watchdog. Never sample
// GPU state here: the caller supplies counters already read after collection.
type trainingProgressFile struct {
	path string
}

type trainingProgressRecord struct {
	PID            int    `json:"pid"`
	Step           int    `json:"step"`
	OptimizerSteps uint64 `json:"optimizer_steps"`
}

func newTrainingProgressFile() (*trainingProgressFile, error) {
	path := os.Getenv("MIXLAB_PROGRESS_FILE")
	if path == "" {
		return nil, nil
	}
	if err := allowDebugParent(); err != nil {
		return nil, err
	}
	p := &trainingProgressFile{path: path}
	if err := p.update(-1, 0); err != nil {
		return nil, err
	}
	return p, nil
}

func (p *trainingProgressFile) update(step int, committed uint64) error {
	if p == nil {
		return nil
	}
	value, err := json.Marshal(trainingProgressRecord{os.Getpid(), step, committed})
	if err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(p.path), ".mixlab-progress-*")
	if err != nil {
		return fmt.Errorf("create training progress: %w", err)
	}
	// Best effort: after a successful rename the temp name is already gone.
	defer func() { _ = os.Remove(f.Name()) }()
	_, writeErr := f.Write(value)
	closeErr := f.Close()
	if writeErr != nil {
		return fmt.Errorf("write training progress: %w", writeErr)
	}
	if closeErr != nil {
		return fmt.Errorf("close training progress: %w", closeErr)
	}
	if err := os.Rename(f.Name(), p.path); err != nil {
		return fmt.Errorf("publish training progress: %w", err)
	}
	return nil
}
