package train

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Recovery owns only completed, disk-backed bundles. It must never inspect the
// trainer after a failed submit/collect: even finite GPU state may be partial.
type gridRecovery struct {
	dir      string
	manifest string
	step     int
}

type gridRecoveryPointer struct {
	Bundle string `json:"bundle"`
	Step   int    `json:"step"`
}

func gridRecoveryEvery(checkpointEvery, valEvery int) int {
	if checkpointEvery > 0 {
		return checkpointEvery
	}
	return max(1, valEvery)
}

func newGridRecovery(checkpointDir string) *gridRecovery {
	if checkpointDir == "" {
		return nil
	}
	return &gridRecovery{dir: filepath.Join(checkpointDir, "recovery")}
}

// publish keeps the previous bundle until both the new bundle and pointer are
// published. A disk failure cannot turn a partially written bundle into latest.
func (r *gridRecovery) publish(step int, write func(string) (string, error)) error {
	if r == nil {
		return nil
	}
	if err := os.MkdirAll(r.dir, 0o755); err != nil {
		return err
	}
	var previous gridRecoveryPointer
	if b, err := os.ReadFile(filepath.Join(r.dir, "latest.json")); err == nil {
		_ = json.Unmarshal(b, &previous)
	}
	staging, err := os.MkdirTemp(r.dir, ".pending-")
	if err != nil {
		return err
	}
	defer func() { _ = os.RemoveAll(staging) }()
	manifest, err := write(staging)
	if err != nil {
		return fmt.Errorf("write grid recovery checkpoint: %w", err)
	}
	// Verify the completion marker and every referenced state file before
	// publishing. This reads disk only, not the live trainer.
	m, err := readResumeManifest(manifest)
	if err != nil {
		return err
	}
	if m.GlobalStep != step {
		return fmt.Errorf("grid recovery checkpoint step=%d, expected %d", m.GlobalStep, step)
	}
	if _, err = loadResumeState(m); err != nil {
		return fmt.Errorf("verify grid recovery checkpoint: %w", err)
	}
	bundle := "bundle-" + strings.TrimPrefix(filepath.Base(staging), ".pending-")
	dest := filepath.Join(r.dir, bundle)
	if err = os.Rename(staging, dest); err != nil {
		return err
	}
	if err = atomicWriteJSON(filepath.Join(r.dir, "latest.json"), gridRecoveryPointer{Bundle: bundle, Step: step}); err != nil {
		_ = os.RemoveAll(dest)
		return err
	}
	r.manifest, r.step = filepath.Join(dest, filepath.Base(manifest)), step
	if err = os.Remove(filepath.Join(r.dir, "failure.json")); err != nil && !os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "warning: remove stale grid recovery failure report: %v\n", err)
	}
	// Delete only a generated immediate child, never a path from an arbitrary
	// manifest. User-requested periodic checkpoints outside recovery are retained.
	if previous.Bundle != bundle && strings.HasPrefix(previous.Bundle, "bundle-") && filepath.Base(previous.Bundle) == previous.Bundle {
		if err = os.RemoveAll(filepath.Join(r.dir, previous.Bundle)); err != nil {
			fmt.Fprintf(os.Stderr, "warning: retain obsolete grid recovery bundle: %v\n", err)
		}
	}
	return nil
}

func (r *gridRecovery) failure(err error) error {
	if err == nil || !isMLXOutOfMemoryError(err) {
		return err
	}
	if r == nil || r.manifest == "" {
		return fmt.Errorf("%w; no recovery bundle available (configure -checkpoint-dir); failed GPU state was not checkpointed", err)
	}
	// Best effort disk-only failure report. The checkpoint already exists even
	// if disk space runs out while writing this diagnostic.
	report := struct {
		Error    string `json:"error"`
		Manifest string `json:"resume_manifest"`
		Step     int    `json:"completed_steps"`
	}{err.Error(), r.manifest, r.step}
	if e := atomicWriteJSON(filepath.Join(r.dir, "failure.json"), report); e != nil {
		fmt.Fprintf(os.Stderr, "warning: write grid recovery failure report: %v\n", e)
	}
	return fmt.Errorf("%w; known-good recovery at completed step %d: resume with -resume %q; work after that checkpoint must be replayed", err, r.step, r.manifest)
}
