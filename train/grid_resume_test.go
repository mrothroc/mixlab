package train

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/data"
)

func TestGridResumeStateValidation(t *testing.T) {
	shapes := []WeightShape{{Name: "frozen", Frozen: true}, {Name: "live"}}
	makeState := func() resumeManifest {
		s, _ := data.NewGridSampler(3, 42)
		if _, _, _, err := s.Next(2); err != nil {
			t.Fatal(err)
		}
		return resumeManifest{GlobalStep: 1, Grid: snapshotGridRun(s, shapes, .5, TrainResult{HasValLoss: true, LastValLoss: .3, FirstLoss: 1, LastLoss: 1}, 2, false)}
	}
	for _, kind := range []string{"valid", "missing", "version", "trainable", "cadence", "cursor", "order", "metric", "incomplete"} {
		t.Run(kind, func(t *testing.T) {
			state := makeState()
			switch kind {
			case "missing":
				state.Grid = nil
			case "version":
				state.Grid.Version = 2
			case "trainable":
				state.Grid.Trainable = []string{"frozen"}
			case "cadence":
				state.Grid.ValEvery = 3
			case "cursor":
				state.Grid.Sampler.Cursor = 0
			case "order":
				state.Grid.Sampler.Order[0] = -1
			case "metric":
				state.Grid.FirstLoss = math.NaN()
			case "incomplete":
				state.Grid.BestRMSE = nil
			}
			s, _ := data.NewGridSampler(3, 42)
			var result TrainResult
			best, _, err := restoreGridRunState(state, s, shapes, 2, 2, &result)
			if (err == nil) != (kind == "valid") {
				t.Fatal(err)
			}
			if err == nil && (best != .5 || result.LastValLoss != .3) {
				t.Fatal(result, best)
			}
		})
	}
}

func TestGridDatasetIdentityDetectsPayloadChanges(t *testing.T) {
	manifest := prepareTinyGrid(t)
	before, err := data.GridDatasetIdentity(manifest)
	if err != nil {
		t.Fatal(err)
	}
	m, err := data.LoadDatasetManifest(manifest)
	if err != nil {
		t.Fatal(err)
	}
	files, err := filepath.Glob(filepath.Join(filepath.Dir(manifest), m.Splits["train"].Pattern))
	if err != nil || len(files) == 0 {
		t.Fatal(files, err)
	}
	info, err := os.Stat(files[0])
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(files[0])
	if err != nil {
		t.Fatal(err)
	}
	b[len(b)-1] ^= 1 // valid mask bit; same size and restored mtime
	if err = os.WriteFile(files[0], b, 0600); err != nil {
		t.Fatal(err)
	}
	if err = os.Chtimes(files[0], info.ModTime(), info.ModTime()); err != nil {
		t.Fatal(err)
	}
	after, err := data.GridDatasetIdentity(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if before == after {
		t.Fatal("changed payload retained dataset identity")
	}
}
