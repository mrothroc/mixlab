package train

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestLoadConfigsFromDir(t *testing.T) {
	dir := t.TempDir()

	// Write two valid configs with matching vocab/seq/batch.
	cfgs := []ArchConfig{
		{
			Name: "alpha", ModelDim: 64, VocabSize: 512, SeqLen: 64,
			Blocks:   []BlockSpec{{Type: "plain", Heads: 2}},
			Training: TrainingSpec{Steps: 10, LR: 1e-3, BatchTokens: 64},
		},
		{
			Name: "beta", ModelDim: 128, VocabSize: 512, SeqLen: 64,
			Blocks:   []BlockSpec{{Type: "plain", Heads: 4}},
			Training: TrainingSpec{Steps: 10, LR: 1e-3, BatchTokens: 64},
		},
	}
	for i, cfg := range cfgs {
		b, err := json.Marshal(cfg)
		if err != nil {
			t.Fatalf("marshal config %d: %v", i, err)
		}
		path := filepath.Join(dir, cfg.Name+".json")
		if err := os.WriteFile(path, b, 0644); err != nil {
			t.Fatalf("write config %d: %v", i, err)
		}
	}

	// Also write a non-json file that should be ignored.
	if err := os.WriteFile(filepath.Join(dir, "README.md"), []byte("# ignore"), 0644); err != nil {
		t.Fatalf("write readme: %v", err)
	}

	loaded, err := loadConfigsFromDir(dir)
	if err != nil {
		t.Fatalf("loadConfigsFromDir: %v", err)
	}
	if len(loaded) != 2 {
		t.Fatalf("expected 2 configs, got %d", len(loaded))
	}
	// Should be sorted alphabetically by filename.
	if loaded[0].Name != "alpha" {
		t.Errorf("first config name = %q, want alpha", loaded[0].Name)
	}
	if loaded[1].Name != "beta" {
		t.Errorf("second config name = %q, want beta", loaded[1].Name)
	}
}

func TestLoadConfigsFromDirEmpty(t *testing.T) {
	dir := t.TempDir()
	loaded, err := loadConfigsFromDir(dir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(loaded) != 0 {
		t.Errorf("expected 0 configs, got %d", len(loaded))
	}
}

func TestLoadConfigsFromDirInvalid(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "bad.json"), []byte("{invalid"), 0644); err != nil {
		t.Fatalf("write: %v", err)
	}
	_, err := loadConfigsFromDir(dir)
	if err == nil {
		t.Fatal("expected error for invalid JSON config")
	}
}

func TestLoadConfigsFromDirNotExists(t *testing.T) {
	_, err := loadConfigsFromDir("/nonexistent/configs/dir")
	if err == nil {
		t.Fatal("expected error for nonexistent directory")
	}
}

func TestRunArchRaceValidation(t *testing.T) {
	// Missing configs dir.
	err := runArchRace("", "data/*.bin", TrainOptions{})
	if err == nil {
		t.Fatal("expected error for missing configs dir")
	}

	// Missing train pattern.
	err = runArchRace("/some/dir", "", TrainOptions{})
	if err == nil {
		t.Fatal("expected error for missing train pattern")
	}
}

func TestRunArchRaceMismatchedConfigs(t *testing.T) {
	dir := t.TempDir()

	// Two configs with different vocab sizes.
	cfgs := []ArchConfig{
		{
			Name: "a", ModelDim: 64, VocabSize: 512, SeqLen: 64,
			Blocks:   []BlockSpec{{Type: "plain", Heads: 2}},
			Training: TrainingSpec{Steps: 10, LR: 1e-3, BatchTokens: 64},
		},
		{
			Name: "b", ModelDim: 64, VocabSize: 1024, SeqLen: 64,
			Blocks:   []BlockSpec{{Type: "plain", Heads: 2}},
			Training: TrainingSpec{Steps: 10, LR: 1e-3, BatchTokens: 64},
		},
	}
	for _, cfg := range cfgs {
		b, err := json.Marshal(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, cfg.Name+".json"), b, 0644); err != nil {
			t.Fatal(err)
		}
	}

	err := runArchRace(dir, "data/*.bin", TrainOptions{})
	if err == nil {
		t.Fatal("expected error for mismatched configs")
	}
}

func TestRunArchRaceEmptyDir(t *testing.T) {
	dir := t.TempDir()
	err := runArchRace(dir, "data/*.bin", TrainOptions{})
	if err == nil {
		t.Fatal("expected error for empty configs dir")
	}
}

func TestSortKey(t *testing.T) {
	withVal := TrainResult{LastLoss: 2.0, LastValLoss: 1.5, HasValLoss: true}
	withoutVal := TrainResult{LastLoss: 2.0, HasValLoss: false}

	if sortKey(withVal) != 1.5 {
		t.Errorf("sortKey with val = %f, want 1.5", sortKey(withVal))
	}
	if sortKey(withoutVal) != 2.0 {
		t.Errorf("sortKey without val = %f, want 2.0", sortKey(withoutVal))
	}
}
