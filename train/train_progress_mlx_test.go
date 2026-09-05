//go:build mlx && cgo && (darwin || linux)

package train

import (
	"encoding/json"
	"path/filepath"
	"testing"
	"time"
)

type slowStartupTelemetryWriter struct {
	records []telemetrySnapshot
	delay   time.Duration
}

func (w *slowStartupTelemetryWriter) Write(data []byte) (int, error) {
	var record telemetrySnapshot
	if err := json.Unmarshal(data, &record); err != nil {
		return 0, err
	}
	if len(w.records) == 0 {
		time.Sleep(w.delay)
	}
	w.records = append(w.records, record)
	return len(data), nil
}

func TestTrainingProgressExcludesStartupLogOverheadMLX(t *testing.T) {
	if !mlxAvailable() {
		t.Skip("MLX backend not available")
	}
	cfg, err := ParseArchConfig([]byte(`{
		"model_dim":8,"vocab_size":8,"seq_len":4,
		"blocks":[{"type":"plain","heads":2}],
		"training":{"optimizer":"adamw","steps":3,"batch_tokens":8,"lr":0.001}
	}`), "startup-timing")
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	tokens := make([]uint16, 128)
	for i := range tokens {
		tokens[i] = uint16(i % cfg.VocabSize)
	}
	writeInferenceShard(t, filepath.Join(dir, "train_000.bin"), tokens)
	writeInferenceShard(t, filepath.Join(dir, "val_000.bin"), tokens)
	writer := &slowStartupTelemetryWriter{delay: 100 * time.Millisecond}
	rt := &telemetryRuntime{state: newTelemetryState(), enc: json.NewEncoder(writer)}
	if _, err := runTrain(cfg, filepath.Join(dir, "train_*.bin"), TrainOptions{LogEvery: 1, telemetry: rt}); err != nil {
		t.Fatal(err)
	}
	if len(writer.records) != 3 {
		t.Fatalf("telemetry records=%d want 3", len(writer.records))
	}
	first := writer.records[0]
	if first.TokensPerSec != 0 || first.SteadyElapsedSeconds != 0 || first.ValLoss == nil {
		t.Fatalf("startup snapshot=%+v", first)
	}
	for _, record := range writer.records[1:] {
		if record.TokensPerSec <= 0 || record.SteadyElapsedSeconds <= 0 {
			t.Fatalf("step %d has no steady estimate", record.Step)
		}
		excluded := record.ElapsedSeconds - record.SteadyElapsedSeconds
		if excluded < first.ElapsedSeconds+writer.delay.Seconds()-0.001 {
			t.Fatalf("step %d excluded only %.4fs; startup plus logging took at least %.4fs",
				record.Step, excluded, first.ElapsedSeconds+writer.delay.Seconds())
		}
	}
}
