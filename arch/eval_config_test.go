package arch

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestEvalConfigValidation(t *testing.T) {
	got, err := ParseArchConfig([]byte(`{
		"model_dim": 128,
		"vocab_size": 1024,
		"seq_len": 128,
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {"steps": 10, "lr": 0.001, "batch_tokens": 1024},
		"eval": {
			"ttt_mode": "legal_chunk_sgd",
			"chunk_tokens": 2048,
			"ttt_epochs": 2,
			"ttt_lr": 0.002,
			"ttt_momentum": 0,
			"ttt_lr_schedule": "constant"
		}
	}`), "eval")
	if err != nil {
		t.Fatalf("parse legal eval: %v", err)
	}
	if got.Eval == nil {
		t.Fatal("eval spec was nil")
	}
	if !got.Eval.LegalChunkSGDEnabled() {
		t.Fatalf("eval mode = %q, want legal_chunk_sgd", got.Eval.TTTMode)
	}
	if got.Eval.ChunkTokens != 2048 || got.Eval.TTTEpochs != 2 || got.Eval.TTTLR != 0.002 || got.Eval.EffectiveTTTMomentum() != 0 || got.Eval.TTTLRSchedule != "constant" {
		t.Fatalf("parsed eval spec = %+v", *got.Eval)
	}
	data, _ := json.Marshal(got)
	roundTrip, err := ParseArchConfig(data, "eval-roundtrip")
	if err != nil {
		t.Fatalf("round-trip legal eval: %v", err)
	}
	if roundTrip.Eval == nil || roundTrip.Eval.TTTMode != "legal_chunk_sgd" || roundTrip.Eval.EffectiveTTTMomentum() != 0 {
		t.Fatalf("round-trip eval spec = %+v", roundTrip.Eval)
	}

	defaulted, err := ParseArchConfig([]byte(`{
		"model_dim": 128,
		"vocab_size": 1024,
		"seq_len": 128,
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {"steps": 10, "lr": 0.001, "batch_tokens": 1024},
		"eval": {"ttt_mode": "legal_chunk_sgd"}
	}`), "eval-defaults")
	if err != nil {
		t.Fatalf("parse legal eval defaults: %v", err)
	}
	wantDefaults := DefaultLegalChunkSGDEvalSpec()
	if defaulted.Eval.ChunkTokens != wantDefaults.ChunkTokens || defaulted.Eval.TTTEpochs != wantDefaults.TTTEpochs ||
		defaulted.Eval.TTTLR != wantDefaults.TTTLR || defaulted.Eval.EffectiveTTTMomentum() != wantDefaults.EffectiveTTTMomentum() ||
		defaulted.Eval.TTTLRSchedule != wantDefaults.TTTLRSchedule {
		t.Fatalf("defaulted eval spec = %+v, want %+v", *defaulted.Eval, wantDefaults)
	}

	omitted, err := ParseArchConfig([]byte(`{
		"model_dim": 128,
		"vocab_size": 1024,
		"seq_len": 128,
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {"steps": 10, "lr": 0.001, "batch_tokens": 1024}
	}`), "eval-omitted")
	if err != nil {
		t.Fatalf("parse omitted eval: %v", err)
	}
	if omitted.Eval != nil {
		t.Fatalf("omitted eval pointer = %+v, want nil", omitted.Eval)
	}
	if omitted.EffectiveEvalSpec().TTTMode != "none" {
		t.Fatalf("omitted effective eval mode = %q, want none", omitted.EffectiveEvalSpec().TTTMode)
	}

	empty, err := ParseArchConfig([]byte(`{
		"model_dim": 128,
		"vocab_size": 1024,
		"seq_len": 128,
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {"steps": 10, "lr": 0.001, "batch_tokens": 1024},
		"eval": {}
	}`), "eval-empty")
	if err != nil {
		t.Fatalf("parse empty eval: %v", err)
	}
	if empty.Eval == nil || empty.Eval.TTTMode != "none" {
		t.Fatalf("empty eval spec = %+v, want mode none", empty.Eval)
	}

	badSchedule := []byte(`{
		"model_dim": 128,
		"vocab_size": 1024,
		"seq_len": 128,
		"blocks": [{"type": "plain", "heads": 4}],
		"training": {"steps": 10, "lr": 0.001, "batch_tokens": 1024},
		"eval": {"ttt_mode": "legal_chunk_sgd", "ttt_lr_schedule": "linear"}
	}`)
	if _, err := ParseArchConfig(badSchedule, "eval-bad-schedule"); err == nil || !strings.Contains(err.Error(), "ttt_lr_schedule") {
		t.Fatalf("invalid schedule error = %v, want ttt_lr_schedule", err)
	}
}
