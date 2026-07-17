package arch

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestMLMMaskUnitScheduleResolutionAndRoundTrip(t *testing.T) {
	cfg := parseObjectiveConfig(t, `"training": {
		"steps": 10, "batch_tokens": 8, "objective": "mlm", "mlm_mask_token_id": 7,
		"mlm_mask_unit": " token ",
		"mlm_mask_unit_schedule": [
			{"step": 0, "unit": " Whole_Word "},
			{"step": 7, "unit": " TOKEN "}
		]
	}`)
	for _, tt := range []struct {
		step int
		want string
	}{{0, MLMMaskUnitWholeWord}, {6, MLMMaskUnitWholeWord}, {7, MLMMaskUnitToken}, {20, MLMMaskUnitToken}} {
		if got := cfg.Training.EffectiveMLMMaskUnitForStep(tt.step); got != tt.want {
			t.Fatalf("EffectiveMLMMaskUnitForStep(%d)=%q, want %q", tt.step, got, tt.want)
		}
	}
	if !cfg.Training.UsesWholeWordMasking() {
		t.Fatal("UsesWholeWordMasking=false, want true")
	}
	cfg.Training.MLMWordStart = []uint8{1, 0}
	cfg.Training.MLMMaskEligible = []uint8{1, 1}
	blob, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if strings.Contains(string(blob), "MLMWordStart") || strings.Contains(string(blob), "mlm_word_start") || strings.Contains(string(blob), "mlm_mask_eligible") {
		t.Fatalf("runtime tokenizer metadata leaked into JSON: %s", blob)
	}
	roundTrip, err := ParseArchConfig(blob, "roundtrip")
	if err != nil {
		t.Fatalf("ParseArchConfig round trip: %v", err)
	}
	if got := roundTrip.Training.EffectiveMLMMaskUnitForStep(7); got != MLMMaskUnitToken {
		t.Fatalf("round-trip step 7 unit=%q", got)
	}
}

func TestMLMMaskUnitValidation(t *testing.T) {
	tests := []struct {
		name    string
		body    string
		wantErr string
	}{
		{name: "unknown base", body: `"training":{"steps":10,"batch_tokens":8,"objective":"mlm","mlm_mask_token_id":7,"mlm_mask_unit":"word"}`, wantErr: "mlm_mask_unit"},
		{name: "missing schedule unit", body: `"training":{"steps":10,"batch_tokens":8,"objective":"mlm","mlm_mask_token_id":7,"mlm_mask_unit_schedule":[{"step":0}]}`, wantErr: "unit"},
		{name: "first step", body: `"training":{"steps":10,"batch_tokens":8,"objective":"mlm","mlm_mask_token_id":7,"mlm_mask_unit_schedule":[{"step":1,"unit":"whole_word"}]}`, wantErr: "first step"},
		{name: "non increasing", body: `"training":{"steps":10,"batch_tokens":8,"objective":"mlm","mlm_mask_token_id":7,"mlm_mask_unit_schedule":[{"step":0,"unit":"whole_word"},{"step":0,"unit":"token"}]}`, wantErr: "strictly increasing"},
		{name: "outside steps", body: `"training":{"steps":10,"batch_tokens":8,"objective":"mlm","mlm_mask_token_id":7,"mlm_mask_unit_schedule":[{"step":0,"unit":"whole_word"},{"step":10,"unit":"token"}]}`, wantErr: "total training steps"},
		{name: "repeated unit", body: `"training":{"steps":10,"batch_tokens":8,"objective":"mlm","mlm_mask_token_id":7,"mlm_mask_unit_schedule":[{"step":0,"unit":"whole_word"},{"step":5,"unit":"whole_word"}]}`, wantErr: "redundantly repeats"},
		{name: "causal scope", body: `"training":{"steps":10,"batch_tokens":8,"objective":"causal","mlm_mask_unit":"whole_word"}`, wantErr: "only on MLM"},
		{name: "mntp scope", body: `"training":{"steps":10,"batch_tokens":8,"objective":"mntp","mlm_mask_token_id":7,"mlm_mask_unit":"whole_word"}`, wantErr: "only on MLM"},
		{name: "hybrid mntp scope", body: `"training":{"steps":10,"batch_tokens":8,"objective":"hybrid","hybrid_secondary_objective":"mntp","mlm_mask_token_id":7,"mlm_mask_unit":"whole_word"}`, wantErr: "only on MLM"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseArchConfig([]byte(objectiveConfigJSON(tt.body)), tt.name)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("error=%v, want substring %q", err, tt.wantErr)
			}
		})
	}

	for _, body := range []string{
		`"training":{"steps":10,"batch_tokens":8,"objective":"mlm","mlm_mask_token_id":7,"mlm_mask_unit":"whole_word"}`,
		`"training":{"steps":10,"batch_tokens":8,"objective":"hybrid","hybrid_secondary_objective":"mlm","mlm_mask_token_id":7,"mlm_mask_unit":"whole_word"}`,
		`"training":{"steps":10,"batch_tokens":8,"objective":"multihead","mlm_mask_token_id":7,"mlm_mask_unit":"whole_word","heads":[{"name":"scorer","objective":" MLM "},{"name":"causal","objective":"causal"}]}`,
	} {
		if _, err := ParseArchConfig([]byte(objectiveConfigJSON(body)), "valid WWM scope"); err != nil {
			t.Fatalf("valid WWM config rejected: %v", err)
		}
	}
}
