package arch

import (
	"encoding/json"
	"testing"
)

func TestMinLRFractionPresenceRoundTrip(t *testing.T) {
	for _, tc := range []struct {
		name, input string
		set         bool
		value       float32
	}{
		{"omitted", `{}`, false, 0},
		{"null", `{"min_lr_fraction":null}`, false, 0},
		{"zero", `{"min_lr_fraction":0}`, true, 0},
		{"positive", `{"min_lr_fraction":0.001}`, true, 0.001},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var spec TrainingSpec
			if err := json.Unmarshal([]byte(tc.input), &spec); err != nil {
				t.Fatal(err)
			}
			for i := 0; i < 3; i++ {
				spec.ApplyDefaults()
				if spec.MinLRFractionConfigured() != tc.set || spec.MinLRFraction != tc.value {
					t.Fatalf("round %d: configured=%v value=%g", i, spec.MinLRFractionConfigured(), spec.MinLRFraction)
				}
				blob, err := json.Marshal(spec)
				if err != nil {
					t.Fatal(err)
				}
				var fields map[string]json.RawMessage
				if err := json.Unmarshal(blob, &fields); err != nil {
					t.Fatal(err)
				}
				if _, present := fields["min_lr_fraction"]; present != tc.set {
					t.Fatalf("serialized floor presence=%v, want %v", present, tc.set)
				}
				if err := json.Unmarshal(blob, &spec); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
	if !(TrainingSpec{MinLRFraction: 0.02}).MinLRFractionConfigured() {
		t.Fatal("programmatic nonzero floor must remain supported")
	}
}

func TestMinLRFractionValidation(t *testing.T) {
	for _, value := range []string{"-0.1", "1", "1.1"} {
		_, err := ParseArchConfig([]byte(`{"model_dim":16,"vocab_size":32,"seq_len":4,
			"blocks":[{"type":"plain","heads":2}],"training":{"min_lr_fraction":`+value+`}}`), "floor")
		if err == nil {
			t.Fatalf("accepted min_lr_fraction=%s", value)
		}
	}
}
