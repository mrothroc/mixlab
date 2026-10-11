package arch

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"
)

func TestAllowTF32RoundTrip(t *testing.T) {
	for _, field := range []string{"", `,"allow_tf32":false`, `,"allow_tf32":true`} {
		b := []byte(fmt.Sprintf(`{"model_dim":16,"vocab_size":32,"seq_len":4,"blocks":[{"type":"plain","heads":2}],"training":{"batch_tokens":8%s}}`, field))
		cfg, err := ParseArchConfig(b, "tf32")
		if err != nil {
			t.Fatal(err)
		}
		blob, err := json.Marshal(cfg)
		if err != nil {
			t.Fatal(err)
		}
		again, err := ParseArchConfig(blob, "roundtrip")
		if err != nil {
			t.Fatal(err)
		}
		if field == "" {
			if again.Training.AllowTF32 != nil {
				t.Fatal("omitted became explicit")
			}
		} else if again.Training.AllowTF32 == nil || *again.Training.AllowTF32 != *cfg.Training.AllowTF32 {
			t.Fatal("lost explicit boolean")
		}
	}
}

func TestGridAllowTF32(t *testing.T) {
	b, err := os.ReadFile("../examples/grid_two_stage_1.json")
	if err != nil {
		t.Fatal(err)
	}
	var raw map[string]any
	if err = json.Unmarshal(b, &raw); err != nil {
		t.Fatal(err)
	}
	raw["training"].(map[string]any)["allow_tf32"] = false
	b, err = json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = ParseArchConfig(b, "grid tf32"); err != nil {
		t.Fatal(err)
	}
}
