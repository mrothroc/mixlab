package arch

import (
	"os"
	"strings"
	"testing"
)

func TestPhasesRequireExplicitBaseLRWithRateOverrides(t *testing.T) {
	const phases = `"phases":[{"steps":4,"lr":0.0001},{"steps":4,"lr":0.00001}]`
	token := func(block, training string) string {
		return `{"model_dim":8,"vocab_size":32,"seq_len":8,"blocks":[` + block +
			`],"training":{"objective":"causal","batch_tokens":8` + training + `}}`
	}
	const plain = `{"type":"plain","heads":2}`
	const s4d = `{"type":"s4d","state_size":16}`
	for _, tc := range []struct {
		name, raw, want string
	}{
		{"matrix_lr without base", token(plain, `,"matrix_lr":0.02,`+phases), "training.matrix_lr"},
		{"embed_lr without base", token(plain, `,"embed_lr":0.02,`+phases), "training.embed_lr"},
		{"scalar_lr without base", token(plain, `,"scalar_lr":0.02,`+phases), "training.scalar_lr"},
		{"head_lr without base", token(plain, `,"head_lr":0.02,`+phases), "training.head_lr"},
		{"several overrides listed", token(plain, `,"matrix_lr":0.02,"head_lr":0.01,`+phases), "training.matrix_lr, training.head_lr"},
		{"block state_lr without base", token(`{"type":"s4d","state_size":16,"state_lr":0.001}`, ","+phases), "blocks[0].state_lr"},
		{"default sobolev rate without base", token(`{"type":"s4d","state_size":16,"sobolev_filter":true}`, ","+phases), "blocks[0].sobolev_filter"},
		{"explicit base with overrides", token(plain, `,"lr":0.0001,"matrix_lr":0.02,`+phases), ""},
		{"explicit zero base keeps legacy policy", token(plain, `,"lr":0,"matrix_lr":0.02,`+phases), ""},
		{"phases without overrides", token(plain, ","+phases), ""},
		{"zero override is not scaled", token(plain, `,"matrix_lr":0,`+phases), ""},
		{"overrides without phases", token(plain, `,"matrix_lr":0.02`), ""},
		{"frozen sobolev rate is unused", token(`{"type":"s4d","state_size":16,"sobolev_filter":{"trainable":false}}`, ","+phases), ""},
		{"s4d without overrides", token(s4d, ","+phases), ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseArchConfig([]byte(tc.raw), tc.name)
			if tc.want == "" {
				if err != nil {
					t.Fatal(err)
				}
				return
			}
			if err == nil {
				t.Fatalf("accepted phases with %s and no explicit training.lr", tc.want)
			}
			for _, part := range []string{tc.want, "training.lr", "first phase"} {
				if !strings.Contains(err.Error(), part) {
					t.Fatalf("error %q does not mention %q", err, part)
				}
			}
		})
	}
}

func TestGridPhasesRequireExplicitBaseLRWithMatrixOverride(t *testing.T) {
	raw, err := os.ReadFile("../examples/grid_regression_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	src := strings.Replace(string(raw), `"lr": 0.001, `, "", 1)
	if strings.Contains(src, `"lr"`) || strings.Contains(src, `"matrix_lr"`) {
		t.Fatal("fixture layout changed; this test needs to remove its training.lr")
	}
	const phases = `"phases":[{"steps":6,"lr":0.0001},{"steps":4,"lr":0.00001}],`
	with := func(fields string) string {
		return strings.Replace(src, `"training": {`, `"training": {`+fields, 1)
	}
	if !strings.Contains(src, `"training": {`) {
		t.Fatal("fixture layout changed")
	}
	if _, err := ParseArchConfig([]byte(with(phases+`"matrix_lr":0.02,`)), "grid"); err == nil ||
		!strings.Contains(err.Error(), "training.matrix_lr") {
		t.Fatalf("grid kernel override accepted without explicit base: %v", err)
	}
	if _, err := ParseArchConfig([]byte(with(phases+`"lr":0.0001,"matrix_lr":0.02,`)), "grid"); err != nil {
		t.Fatal(err)
	}
}
