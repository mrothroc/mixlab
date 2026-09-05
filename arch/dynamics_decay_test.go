package arch

import "testing"

func TestDynamicsDecayExemptions(t *testing.T) {
	for _, tc := range []struct {
		block BlockSpec
		want  map[string]bool
	}{
		{BlockSpec{Type: "mamba3-canonical", InnerDim: 48, StateSize: 64, NGroups: 4}, map[string]bool{"A_log": true, "dt_bias": true}},
		{BlockSpec{Type: "gated_deltanet", Heads: 4, DK: 8, DV: 8}, map[string]bool{"A_log": true, "dt_bias": true}},
		{BlockSpec{Type: "hgrn2", Heads: 4}, nil},
		{BlockSpec{Type: "mlstm", Heads: 4, DK: 8, DV: 8}, nil},
		{BlockSpec{Type: "retnet", Heads: 4}, nil},
		{BlockSpec{Type: "ttt_mlp", Heads: 4}, nil},
		{BlockSpec{Type: "rwkv"}, nil},
		{BlockSpec{Type: "legacy_mamba"}, nil},
		{BlockSpec{Type: "gated_linear_ssm"}, nil},
	} {
		t.Run(tc.block.Type, func(t *testing.T) {
			metas, err := BlockWeightShapes(tc.block, 32, 16, 1, 64)
			if err != nil {
				t.Fatal(err)
			}
			seen := map[string]bool{}
			for _, meta := range metas {
				seen[meta.Name] = true
				if meta.ForceNoDecay != tc.want[meta.Name] {
					t.Errorf("%s ForceNoDecay=%v want %v", meta.Name, meta.ForceNoDecay, tc.want[meta.Name])
				}
				if meta.ForceNoDecay && meta.ForceDecay {
					t.Errorf("%s has conflicting decay overrides", meta.Name)
				}
			}
			for name := range tc.want {
				if !seen[name] {
					t.Errorf("missing exempt weight %s", name)
				}
			}
		})
	}
}
