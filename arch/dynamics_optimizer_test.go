package arch

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"
)

func TestDynamicsStateLRValidation(t *testing.T) {
	for _, kind := range []string{"s4d", "mamba3-canonical", "gated_deltanet"} {
		for _, lr := range []float64{0.001, 0, -1, math.NaN(), math.Inf(1), 1e100, 1e-100} {
			b := BlockSpec{Type: kind, Heads: 2, DK: 4, StateLR: &lr}
			err := validateBlockSpec(b, "state-lr", "blocks", 0)
			if lr == 0.001 {
				if err != nil {
					t.Fatal(err)
				}
			} else if err == nil || !strings.Contains(err.Error(), "state_lr") {
				t.Fatalf("%s lr=%g: %v", kind, lr, err)
			}
		}
	}
	lr := 0.001
	for _, kind := range []string{"plain", "hgrn2", "legacy_mamba", "gated_linear_ssm"} {
		err := validateBlockSpec(BlockSpec{Type: kind, StateLR: &lr}, "state-lr", "blocks", 0)
		if err == nil || !strings.Contains(err.Error(), "type=s4d, type=mamba3-canonical, or type=gated_deltanet") {
			t.Fatalf("%s: %v", kind, err)
		}
	}
}

func TestDynamicsStateLRMetadata(t *testing.T) {
	for _, kind := range []string{"mamba3-canonical", "gated_deltanet"} {
		for _, bidir := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/bidir=%t", kind, bidir), func(t *testing.T) {
				raw := fmt.Sprintf(`{"model_dim":8,"vocab_size":16,"seq_len":4,
				"blocks":[{"type":%q,"heads":2,"d_k":4,"bidirectional":%t,"state_lr":0.001}],
				"training":{"objective":"mlm","mlm_mask_token_id":1,"batch_tokens":4}}`, kind, bidir)
				cfg, err := ParseArchConfig([]byte(raw), "state-lr")
				if err != nil {
					t.Fatal(err)
				}
				encoded, err := json.Marshal(cfg)
				if err != nil {
					t.Fatal(err)
				}
				round, err := ParseArchConfig(encoded, "roundtrip")
				if err != nil {
					t.Fatal(err)
				}
				if round.Blocks[0].StateLR == nil || *round.Blocks[0].StateLR != 0.001 {
					t.Fatal("state_lr lost in roundtrip")
				}
				b := cfg.Blocks[0]
				got, err := BlockWeightShapes(b, 8, 4, 1, 16)
				if err != nil {
					t.Fatal(err)
				}
				b.StateLR = nil
				baseline, err := BlockWeightShapes(b, 8, 4, 1, 16)
				if err != nil {
					t.Fatal(err)
				}
				seen := 0
				for i := range got {
					if got[i].Name == "A_log" || got[i].Name == "dt_bias" {
						seen++
						if got[i].OptimizerRole != "ssm_state" || got[i].OptimizerLR != 0.001 || !got[i].ForceNoDecay {
							t.Fatalf("bad state metadata: %+v", got[i])
						}
						got[i].OptimizerRole = ""
						got[i].OptimizerLR = 0
					}
				}
				if seen != 2 || !reflect.DeepEqual(got, baseline) {
					t.Fatal("state_lr changed unrelated metadata/layout")
				}
			})
		}
	}
}

func TestDynamicsStateLRWeightSharing(t *testing.T) {
	for _, kind := range []string{"s4d", "mamba3-canonical", "gated_deltanet"} {
		for _, second := range []string{`"state_lr":0.001,`, `"state_lr":0.002,`, ``} {
			raw := fmt.Sprintf(`{"model_dim":8,"vocab_size":16,"seq_len":4,"blocks":[
			{"type":%q,"heads":2,"d_k":4,"weight_group":"shared","state_lr":0.001},
			{%s"type":%q,"heads":2,"d_k":4,"weight_group":"shared"}],"training":{"batch_tokens":4}}`, kind, second, kind)
			_, err := ParseArchConfig([]byte(raw), "state-lr-sharing")
			if second == `"state_lr":0.001,` {
				if err != nil {
					t.Fatal(err)
				}
			} else if err == nil || !strings.Contains(err.Error(), "state_lr") {
				t.Fatalf("expected shared state_lr error: %v", err)
			}
		}
	}
}
