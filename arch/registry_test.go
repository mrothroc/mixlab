package arch

import "testing"

func TestRegistryBuiltinsRegistered(t *testing.T) {
	for _, name := range []string{"plain", "swiglu", "mlp", "mamba", "mamba3", "rwkv", "perceiver", "bottleneck", "retnet", "cross_attention", "token_blend", "custom"} {
		reg, err := lookupBlock(BlockSpec{Type: name})
		if err != nil {
			t.Fatalf("lookupBlock(%q): %v", name, err)
		}
		if reg.Emitter == nil || reg.WeightCount == nil || reg.WeightShapes == nil {
			t.Fatalf("registration %q is incomplete: %+v", name, reg)
		}
	}
}

func TestRegisterBlockOverridesLookup(t *testing.T) {
	const name = "registry_test_block"
	RegisterBlock(name, BlockRegistration{
		Emitter: func(_ *Program, _ BlockSpec, _ string, wi, _, _, _, _, _ int, _ EmitOptions) (int, error) {
			return wi + 2, nil
		},
		WeightCount: func(BlockSpec, bool, bool) (int, error) {
			return 2, nil
		},
		WeightShapes: func(BlockSpec, int, int, int, int) ([]WeightMeta, error) {
			return []WeightMeta{{Name: "a"}, {Name: "b"}}, nil
		},
	})

	n, err := BlockWeightCount(BlockSpec{Type: name}, false, false)
	if err != nil {
		t.Fatalf("BlockWeightCount: %v", err)
	}
	if n != 2 {
		t.Fatalf("BlockWeightCount=%d want 2", n)
	}

	next, err := emitBlockIR(NewProgram(0), BlockSpec{Type: name}, "x", 4, 8, 4, 1, 16, 0, nil, DefaultFFNMultiplier, false)
	if err != nil {
		t.Fatalf("emitBlockIR: %v", err)
	}
	if next != 6 {
		t.Fatalf("next weight index=%d want 6", next)
	}
}
