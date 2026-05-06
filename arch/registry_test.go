package arch

import "testing"

func TestRegistryBuiltinsRegistered(t *testing.T) {
	for _, name := range []string{"plain", "swiglu", "mlp", "mamba", "gated_linear_ssm", "mamba3", "mamba3-canonical", "rwkv", "perceiver", "bottleneck", "retnet", "cross_attention", "token_blend", "custom"} {
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

func TestEmitBlock_DelegatesToRegisteredEmitter(t *testing.T) {
	prog := NewProgram(7)
	spec := BlockSpec{Type: "plain", Heads: 2}
	wi, err := EmitBlock(prog, spec, "x", 0, 64, 16, 1, 1024, 0, EmitOptions{MLPMult: 2.67})
	if err != nil {
		t.Fatalf("EmitBlock(plain): %v", err)
	}
	if wi != 7 {
		t.Errorf("plain emitter consumed %d weights, want 7", wi)
	}

	hasNorm := false
	for _, op := range prog.Ops {
		if op.Code == OpRMSNorm {
			hasNorm = true
		}
	}
	if !hasNorm {
		t.Error("plain block missing RMSNorm op")
	}
}

func TestEmitBlock_UnregisteredTypeReturnsError(t *testing.T) {
	prog := NewProgram(0)
	spec := BlockSpec{Type: "nonexistent_block_type"}
	_, err := EmitBlock(prog, spec, "x", 0, 64, 16, 1, 1024, 0, EmitOptions{})
	if err == nil {
		t.Fatal("expected error for unregistered block type, got nil")
	}
}
