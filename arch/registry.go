package arch

import (
	"fmt"
	"strings"
)

// EmitOptions carries optional context for block emitters.
type EmitOptions struct {
	StreamSeqLens map[string]int
	MLPMult       float64
	BlockScales   bool
	ResidMix      bool
	Dropout       float32
}

// BlockEmitter emits IR ops for a block and returns the next weight index.
type BlockEmitter func(prog *Program, spec BlockSpec, stream string, wi, D, T, B, V, idx int, opts EmitOptions) (int, error)

// BlockWeightCounter returns the number of weights a block consumes.
type BlockWeightCounter func(spec BlockSpec, blockScales, residMix bool) (int, error)

// BlockWeightShaper returns weight metadata for a block.
type BlockWeightShaper func(spec BlockSpec, D, T, B, V int) ([]WeightMeta, error)

type blockWeightShaperWithOptions func(spec BlockSpec, D, T, B, V int, opts EmitOptions) ([]WeightMeta, error)

// BlockRegistration describes how a block type participates in IR construction.
type BlockRegistration struct {
	Emitter      BlockEmitter
	WeightCount  BlockWeightCounter
	WeightShapes BlockWeightShaper

	weightShapesWithOptions blockWeightShaperWithOptions
}

type blockRegistration = BlockRegistration

var registry = map[string]BlockRegistration{}

// RegisterBlock registers or replaces a block type.
func RegisterBlock(name string, reg BlockRegistration) {
	registry[blockTypeName(name)] = reg
}

func blockTypeName(name string) string {
	return strings.ToLower(strings.TrimSpace(name))
}

func lookupBlock(spec BlockSpec) (BlockRegistration, error) {
	key := blockTypeName(spec.Type)
	reg, ok := registry[key]
	if !ok {
		return BlockRegistration{}, fmt.Errorf("unsupported block type %q", spec.Type)
	}
	return reg, nil
}

func init() {
	RegisterBlock("plain", blockRegistration{
		Emitter: func(prog *Program, spec BlockSpec, stream string, wi, D, T, B, V, idx int, opts EmitOptions) (int, error) {
			heads := spec.Heads
			if heads <= 0 {
				heads = 4
			}
			return emitPlainAttentionIRWithDropout(prog, stream, wi, heads, spec.KVHeads, D, T, B, idx, opts.MLPMult, opts.BlockScales, opts.Dropout)
		},
		WeightCount: plainWeightCount,
		WeightShapes: func(spec BlockSpec, D, T, B, V int) ([]WeightMeta, error) {
			return builtinBlockWeightShapes(spec, D, T, B, V, DefaultFFNMultiplier, false, false)
		},
		weightShapesWithOptions: func(spec BlockSpec, D, T, B, V int, opts EmitOptions) ([]WeightMeta, error) {
			return builtinBlockWeightShapes(spec, D, T, B, V, opts.MLPMult, opts.BlockScales, opts.ResidMix)
		},
	})
	RegisterBlock("swiglu", blockRegistration{
		Emitter: func(prog *Program, spec BlockSpec, stream string, wi, D, T, B, V, idx int, opts EmitOptions) (int, error) {
			return emitSwiGLUIRWithDropout(prog, stream, wi, idx, opts.MLPMult, opts.BlockScales, opts.Dropout)
		},
		WeightCount: swigluWeightCount,
		WeightShapes: func(spec BlockSpec, D, T, B, V int) ([]WeightMeta, error) {
			return builtinBlockWeightShapes(spec, D, T, B, V, DefaultFFNMultiplier, false, false)
		},
		weightShapesWithOptions: func(spec BlockSpec, D, T, B, V int, opts EmitOptions) ([]WeightMeta, error) {
			return builtinBlockWeightShapes(spec, D, T, B, V, opts.MLPMult, opts.BlockScales, opts.ResidMix)
		},
	})
	for _, name := range []string{"mamba", "mamba3", "rwkv", "perceiver", "bottleneck", "retnet", "cross_attention", "token_blend", "custom"} {
		RegisterBlock(name, blockRegistration{
			Emitter:     builtinBlockEmitter,
			WeightCount: builtinBlockWeightCount,
			WeightShapes: func(spec BlockSpec, D, T, B, V int) ([]WeightMeta, error) {
				return builtinBlockWeightShapes(spec, D, T, B, V, DefaultFFNMultiplier, false, false)
			},
			weightShapesWithOptions: func(spec BlockSpec, D, T, B, V int, opts EmitOptions) ([]WeightMeta, error) {
				return builtinBlockWeightShapes(spec, D, T, B, V, opts.MLPMult, opts.BlockScales, opts.ResidMix)
			},
		})
	}
}

func plainWeightCount(_ BlockSpec, blockScales, residMix bool) (int, error) {
	total := 7
	if blockScales {
		total += 2
	}
	if residMix {
		total++
	}
	return total, nil
}

func swigluWeightCount(_ BlockSpec, blockScales, _ bool) (int, error) {
	total := 4
	if blockScales {
		total++
	}
	return total, nil
}

func builtinBlockWeightCount(spec BlockSpec, blockScales, residMix bool) (int, error) {
	switch blockTypeName(spec.Type) {
	case "plain":
		return plainWeightCount(spec, blockScales, residMix)
	case "swiglu":
		return swigluWeightCount(spec, blockScales, residMix)
	case "mamba":
		return 4, nil
	case "mamba3":
		return 6, nil
	case "rwkv":
		return 10, nil
	case "perceiver", "bottleneck":
		return 15, nil
	case "retnet":
		return 8, nil
	case "cross_attention":
		return 7, nil
	case "token_blend":
		return 1, nil
	case "custom":
		return len(spec.Weights), nil
	default:
		return 0, fmt.Errorf("unsupported block type %q", spec.Type)
	}
}

func builtinBlockEmitter(prog *Program, spec BlockSpec, stream string, wi, D, T, B, V, idx int, opts EmitOptions) (int, error) {
	switch blockTypeName(spec.Type) {
	case "mamba":
		inner := spec.InnerDim
		if inner <= 0 {
			inner = D
		}
		return emitMambaIR(prog, stream, wi, inner, T, B, idx)
	case "mamba3":
		inner := spec.InnerDim
		if inner <= 0 {
			inner = D
		}
		return emitMamba3IR(prog, stream, wi, inner, T, B, idx)
	case "rwkv":
		return emitRWKVIR(prog, stream, wi, D, T, B, idx)
	case "perceiver":
		L := spec.NumLatents
		if L <= 0 {
			L = 32
		}
		heads := spec.Heads
		if heads <= 0 {
			heads = 4
		}
		return emitPerceiverIR(prog, stream, wi, heads, L, D, T, B, idx)
	case "bottleneck":
		L := spec.NumLatents
		if L <= 0 {
			L = 4
		}
		heads := spec.Heads
		if heads <= 0 {
			heads = 4
		}
		return emitPerceiverIR(prog, stream, wi, heads, L, D, T, B, idx)
	case "retnet":
		heads := spec.Heads
		if heads <= 0 {
			heads = 4
		}
		return emitRetNetIR(prog, stream, wi, heads, D, T, B, idx)
	case "cross_attention":
		heads := spec.Heads
		if heads <= 0 {
			heads = 4
		}
		kvStream := spec.SourceStream
		if kvStream == "" {
			return wi, fmt.Errorf("cross_attention block requires source_stream")
		}
		Tkv, ok := opts.StreamSeqLens[kvStream]
		if !ok {
			return wi, fmt.Errorf("cross_attention source_stream %q not found in stream map", kvStream)
		}
		return emitCrossAttentionIR(prog, stream, kvStream, wi, heads, D, T, Tkv, B, idx)
	case "token_blend":
		return emitTokenBlendIR(prog, stream, wi, D, T, B, idx)
	case "custom":
		return emitCustomBlockIR(prog, spec, stream, wi, D, T, B, V, idx)
	default:
		return wi, fmt.Errorf("unsupported IR block type %q", spec.Type)
	}
}
