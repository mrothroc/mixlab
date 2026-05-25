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
	BlockIndex    int
	KVCache       map[int]BlockKVOutputs
}

// BlockKVOutputs tracks the named K/V tensors emitted by a plain attention block.
type BlockKVOutputs struct {
	K string
	V string
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
	key := strings.ToLower(strings.TrimSpace(name))
	if key == "mamba3" {
		return "gated_linear_ssm"
	}
	return key
}

func lookupBlock(spec BlockSpec) (BlockRegistration, error) {
	key := blockTypeName(spec.Type)
	reg, ok := registry[key]
	if !ok {
		return BlockRegistration{}, fmt.Errorf("unsupported block type %q", spec.Type)
	}
	return reg, nil
}

// EmitBlock emits the block specified by spec using the registry's registered
// emitter. It returns the new weight index after consuming this block's weights.
func EmitBlock(prog *Program, spec BlockSpec, stream string, wi, D, T, B, V, idx int, opts EmitOptions) (int, error) {
	reg, err := lookupBlock(spec)
	if err != nil {
		return wi, err
	}
	if reg.Emitter == nil {
		return wi, fmt.Errorf("block type %q has no registered emitter", spec.Type)
	}
	return reg.Emitter(prog, spec, stream, wi, D, T, B, V, idx, opts)
}

// BlockWeightShapes returns the weight metadata for a block via the registry's
// registered WeightShapes function.
func BlockWeightShapes(spec BlockSpec, D, T, B, V int) ([]WeightMeta, error) {
	reg, err := lookupBlock(spec)
	if err != nil {
		return nil, err
	}
	if reg.WeightShapes == nil {
		return nil, fmt.Errorf("block type %q has no registered weight shaper", spec.Type)
	}
	return reg.WeightShapes(spec, D, T, B, V)
}

func init() {
	RegisterBlock("plain", blockRegistration{
		Emitter: func(prog *Program, spec BlockSpec, stream string, wi, D, T, B, V, idx int, opts EmitOptions) (int, error) {
			heads := spec.Heads
			if heads <= 0 {
				heads = 4
			}
			return emitPlainAttentionIRWithKVOptions(prog, stream, wi, heads, spec.KVHeads, D, T, B, idx, opts.MLPMult, opts.BlockScales, opts.Dropout, spec.SkipAttention, spec.QKGain, spec.RopeDims, spec.XSA, spec.SparseAttnGate, spec.WindowSize, spec.KVSource, opts.KVCache, opts.BlockIndex)
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
	RegisterBlock("geglu", blockRegistration{
		Emitter: func(prog *Program, spec BlockSpec, stream string, wi, D, T, B, V, idx int, opts EmitOptions) (int, error) {
			return emitGEGLUIRWithDropout(prog, stream, wi, idx, opts.MLPMult, opts.BlockScales, opts.Dropout)
		},
		WeightCount: swigluWeightCount,
		WeightShapes: func(spec BlockSpec, D, T, B, V int) ([]WeightMeta, error) {
			return builtinBlockWeightShapes(spec, D, T, B, V, DefaultFFNMultiplier, false, false)
		},
		weightShapesWithOptions: func(spec BlockSpec, D, T, B, V int, opts EmitOptions) ([]WeightMeta, error) {
			return builtinBlockWeightShapes(spec, D, T, B, V, opts.MLPMult, opts.BlockScales, opts.ResidMix)
		},
	})
	RegisterBlock("mlp", blockRegistration{
		Emitter: func(prog *Program, spec BlockSpec, stream string, wi, D, T, B, V, idx int, opts EmitOptions) (int, error) {
			return emitMLPIR(prog, stream, wi, idx, spec.Activation, spec.LeakySlope)
		},
		WeightCount: mlpWeightCount,
		WeightShapes: func(spec BlockSpec, D, T, B, V int) ([]WeightMeta, error) {
			return builtinBlockWeightShapes(spec, D, T, B, V, DefaultFFNMultiplier, false, false)
		},
		weightShapesWithOptions: func(spec BlockSpec, D, T, B, V int, opts EmitOptions) ([]WeightMeta, error) {
			return builtinBlockWeightShapes(spec, D, T, B, V, opts.MLPMult, opts.BlockScales, opts.ResidMix)
		},
	})
	RegisterBlock("gated_deltanet", blockRegistration{
		Emitter: func(prog *Program, spec BlockSpec, stream string, wi, D, T, B, V, idx int, _ EmitOptions) (int, error) {
			return emitGatedDeltaNetIR(prog, spec, stream, wi, D, T, B, idx)
		},
		WeightCount:  gatedDeltaNetWeightCount,
		WeightShapes: gatedDeltaNetWeightShapes,
		weightShapesWithOptions: func(spec BlockSpec, D, T, B, V int, _ EmitOptions) ([]WeightMeta, error) {
			return gatedDeltaNetWeightShapes(spec, D, T, B, V)
		},
	})
	for _, name := range []string{"mamba", "gated_linear_ssm", "mamba3", "mamba3-canonical", "rwkv", "perceiver", "bottleneck", "retnet", "cross_attention", "token_blend", "custom"} {
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

func plainWeightCount(spec BlockSpec, blockScales, residMix bool) (int, error) {
	total := 7
	if spec.KVSource > 0 {
		total -= 2
	}
	if spec.QKGain > 0 {
		total++
	}
	if spec.SparseAttnGate {
		total++
	}
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

func mlpWeightCount(_ BlockSpec, _, _ bool) (int, error) {
	return 3, nil
}

func builtinBlockWeightCount(spec BlockSpec, blockScales, residMix bool) (int, error) {
	switch blockTypeName(spec.Type) {
	case "plain":
		return plainWeightCount(spec, blockScales, residMix)
	case "swiglu", "geglu":
		return swigluWeightCount(spec, blockScales, residMix)
	case "mlp":
		return mlpWeightCount(spec, blockScales, residMix)
	case "mamba":
		return 4, nil
	case "gated_linear_ssm":
		return 6, nil
	case "mamba3-canonical":
		total := 19
		if spec.UseConv == nil || *spec.UseConv {
			total++
		}
		return total, nil
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
	case "gated_linear_ssm":
		inner := spec.InnerDim
		if inner <= 0 {
			inner = D
		}
		return emitMamba3IR(prog, stream, wi, inner, T, B, idx)
	case "mamba3-canonical":
		inner := spec.InnerDim
		if inner <= 0 {
			inner = D
		}
		stateSize := spec.StateSize
		if stateSize <= 0 {
			stateSize = 16
		}
		nGroups := spec.NGroups
		if nGroups <= 0 {
			nGroups = 4
		}
		dtRank := spec.DTRank
		if dtRank <= 0 {
			dtRank = defaultMamba3CanonicalRank(inner)
		}
		convKernel := spec.ConvKernel
		if convKernel <= 0 {
			convKernel = 4
		}
		useConv := spec.UseConv == nil || *spec.UseConv
		scanChunkSize := effectiveMamba3CanonicalScanChunkSize(spec)
		return emitMamba3CanonicalIR(prog, stream, wi, inner, stateSize, nGroups, dtRank, convKernel, useConv, scanChunkSize, T, B)
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
