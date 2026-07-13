//nolint:unused // MLX-only fields and helpers are used when built with -tags mlx.
package gpu

import (
	"fmt"
	"sync"
	"unsafe"
)

const (
	OpEmbed                = 1
	OpMatMul               = 2
	OpAdd                  = 3
	OpMul                  = 4
	OpScalarMul            = 5
	OpSigmoid              = 6
	OpSiLU                 = 7
	OpSoftmax              = 8
	OpReshape              = 9
	OpTranspose            = 10
	OpSlice                = 11
	OpConcat               = 12
	OpCausalMask           = 13
	OpCrossEntropy         = 14
	OpDropout              = 15
	OpSquare               = 20
	OpSub                  = 21
	OpDiv                  = 22
	OpWhere                = 25
	OpLessThan             = 26
	OpGreaterEq            = 27
	OpArange               = 28
	OpMeanAxis             = 29
	OpFull                 = 30
	OpRMSNorm              = 33
	OpRoPE                 = 34
	OpSqrt                 = 35
	OpRSqrt                = 36
	OpExp                  = 39
	OpOuter                = 40
	OpGELU                 = 42
	OpReLU                 = 43
	OpTanh                 = 44
	OpScan                 = 49
	OpGatherPositions      = 51
	OpScatterPositions     = 52
	OpRoPEIndexed          = 53
	OpLeakyReLU            = 54
	OpXSAProject           = 55
	OpCrossEntropyPerToken = 56
	OpMatrixScan           = 57
	OpScanTV               = 58
	OpSoftplus             = 59
	OpGatedDeltaScan       = 60
	OpStopGradient         = 61
	OpDepthwiseConv1D      = 62
	OpMamba3SelectiveScan  = 63
	OpMamba3CanonicalBlock = 64
	OpRandomNormal         = 65
	OpFirstByteMaskedCE    = 66
	OpMaskedCrossEntropy   = 67
	OpMaskedCEPerToken     = 68
	OpDistillationKL       = 69
	OpHGRN2Scan            = 70
	OpMLSTMScan            = 71
	OpDebertaRelativeBias  = 72
	OpCharFeatureBag       = 73
	OpMoEFeedForward       = 74
	OpMaskedSmoothL1       = 75
	OpZLoss                = 76
	OpLog                  = 77
	OpReciprocal           = 78
	OpPow                  = 79
	OpAbs                  = 80
	OpClamp                = 81
	OpMinimum              = 82
	OpMaximum              = 83
	OpGreaterThan          = 84
	OpLessEq               = 85
	OpEqual                = 86
	OpLayerNorm            = 87
	OpSelectiveCausalMask  = 88
	OpSegmentAttentionMask = 89
	OpBlockDiffusionMask   = 90
	OpGELUExact            = 91
	OpMaskedBCEWithLogits  = 92
	OpMaskedBinaryAccuracy = 93
	OpEnergyPairwiseLoss   = 94
	OpEnergySpanPool       = 95
	OpEnergySpanPairwise   = 96
	OpSpanPLLPool          = 97
	OpSpanPLLPairwise      = 98
	OpMaskedDistillationKL = 99
	OpMaskedSymmetricKL    = 100
	OpMaskedMarginPLL      = 101
	OpMaskedZLoss          = 102
	OpTTTMLPScan           = 103
	OpTTTMLPStatefulScan   = 104
)

// HandleInput binds an existing GPU array handle to a declared IR input.
// The handle remains owned by the caller.
type HandleInput struct {
	Name   string
	Handle int64
}

const (
	TensorInt32   = 0
	TensorFloat32 = 1
)

type TensorInput struct {
	Name  string
	DType int // 0=Int32, 1=Float32
	Shape []int
	Data  interface{} // []int32 or []float32
}

type MemoryStats struct {
	ActiveBytes uint64
	CacheBytes  uint64
	PeakBytes   uint64
}

type TrainerCompileStats struct {
	TrainingStepCacheHits         uint64
	TrainingStepCacheMisses       uint64
	CategoricalSamplerCacheHits   uint64
	CategoricalSamplerCacheMisses uint64
}

type TrainerOptimizerStats struct {
	AttemptedSteps        uint64
	CommittedSteps        uint64
	SkippedSteps          uint64
	ConsecutiveSkipped    uint64
	LastStepSkipped       bool
	LastLossNonfinite     uint64
	LastGradientNonfinite uint64
	LastStateNonfinite    uint64
}

type TrainerBackwardTraceStats struct {
	BadEdges              uint64
	FirstForwardBadOp     int
	FirstForwardBadOpType int
	FirstForwardBadOutput int
	FirstBadOp            int
	FirstBadOpType        int
	FirstBadInput         int
}

type Program struct {
	handle      int64
	nWeights    int
	opCount     int
	opTypes     []int
	inputDecls  map[string]TensorInput
	outputDecls map[string]TensorInput
	outputOrder []string
}

func (p *Program) operationCount() int {
	if p == nil {
		return 0
	}
	return p.opCount
}

func (p *Program) operationTypes() []int {
	if p == nil || len(p.opTypes) == 0 {
		return nil
	}
	out := make([]int, len(p.opTypes))
	copy(out, p.opTypes)
	return out
}

type TrainerHandle int64

type ComputeDType int

const (
	ComputeDTypeFloat32 ComputeDType = iota
	ComputeDTypeBF16
)

type OptimizerKind int

const (
	OptimizerAdamW OptimizerKind = iota
	OptimizerMuon
	OptimizerSGD
	OptimizerLAMB
)

type NewtonSchulzVariant int

const (
	NewtonSchulzFixed NewtonSchulzVariant = iota
	NewtonSchulzPolarExpress
)

type MuonNormalization int

const (
	MuonNormalizationNone MuonNormalization = iota
	MuonNormalizationRowL2
	MuonNormalizationNorMuon
)

type OptimizerGroup struct {
	Kind                              OptimizerKind
	LR                                float32
	Beta1                             float32
	Beta2                             float32
	Epsilon                           float32
	WeightDecay                       float32
	LAMBTrustRatioCap                 float32
	CautiousWeightDecay               bool
	CautiousWeightDecayActivationStep int
	BackendSteps                      int
	NewtonSchulzVariant               NewtonSchulzVariant
	Nesterov                          bool
	MuonNormalization                 MuonNormalization
	RowNormalize                      bool
}

type WeightOptimizer struct {
	GroupIndex int
	Decay      bool
}

type TrainerOptimizerSpec struct {
	Groups        []OptimizerGroup
	Weights       []WeightOptimizer
	MaxGradNorm   float32
	DefaultBaseLR float32
	ComputeDType  ComputeDType
}

var (
	handleMetaMu sync.RWMutex
	handleSizes  = map[int64]int{}
)

func setHandleSize(handle int64, size int) {
	handleMetaMu.Lock()
	handleSizes[handle] = size
	handleMetaMu.Unlock()
}

func clearHandleSize(handle int64) {
	handleMetaMu.Lock()
	delete(handleSizes, handle)
	handleMetaMu.Unlock()
}

func getHandleSize(handle int64) (int, bool) {
	handleMetaMu.RLock()
	size, ok := handleSizes[handle]
	handleMetaMu.RUnlock()
	return size, ok
}

func cloneShape(shape []int) []int {
	if len(shape) == 0 {
		return nil
	}
	out := make([]int, len(shape))
	copy(out, shape)
	return out
}

func validateTensorDecl(name string, dtype int, shape []int) error {
	if name == "" {
		return fmt.Errorf("tensor declaration missing name")
	}
	if dtype != TensorInt32 && dtype != TensorFloat32 {
		return fmt.Errorf("tensor %q has unsupported dtype=%d", name, dtype)
	}
	if len(shape) == 0 {
		return fmt.Errorf("tensor %q has empty shape", name)
	}
	for i, dim := range shape {
		if dim <= 0 {
			return fmt.Errorf("tensor %q has invalid shape[%d]=%d", name, i, dim)
		}
	}
	return nil
}

func cBytesFromInt32(data []int32) []byte {
	if len(data) == 0 {
		return nil
	}
	n := len(data) * 4
	return (*[1 << 30]byte)(unsafe.Pointer(&data[0]))[:n:n]
}

func cBytesFromFloat32(data []float32) []byte {
	if len(data) == 0 {
		return nil
	}
	n := len(data) * 4
	return (*[1 << 30]byte)(unsafe.Pointer(&data[0]))[:n:n]
}

func shapeEquals(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func shapeElemCount(shape []int) (int, error) {
	if len(shape) == 0 {
		return 0, fmt.Errorf("empty shape")
	}
	total := 1
	for i, dim := range shape {
		if dim <= 0 {
			return 0, fmt.Errorf("invalid shape[%d]=%d", i, dim)
		}
		total *= dim
	}
	return total, nil
}

func (p *Program) validateDeclaredInputs(inputs []TensorInput) error {
	if p == nil || len(p.inputDecls) == 0 {
		return nil
	}
	byName := make(map[string]TensorInput, len(inputs))
	for _, in := range inputs {
		byName[in.Name] = in
	}
	for name, decl := range p.inputDecls {
		got, ok := byName[name]
		if !ok {
			return fmt.Errorf("missing required input %q", name)
		}
		if got.DType != decl.DType {
			return fmt.Errorf("input %q dtype mismatch: got=%d want=%d", name, got.DType, decl.DType)
		}
		if !shapeEquals(got.Shape, decl.Shape) {
			return fmt.Errorf("input %q shape mismatch: got=%v want=%v", name, got.Shape, decl.Shape)
		}
	}
	return nil
}

func (p *Program) validateDeclaredOutput(name string, out []float32) error {
	if p == nil || len(p.outputDecls) == 0 {
		return nil
	}
	decl, ok := p.outputDecls[name]
	if !ok {
		return nil
	}
	if decl.DType != TensorFloat32 {
		return fmt.Errorf("output %q dtype=%d unsupported by float32 bridge", name, decl.DType)
	}
	want, err := shapeElemCount(decl.Shape)
	if err != nil {
		return fmt.Errorf("output %q invalid declared shape: %w", name, err)
	}
	if len(out) != want {
		return fmt.Errorf("output %q size mismatch: got=%d want=%d", name, len(out), want)
	}
	return nil
}
