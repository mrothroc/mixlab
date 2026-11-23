package gpu

import (
	"os"
	"runtime"
	"strconv"

	ir "github.com/mrothroc/mixlab/arch"
)

const (
	MLXMaxOpsPerBufferEnv = "MLX_MAX_OPS_PER_BUFFER"
	MLXMaxMBPerBufferEnv  = "MLX_MAX_MB_PER_BUFFER"
	MLXCUDAGraphCacheEnv  = "MLX_CUDA_GRAPH_CACHE_SIZE"

	minGatedDeltaNetMaxOpsPerBuffer = 16000

	// Canonical Mamba3 expands each selective-scan IR op into a large MLX graph.
	// Keeping CUDA graph batches modest avoids cudaGraphInstantiate OOM on
	// H100-scale D=448/T=4096 runs without changing canonical Mamba3 math.
	maxMamba3SelectiveScanOpsPerBuffer = 64
	maxMamba3SelectiveScanMBPerBuffer  = 128
	mamba3CUDAGraphCacheSize           = 1024

	// TTT-MLP expands each scan into a chunk-local differentiable recurrence.
	// Bound CUDA graph construction while retaining enough variants for the
	// stateful decoder's chunk offsets and fragment lengths.
	maxTTTMLPScanOpsPerBuffer = 128
	maxTTTMLPScanMBPerBuffer  = 128
	tttMLPCUDAGraphCacheSize  = 512

	// Polar graphs retain large feature maps alongside convolution adjoints.
	// Pair these bounds with the CUDA image's tiled convolution weight VJP.
	maxPolarOpsPerBuffer = 20
	maxPolarMBPerBuffer  = 100
	polarGraphCacheSize  = 64
)

type CUDAGraphLimits struct {
	MaxOpsPerBuffer int
	MaxMBPerBuffer  int
	GraphCacheSize  int
}

// TuneCUDAGraphLimits derives MLX CUDA graph batching limits from backend IR.
// It is pure policy: no GPU, no CGO, and no process environment mutation.
func TuneCUDAGraphLimits(prog *ir.Program) CUDAGraphLimits {
	return tuneCUDAGraphLimitsForOS(prog, runtime.GOOS)
}

func tuneCUDAGraphLimitsForOS(prog *ir.Program, goos string) CUDAGraphLimits {
	if prog == nil {
		return CUDAGraphLimits{}
	}

	maxOps := len(prog.Ops) * 3 // forward + backward + optimizer margin
	hasMamba := programHasOp(prog, ir.OpMamba3SelectiveScan) || programHasOp(prog, ir.OpMamba3CanonicalBlock)
	hasTTT := programHasOp(prog, ir.OpTTTMLPScan) || programHasOp(prog, ir.OpTTTMLPStatefulScan)
	if !hasMamba && !hasTTT && programHasOp(prog, ir.OpGatedDeltaScan) && maxOps < minGatedDeltaNetMaxOpsPerBuffer {
		maxOps = minGatedDeltaNetMaxOpsPerBuffer
	}
	limits := CUDAGraphLimits{MaxOpsPerBuffer: maxOps}
	capLimit := func(current *int, cap int) {
		if *current == 0 || *current > cap {
			*current = cap
		}
	}
	capAll := func(ops, mb, cache int) {
		capLimit(&limits.MaxOpsPerBuffer, ops)
		capLimit(&limits.MaxMBPerBuffer, mb)
		capLimit(&limits.GraphCacheSize, cache)
	}
	if hasMamba {
		capAll(maxMamba3SelectiveScanOpsPerBuffer, maxMamba3SelectiveScanMBPerBuffer, mamba3CUDAGraphCacheSize)
	} else if hasTTT {
		capAll(maxTTTMLPScanOpsPerBuffer, maxTTTMLPScanMBPerBuffer, tttMLPCUDAGraphCacheSize)
	}
	// MLX's unprefixed buffer env vars also affect Metal. This new memory
	// policy is CUDA-only; preserve the established Metal graph batching.
	if goos == "linux" && (programHasOp(prog, ir.OpCartesianToPolar) || programHasOp(prog, ir.OpPolarToCartesian)) {
		capAll(maxPolarOpsPerBuffer, maxPolarMBPerBuffer, polarGraphCacheSize)
	}
	return limits
}

// ApplyCUDAGraphLimits applies missing MLX CUDA graph env vars only. Explicit
// user-provided values are preserved.
func ApplyCUDAGraphLimits(limits CUDAGraphLimits) {
	maxOps := limits.MaxOpsPerBuffer
	if os.Getenv(MLXMaxOpsPerBufferEnv) != "" {
		maxOps = 0
	}
	maxMB := limits.MaxMBPerBuffer
	if os.Getenv(MLXMaxMBPerBufferEnv) != "" {
		maxMB = 0
	}
	graphCacheSize := limits.GraphCacheSize
	if os.Getenv(MLXCUDAGraphCacheEnv) != "" {
		graphCacheSize = 0
	}
	SetCUDAGraphLimits(maxOps, maxMB)
	setCUDAGraphCacheSizeEnv(graphCacheSize)
}

func setCUDAGraphLimitEnv(maxOps, maxMB int) {
	if maxOps > 0 {
		_ = os.Setenv(MLXMaxOpsPerBufferEnv, strconv.Itoa(maxOps))
	}
	if maxMB > 0 {
		_ = os.Setenv(MLXMaxMBPerBufferEnv, strconv.Itoa(maxMB))
	}
}

func setCUDAGraphCacheSizeEnv(size int) {
	if size > 0 {
		_ = os.Setenv(MLXCUDAGraphCacheEnv, strconv.Itoa(size))
	}
}

func programHasOp(prog *ir.Program, code int) bool {
	if prog == nil {
		return false
	}
	for _, op := range prog.Ops {
		if op.Code == code {
			return true
		}
	}
	return false
}
