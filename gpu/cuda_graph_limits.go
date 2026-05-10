package gpu

import (
	"os"
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
)

type CUDAGraphLimits struct {
	MaxOpsPerBuffer int
	MaxMBPerBuffer  int
	GraphCacheSize  int
}

// TuneCUDAGraphLimits derives MLX CUDA graph batching limits from backend IR.
// It is pure policy: no GPU, no CGO, and no process environment mutation.
func TuneCUDAGraphLimits(prog *ir.Program) CUDAGraphLimits {
	if prog == nil {
		return CUDAGraphLimits{}
	}

	maxOps := len(prog.Ops) * 3 // forward + backward + optimizer margin
	if programHasOp(prog, ir.OpMamba3SelectiveScan) || programHasOp(prog, ir.OpMamba3CanonicalBlock) {
		if maxOps > maxMamba3SelectiveScanOpsPerBuffer {
			maxOps = maxMamba3SelectiveScanOpsPerBuffer
		}
		return CUDAGraphLimits{
			MaxOpsPerBuffer: maxOps,
			MaxMBPerBuffer:  maxMamba3SelectiveScanMBPerBuffer,
			GraphCacheSize:  mamba3CUDAGraphCacheSize,
		}
	}
	if programHasOp(prog, ir.OpGatedDeltaScan) && maxOps < minGatedDeltaNetMaxOpsPerBuffer {
		// GatedDeltaScan is a compact IR op that expands into a much larger MLX
		// graph on Metal/CUDA, so raw IR op count can under-size MLX buffers.
		maxOps = minGatedDeltaNetMaxOpsPerBuffer
	}
	return CUDAGraphLimits{MaxOpsPerBuffer: maxOps}
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
