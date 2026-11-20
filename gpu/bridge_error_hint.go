package gpu

import "strings"

const mlxCacheThrashMarker = "Cache thrashing is happening"

// explainBridgeError keeps MLX's own error text and, for a CUDA graph cache
// thrash, appends the batching settings in effect. MLX names only the cache
// size, but a thrash can also come from batches small enough that one step
// needs more distinct graphs than the cache holds.
func explainBridgeError(message string, lookup func(string) (string, bool)) string {
	if !strings.Contains(message, mlxCacheThrashMarker) {
		return message
	}
	current := func(key string) string {
		if v, ok := lookup(key); ok && v != "" {
			return v
		}
		return "unset"
	}
	return message + " The MLX CUDA buffer-batching limits may be too small for this workload: " +
		"increase " + MLXMaxOpsPerBufferEnv + " (current " + current(MLXMaxOpsPerBufferEnv) + ") and/or " +
		MLXMaxMBPerBufferEnv + " (current " + current(MLXMaxMBPerBufferEnv) + "), and set " +
		MLXCUDAGraphCacheEnv + " (current " + current(MLXCUDAGraphCacheEnv) + ") above the upstream default. " +
		"For canonical Mamba3, mixlab auto-tunes these before MLX initializes unless the environment " +
		"already overrides them."
}
