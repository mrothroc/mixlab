package train

import "github.com/mrothroc/mixlab/gpu"

// ConfigureCUDAGraphLimits performs a best-effort preflight from architecture
// config to MLX CUDA graph limits. It must run before MLX initializes.
func ConfigureCUDAGraphLimits(configPath, configsDir string) {
	gpu.ApplyCUDAGraphLimits(cudaGraphLimitsForSelection(configPath, configsDir))
}

func cudaGraphLimitsForSelection(configPath, configsDir string) gpu.CUDAGraphLimits {
	if configPath != "" {
		return cudaGraphLimitsForConfigPath(configPath)
	}
	if configsDir != "" {
		return cudaGraphLimitsForConfigDir(configsDir)
	}
	return gpu.CUDAGraphLimits{}
}

func cudaGraphLimitsForConfigPath(path string) gpu.CUDAGraphLimits {
	// Quiet load: this preflight parses the config only to size CUDA graph
	// limits and runs before the primary, user-facing load. Emitting advisory
	// config warnings here would duplicate them.
	cfg, err := LoadArchConfigQuiet(path)
	if err != nil {
		return gpu.CUDAGraphLimits{}
	}
	return cudaGraphLimitsForConfig(cfg)
}

func cudaGraphLimitsForConfig(cfg *ArchConfig) gpu.CUDAGraphLimits {
	prog, err := BuildIRProgramFromConfig(cfg)
	if err != nil {
		return gpu.CUDAGraphLimits{}
	}
	return gpu.TuneCUDAGraphLimits(prog)
}

func cudaGraphLimitsForConfigDir(dir string) gpu.CUDAGraphLimits {
	// Quiet load: preflight only; the arch_race run surfaces config warnings.
	configs, err := loadConfigsFromDirQuiet(dir)
	if err != nil {
		return gpu.CUDAGraphLimits{}
	}
	limits := gpu.CUDAGraphLimits{}
	for _, cfg := range configs {
		limits = mergeCUDAGraphLimits(limits, cudaGraphLimitsForConfig(cfg))
	}
	return limits
}

func mergeCUDAGraphLimits(a, b gpu.CUDAGraphLimits) gpu.CUDAGraphLimits {
	if a == (gpu.CUDAGraphLimits{}) {
		return b
	}
	if b == (gpu.CUDAGraphLimits{}) {
		return a
	}

	out := gpu.CUDAGraphLimits{
		MaxOpsPerBuffer: max(a.MaxOpsPerBuffer, b.MaxOpsPerBuffer),
		MaxMBPerBuffer:  max(a.MaxMBPerBuffer, b.MaxMBPerBuffer),
		GraphCacheSize:  max(a.GraphCacheSize, b.GraphCacheSize),
	}
	if out.MaxMBPerBuffer > 0 {
		out.MaxOpsPerBuffer = minPositive(a.MaxOpsPerBuffer, b.MaxOpsPerBuffer)
	}
	return out
}

func minPositive(a, b int) int {
	switch {
	case a <= 0:
		return b
	case b <= 0:
		return a
	case a < b:
		return a
	default:
		return b
	}
}
