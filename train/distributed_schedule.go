package train

type gpuDistributedContextReporter interface {
	DistributedContextActiveGPU() bool
}

func distributedContextActive(trainer any) bool {
	reporter, ok := trainer.(gpuDistributedContextReporter)
	return ok && reporter.DistributedContextActiveGPU()
}

func trainerAllowsStepLookahead(trainer any, configured bool) bool {
	return configured && !distributedContextActive(trainer)
}
