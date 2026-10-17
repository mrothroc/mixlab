package workerhost

func processSampleArgs(_ int) []string {
	// Linux ps -g does not share Darwin's process-group semantics. Keep the
	// portable full snapshot and filter PGID in parseProcessUsage.
	return []string{"-axo", "pid=,pgid=,time=,rss=,lstart="}
}
