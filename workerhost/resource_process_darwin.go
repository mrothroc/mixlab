package workerhost

import "strconv"

func processSampleArgs(group int) []string {
	// Darwin ps(1) -g selects all members of the named process group. No -a/-A:
	// selection flags are additive and would broaden this to unrelated groups.
	return []string{"-g", strconv.Itoa(group), "-o", "pid=,pgid=,time=,rss=,lstart="}
}
