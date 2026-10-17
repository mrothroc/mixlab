package workerhost

import "strconv"

func processSampleArgs(group int) []string {
	// Darwin ps(1) -g selects process group leaders. Do not add -a or -A:
	// selection flags are additive and would broaden this to unrelated groups.
	return []string{"-g", strconv.Itoa(group), "-o", "pid=,pgid=,time=,rss=,lstart="}
}
