// Package executionlimits defines bounded resource budgets shared by job
// admission and local hosting. It contains no enforcement or scheduling policy.
package executionlimits

import "fmt"

type Limits struct {
	RuntimeSeconds int    `json:"runtime_seconds"`
	CPUSeconds     int64  `json:"cpu_seconds"`
	MemoryBytes    uint64 `json:"memory_bytes"`
	DiskBytes      uint64 `json:"disk_bytes"`
	LogBytes       uint64 `json:"log_bytes"`
}

func (l Limits) Validate() error {
	if l.RuntimeSeconds < 1 || l.RuntimeSeconds > 7*24*3600 || l.CPUSeconds < 1 || l.CPUSeconds > 64*7*24*3600 || l.MemoryBytes < 1<<20 || l.MemoryBytes > 1<<40 || l.DiskBytes < 1<<20 || l.DiskBytes > 1<<40 || l.LogBytes < 1 || l.LogBytes > 1<<30 || l.LogBytes > l.DiskBytes {
		return fmt.Errorf("bounded execution resource limits required")
	}
	return nil
}
func (l Limits) Within(maximum Limits) bool {
	return l.RuntimeSeconds <= maximum.RuntimeSeconds && l.CPUSeconds <= maximum.CPUSeconds && l.MemoryBytes <= maximum.MemoryBytes && l.DiskBytes <= maximum.DiskBytes && l.LogBytes <= maximum.LogBytes
}
