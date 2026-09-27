//go:build darwin || linux

package main

import (
	"fmt"
	"io"
	"os"

	"github.com/mrothroc/mixlab/workerhost"
)

func runInternalWorkerHost(args []string, stderr io.Writer) int {
	if len(args) != 0 {
		_, _ = fmt.Fprintln(stderr, "internal worker host accepts only its inherited control descriptor")
		return 2
	}
	if err := workerhost.ServeGuardian(os.NewFile(3, "guardian-control")); err != nil {
		_, _ = fmt.Fprintln(stderr, "internal worker host:", err)
		return 1
	}
	return 0
}
