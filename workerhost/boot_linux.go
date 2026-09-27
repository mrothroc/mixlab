package workerhost

import (
	"fmt"
	"os"
	"strings"
)

func hostBootIdentity() (string, error) {
	b, err := os.ReadFile("/proc/sys/kernel/random/boot_id")
	id := strings.TrimSpace(string(b))
	if err != nil || !validBootIdentity(id) {
		return "", fmt.Errorf("cannot identify host boot: %v", err)
	}
	return id, nil
}
