package workerhost

import (
	"fmt"
	"strings"

	"golang.org/x/sys/unix"
)

func hostBootIdentity() (string, error) {
	id, err := unix.Sysctl("kern.bootsessionuuid")
	id = strings.ToLower(strings.TrimSpace(id))
	if err != nil || !validBootIdentity(id) {
		return "", fmt.Errorf("cannot identify host boot: %v", err)
	}
	return id, nil
}
