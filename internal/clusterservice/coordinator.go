package clusterservice

import (
	"fmt"
	"net"
	"path/filepath"
	"strconv"
	"strings"
)

func validRole(role string) bool {
	return role == "agent" || role == "authority" || role == "coordinator"
}

// CoordinatorPathFlags are independent context roots, not subdirectories of an
// authority/key store. Admission and coordination deliberately use separate
// enrolled controller and coordinator identities.
func CoordinatorPathFlags() []string {
	return []string{"principal-state-dir", "admission-principal-state-dir", "coordinator-state-dir", "admission-state-dir", "checkpoint-dir", "artifact-dir"}
}

func validateCoordinatorArguments(args []string) error {
	if len(args) == 0 || args[0] != "serve" {
		return fmt.Errorf("coordinator service must serve")
	}
	args = args[1:]
	paths := map[string]bool{}
	for _, name := range CoordinatorPathFlags() {
		paths["-"+name] = true
	}
	seen := map[string]string{}
	for len(args) > 0 {
		if len(args) < 2 || seen[args[0]] != "" || (!paths[args[0]] && args[0] != "-coordinator-listen") || args[1] == "" {
			return fmt.Errorf("invalid, missing or duplicate coordinator service option")
		}
		name, value := args[0], args[1]
		if paths[name] && (!filepath.IsAbs(value) || filepath.Clean(value) != value) {
			return fmt.Errorf("%s requires a canonical absolute context path", name)
		}
		seen[name] = value
		args = args[2:]
	}
	for name := range paths {
		if seen[name] == "" {
			return fmt.Errorf("explicit %s required", name)
		}
		for other := range paths {
			if other != name && (seen[name] == seen[other] || strings.HasPrefix(seen[name], seen[other]+string(filepath.Separator))) {
				return fmt.Errorf("coordinator context paths must not overlap")
			}
		}
	}
	host, port, err := net.SplitHostPort(seen["-coordinator-listen"])
	n, parseErr := strconv.Atoi(port)
	if err != nil || parseErr != nil || host == "" || n < 1 || n > 65535 {
		return fmt.Errorf("explicit coordinator-listen host:port required")
	}
	return nil
}
