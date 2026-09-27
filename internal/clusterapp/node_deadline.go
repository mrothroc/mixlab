package clusterapp

import (
	"net/http"
	"strings"
	"time"
)

// Checkpoint assembly/verification can read up to 1 GiB. Keep body reads short,
// but allow bounded local disk work without weakening unrelated route limits.
func nodeOperationTimeout(method, path string) time.Duration {
	parts := strings.Split(strings.TrimPrefix(path, "/v1/agent/"), "/")
	if strings.HasPrefix(path, "/v1/agent/") && len(parts) == 3 && parts[0] == "jobs" && nodeRouteID(parts[1]) {
		if method == http.MethodPut && (parts[2] == "input" || parts[2] == "prepare") || method == http.MethodPost && parts[2] == "start" {
			return 50 * time.Second
		}
	}
	return 10 * time.Second
}
