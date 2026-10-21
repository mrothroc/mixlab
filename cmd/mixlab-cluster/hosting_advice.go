package main

import (
	"fmt"
	"io"
)

const macOSLocalNetworkAdvice = "No supported BSD-socket permission query. Signed LaunchAgents require a logged-in GUI session and Local Network approval. For manual headless SSH operation, an administrator can configure Apple's system-wide Local Network allowlist and restart the Mac; use explicitly provisioned file keys. See docs/cluster-headless.md. EHOSTUNREACH can also mean a routing problem; no settings changed."

func reportServiceInstalled(w io.Writer, role, platform string) {
	_, _ = fmt.Fprintf(w, "%s service installed. Use '%s status' and 'doctor' to check readiness.\n", role, role)
	if platform == "darwin" {
		_, _ = fmt.Fprintln(w, "This installed LaunchAgent requires a logged-in GUI session. Approve Local Network at the console, or use an administrator-configured Local Network allowlist. The allowlist does not make a LaunchAgent run while logged out; manual headless SSH operation uses foreground commands and file keys instead. See docs/cluster-headless.md. Logs: ~/.mixlab/services/"+role+"/service.log")
	} else {
		_, _ = fmt.Fprintln(w, "Logged-out operation requires administrator-enabled systemd lingering; installation does not change it.")
	}
}
