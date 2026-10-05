// Package buildinfo reports executable identity using the shared worker
// contract, without importing training or trust application components.
package buildinfo

import (
	"runtime/debug"
	"strings"

	"github.com/mrothroc/mixlab/workercontrol"
)

// WorkerProtocol identifies the local worker-control envelope contract, not
// enrollment support or compatibility of its context-owned payloads.
const WorkerProtocol = workercontrol.Version

const shortRevisionLen = 12

// Version and Revision are set by container builds without module/VCS metadata.
// Empty values retain Go's normal module and VCS reporting.
var Version string
var Revision string

// Report reads linker metadata and includes the worker protocol identity.
func Report(binary string) string {
	info, _ := debug.ReadBuildInfo()
	return FormatReport(binary, info)
}

// FormatReport preserves the historical version line as its first line.
func FormatReport(binary string, info *debug.BuildInfo) string {
	return FormatVersion(binary, info) + "\nworker_protocol " + WorkerProtocol
}

// FormatVersion retains the original mixlab version formatting. VCS time is
// the source commit time stamped by Go, not the time the binary was compiled.
func FormatVersion(binary string, info *debug.BuildInfo) string {
	version := "unknown"
	var revision, buildTime string
	modified := false
	if info != nil {
		if info.Main.Version != "" {
			version = info.Main.Version
		}
		for _, setting := range info.Settings {
			switch setting.Key {
			case "vcs.revision":
				revision = setting.Value
			case "vcs.time":
				buildTime = setting.Value
			case "vcs.modified":
				modified = setting.Value == "true"
			}
		}
	}
	if Version != "" {
		version = Version
	}
	if Revision != "" {
		revision = Revision
	}
	out := binary + " " + version
	if revision == "" && buildTime == "" {
		return out
	}
	var detail []string
	if revision != "" {
		if len(revision) > shortRevisionLen {
			revision = revision[:shortRevisionLen]
		}
		if modified && !strings.HasSuffix(version, "+dirty") {
			revision += "-dirty"
		}
		detail = append(detail, revision)
	}
	if buildTime != "" {
		detail = append(detail, buildTime)
	}
	return out + " (" + strings.Join(detail, ", ") + ")"
}
