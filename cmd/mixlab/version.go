package main

import (
	"runtime/debug"
	"strings"
)

// shortRevisionLen matches the abbreviation git uses in most porcelain output,
// which is what a reader will compare the value against.
const shortRevisionLen = 12

// versionString reports this binary's version from the information the Go
// linker stamps in. Nothing here is hand-maintained: a release bumps the tag
// and the module version follows, so the flag cannot drift from reality the
// way a hardcoded constant does.
func versionString() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return formatVersion(nil)
	}
	return formatVersion(info)
}

func formatVersion(info *debug.BuildInfo) string {
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

	out := "mixlab " + version
	if revision == "" && buildTime == "" {
		return out
	}

	var detail []string
	if revision != "" {
		if len(revision) > shortRevisionLen {
			revision = revision[:shortRevisionLen]
		}
		// A dirty tree means the binary does not correspond to the commit, so
		// say so rather than letting the revision imply reproducibility. Go
		// already suffixes a pseudo-version with "+dirty"; don't repeat it.
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
