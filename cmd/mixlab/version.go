package main

import (
	"runtime/debug"

	"github.com/mrothroc/mixlab/internal/buildinfo"
)

func versionString() string {
	return buildinfo.Report("mixlab")
}

// Keep the legacy version line unchanged for existing consumers.
func formatVersion(info *debug.BuildInfo) string {
	return buildinfo.FormatVersion("mixlab", info)
}
