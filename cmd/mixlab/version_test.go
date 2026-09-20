package main

import (
	"runtime/debug"
	"strings"
	"testing"
)

// buildInfoFixture builds the debug.BuildInfo shapes the linker actually
// produces, so the formatter is exercised against real inputs rather than a
// restatement of its own logic.
func buildInfoFixture(version string, settings map[string]string) *debug.BuildInfo {
	info := &debug.BuildInfo{Main: debug.Module{Version: version}}
	for key, value := range settings {
		info.Settings = append(info.Settings, debug.BuildSetting{Key: key, Value: value})
	}
	return info
}

func TestFormatVersion(t *testing.T) {
	const rev = "e00e54f2efe2416c44d35e6d94be6c796827e21e"
	for _, tc := range []struct {
		name string
		info *debug.BuildInfo
		want string
	}{
		{
			name: "tagged release",
			info: buildInfoFixture("v0.115.1", map[string]string{
				"vcs.revision": rev, "vcs.time": "2026-09-18T13:28:21Z", "vcs.modified": "false",
			}),
			want: "mixlab v0.115.1 (e00e54f2efe2, 2026-09-18T13:28:21Z)",
		},
		{
			name: "dirty tree is marked",
			info: buildInfoFixture("v0.115.1", map[string]string{
				"vcs.revision": rev, "vcs.time": "2026-09-18T13:28:21Z", "vcs.modified": "true",
			}),
			want: "mixlab v0.115.1 (e00e54f2efe2-dirty, 2026-09-18T13:28:21Z)",
		},
		{
			name: "untagged build reports devel",
			info: buildInfoFixture("(devel)", map[string]string{
				"vcs.revision": rev, "vcs.time": "2026-09-18T13:28:21Z", "vcs.modified": "false",
			}),
			want: "mixlab (devel) (e00e54f2efe2, 2026-09-18T13:28:21Z)",
		},
		{
			name: "no vcs stamping still reports the module version",
			info: buildInfoFixture("v0.115.1", nil),
			want: "mixlab v0.115.1",
		},
		{
			name: "short revision is not truncated past its length",
			info: buildInfoFixture("v1.0.0", map[string]string{"vcs.revision": "abc123"}),
			want: "mixlab v1.0.0 (abc123)",
		},
		{
			name: "missing build info degrades rather than crashing",
			info: nil,
			want: "mixlab unknown",
		},
		{
			name: "empty module version degrades",
			info: buildInfoFixture("", map[string]string{"vcs.revision": rev}),
			want: "mixlab unknown (e00e54f2efe2)",
		},
		{
			// Go stamps its own "+dirty" onto a pseudo-version; saying it twice
			// adds nothing.
			name: "dirty is not reported twice",
			info: buildInfoFixture("v0.115.2-0.20260920202034-c56d1b39c7e4+dirty", map[string]string{
				"vcs.revision": "c56d1b39c7e4dddddddddddddddddddddddddddd", "vcs.modified": "true",
			}),
			want: "mixlab v0.115.2-0.20260920202034-c56d1b39c7e4+dirty (c56d1b39c7e4)",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := formatVersion(tc.info); got != tc.want {
				t.Fatalf("formatVersion() = %q, want %q", got, tc.want)
			}
		})
	}
}

// The flag must report what the linker stamped into this binary, not a
// constant someone has to remember to bump at release time.
func TestVersionStringReflectsRealBuildInfo(t *testing.T) {
	got := versionString()
	if !strings.HasPrefix(got, "mixlab ") {
		t.Fatalf("versionString() = %q, want a \"mixlab \" prefix", got)
	}
	info, ok := debug.ReadBuildInfo()
	if !ok {
		t.Skip("build info unavailable")
	}
	if want := formatVersion(info); got != want {
		t.Fatalf("versionString() = %q, want %q from this binary's build info", got, want)
	}
	for _, hardcoded := range []string{"0.115.1", "0.115.0", "0.114.0"} {
		if strings.Contains(got, hardcoded) {
			t.Fatalf("versionString() = %q looks hardcoded; it must come from build info", got)
		}
	}
}
