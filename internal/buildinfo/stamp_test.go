package buildinfo

import (
	"runtime/debug"
	"testing"
)

func TestExplicitBuildStamp(t *testing.T) {
	oldV, oldR := Version, Revision
	t.Cleanup(func() { Version, Revision = oldV, oldR })
	Version, Revision = "v1.2.3", "0123456789abcdef"
	for _, info := range []*debug.BuildInfo{nil, {Main: debug.Module{Version: "(devel)"}}} {
		if got := FormatVersion("mixlab", info); got != "mixlab v1.2.3 (0123456789ab)" {
			t.Fatal(got)
		}
	}
}
