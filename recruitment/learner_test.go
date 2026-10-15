package recruitment

import (
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/nodejob"
)

func TestLearnerSelectionDoesNotRequireDDPRing(t *testing.T) {
	r, o := fixture()
	o.Capabilities.Probe.Backends = []string{}
	o.Capabilities.Recruitable = false
	o.Capabilities.ManifestVersions = []string{nodejob.LearnerVersion}
	if reason := rejectFor(o, r, selectionNow, true); reason != "" {
		t.Fatal(reason)
	}
	if reason := reject(o, r, selectionNow); reason == "" {
		t.Fatal("DDP accepted non-ring node")
	}
	o.Capabilities.ManifestVersions = []string{nodejob.Version}
	if reason := rejectFor(o, r, selectionNow, true); reason != "incompatible_worker" {
		t.Fatal(reason)
	}
	o.Capabilities.ManifestVersions = []string{nodejob.LearnerVersion}
	o.Peer.Principal = strings.Repeat("f", 32)
	if reason := rejectFor(o, r, selectionNow, true); reason != "wrong_identity" {
		t.Fatal(reason)
	}
}
