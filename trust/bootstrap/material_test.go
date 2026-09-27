package bootstrap

import (
	"testing"

	"github.com/mrothroc/mixlab/trust/principal"
)

func TestBootstrapRuntimeMaterial(t *testing.T) {
	c := fixture(t)
	if _, err := LoadAuthority(c.Authority, testNow); err == nil {
		t.Fatal("missing bootstrap accepted")
	}
	_, err := Initialize(testContext, c, testNow)
	require(t, err)
	m, err := LoadAuthority(c.Authority, testNow)
	require(t, err)
	if m.Anchor.Cluster() != c.Cluster || m.AuthorityPrincipal.Dir() != c.Principals[0].Final.Dir() || m.IssuerKey.ID == m.SnapshotKey.ID {
		t.Fatal("runtime handoff mismatch")
	}
	for _, p := range c.Principals {
		s, err := principal.Open(p.Final, testNow)
		require(t, err)
		r, k, err := s.Active(testNow)
		require(t, err)
		if r.Principal != p.Principal || r.Role != p.Role || k == nil {
			t.Fatal("bootstrap principal cannot operate")
		}
		require(t, s.Close())
	}
}
