package learnerapi

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/diloco/authenticated"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
)

func TestGrantsPersistExactEnvelopeAcrossRestart(t *testing.T) {
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err = os.Chmod(dir, 0700); err != nil {
		t.Fatal(err)
	}
	p, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Principal, Role: "coordinator"})
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile("../../diloco/testdata/grant.json")
	if err != nil {
		t.Fatal(err)
	}
	var body run.Grant
	if err = json.Unmarshal(b, &body); err != nil {
		t.Fatal(err)
	}
	store, err := OpenGrants(p)
	if err != nil {
		t.Fatal(err)
	}
	// Structural envelope only here. Integration tests verify real signatures,
	// TLS possession and fresh trust before this journal can authorize any I/O.
	envelope := authenticated.Grant{Body: body, Proof: trust.SignedProof{Request: trust.SignRequest{Version: trust.ProofVersion, Purpose: trust.ArtifactGrant, Digest: body.SHA256, Context: body.Binding.RunID, Audience: body.Coordinator.Audience}, Evidence: trust.TrustEvidence{Cluster: body.Coordinator.ClusterID, Principal: body.Coordinator.PrincipalID, Role: trust.Coordinator}, Signature: make([]byte, 64)}}
	first, err := store.issue(context.Background(), body, func() (authenticated.Grant, error) { return envelope, nil })
	if err != nil {
		t.Fatal(err)
	}
	store, err = OpenGrants(p)
	if err != nil {
		t.Fatal(err)
	}
	again, err := store.issue(context.Background(), body, func() (authenticated.Grant, error) {
		t.Fatal("restart re-signed an accepted grant")
		return authenticated.Grant{}, nil
	})
	if err != nil || !reflect.DeepEqual(first, again) {
		t.Fatal("grant envelope changed", err)
	}
	changed := body
	changed.Artifact.Ref.Bytes++
	changed.SHA256, err = changed.Digest()
	if err != nil {
		t.Fatal(err)
	}
	if _, err = store.issue(context.Background(), changed, func() (authenticated.Grant, error) {
		t.Fatal("rebound grant was signed")
		return authenticated.Grant{}, nil
	}); err == nil {
		t.Fatal("grant ID rebound")
	}
}
