package clusterapp

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/nodecredentials"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
	"github.com/mrothroc/mixlab/trust/bootstrap"
	"github.com/mrothroc/mixlab/trust/keylifecycle"
	"github.com/mrothroc/mixlab/trust/workload"
)

func TestWorkloadTransportLostStateNeverRegenerates(t *testing.T) {
	p := initialized(t)
	a, err := bootstrap.LoadAuthority(p, testNow)
	check(t, err)
	id := strings.Repeat("a", 32)
	scope := workload.Scope{Cluster: a.Anchor.Cluster(), Controller: id, Node: id, Lease: id, Run: id, Job: id, Attempt: id, Group: id, Member: id, Workload: id, Generation: 1, MembershipHash: strings.Repeat("a", 64), ManifestHash: strings.Repeat("b", 64), Created: testNow.Unix(), AdmitUntil: testNow.Add(time.Minute).Unix(), Deadline: testNow.Add(time.Hour).Unix()}
	for _, damage := range []string{"key", "intent", "credential", "context"} {
		t.Run(damage, func(t *testing.T) {
			path, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(filepath.Dir(p.Dir()), "lost-"+damage)}, statehome.Context{Kind: statehome.Agent})
			check(t, err)
			c, err := nodecredentials.InitializeTransport(context.Background(), path, a.Anchor, scope, "file")
			check(t, err)
			stop := errors.New("signing interrupted after key creation")
			if _, err := c.PrepareRequest(context.Background(), func(context.Context, trust.SignRequest) (trust.SignedProof, error) {
				return trust.SignedProof{}, stop
			}, trust.VerifiedSnapshot{}, testNow); !errors.Is(err, stop) {
				t.Fatal(err)
			}
			check(t, c.Close())
			raw, err := path.ReadFile("key-intent-workload.json")
			check(t, err)
			var intent keylifecycle.Record
			check(t, json.Unmarshal(raw, &intent))
			if intent.Stage != "active" || intent.Active == nil {
				t.Fatal("expected durable protected key")
			}
			file := map[string]string{"key": "key-" + intent.Active.ID + ".json", "intent": "key-intent-workload.json", "credential": "transport-credential.json", "context": "key-context.json"}[damage]
			check(t, os.Remove(filepath.Join(path.Dir(), file)))
			c, err = nodecredentials.OpenTransport(path, a.Anchor, scope)
			if err == nil {
				defer func() { _ = c.Close() }()
				if _, err := c.PrepareRequest(context.Background(), func(context.Context, trust.SignRequest) (trust.SignedProof, error) {
					t.Fatal("lost key/history reached signing callback")
					return trust.SignedProof{}, nil
				}, trust.VerifiedSnapshot{}, testNow); err == nil {
					t.Fatal("lost state recovered implicitly")
				}
				if damage == "intent" {
					for range 2 {
						if err := c.Destroy(context.Background()); err == nil {
							t.Fatal("missing claimed key history was declared destroyed")
						}
					}
					raw, err := path.ReadFile("transport-credential.json")
					check(t, err)
					var state struct {
						Stage string `json:"stage"`
					}
					check(t, json.Unmarshal(raw, &state))
					if state.Stage != "destroying" {
						t.Fatal("uncertain destruction lost its fence", state.Stage)
					}
				}
			}
			if _, err := nodecredentials.InitializeTransport(context.Background(), path, a.Anchor, scope, "file"); err == nil {
				t.Fatal("published job identity reinitialized")
			}
			if _, err := path.ReadFile(file); !errors.Is(err, os.ErrNotExist) {
				t.Fatal("missing state recreated", err)
			}
		})
	}
}
