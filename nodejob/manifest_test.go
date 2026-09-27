package nodejob

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/distributed"
)

func example(t *testing.T) Manifest {
	t.Helper()
	id := func(s string) string { return strings.Repeat(s, 32) }
	membership, e := distributed.NewDDPGroupMembership(id("1"), id("2"), 1, "ring", []distributed.DDPGroupMember{{MemberID: id("3"), Rank: 0}, {MemberID: id("4"), Rank: 1}})
	if e != nil {
		t.Fatal(e)
	}
	config := json.RawMessage(`{"model_dim":16}`)
	hash := strings.Repeat("a", 64)
	return Manifest{Version: Version, Job: id("5"), Attempt: id("6"), Lease: id("7"), Node: id("8"), Controller: id("9"), Nonce: id("a"), Membership: membership, Members: []Member{{id("8"), id("3"), 0}, {id("b"), id("4"), 1}}, Rank: 0, BuildID: hash, Config: config, ConfigHash: Hash(config), ProgramHash: hash, WeightLayoutHash: hash, OptimizerHash: hash, DatasetSelector: "toy.train", DatasetID: hash, Artifacts: []ArtifactRef{}, Mode: "arch", Transport: "tls13-ring", Limits: Limits{RuntimeSeconds: 60, CPUSeconds: 120, MemoryBytes: 1 << 30, DiskBytes: 1 << 30, LogBytes: 1 << 20}, Created: 1000, Expires: 1060}
}
func TestManifestStrictBindingAndBudgets(t *testing.T) {
	m := example(t)
	if e := m.Validate(); e != nil {
		t.Fatal(e)
	}
	r, e := m.SigningRequest()
	if e != nil || r.Audience != m.Node || r.Context != m.Job+"/"+m.Attempt {
		t.Fatal(r, e)
	}
	for name, mutate := range map[string]func(*Manifest){
		"raw transport":         func(m *Manifest) { m.Transport = "raw" },
		"shell mode":            func(m *Manifest) { m.Mode = "shell" },
		"dataset path":          func(m *Manifest) { m.DatasetSelector = "/tmp/train.bin" },
		"wrong node":            func(m *Manifest) { m.Node = m.Members[1].Node },
		"duplicate node":        func(m *Manifest) { m.Members[1].Node = m.Node },
		"wrong rank":            func(m *Manifest) { m.Rank = 3 },
		"changed member":        func(m *Manifest) { m.Members[0].MemberID = m.Members[1].MemberID },
		"stale membership hash": func(m *Manifest) { m.Membership.MembersHash = strings.Repeat("f", 64) },
		"config hash":           func(m *Manifest) { m.ConfigHash = strings.Repeat("f", 64) },
		"duplicate config key": func(m *Manifest) {
			m.Config = json.RawMessage(`{"model_dim":16,"model_dim":32}`)
			m.ConfigHash = Hash(m.Config)
		},
		"long expiry":        func(m *Manifest) { m.Expires = m.Created + 3601 },
		"unlimited memory":   func(m *Manifest) { m.Limits.MemoryBytes = 0 },
		"oversized artifact": func(m *Manifest) { m.Artifacts = []ArtifactRef{{strings.Repeat("f", 64), 2 << 30, "weights"}} },
	} {
		t.Run(name, func(t *testing.T) {
			copy := example(t)
			mutate(&copy)
			if e := copy.Validate(); e == nil {
				t.Fatal("invalid manifest accepted")
			}
		})
	}
	if _, _, e := (Accepted{}).Value(); e == nil {
		t.Fatal("zero acceptance value grants job")
	}
}
