package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust/bootstrap"
)

func TestRevokeDeliveryFailurePreservesCommittedRevocation(t *testing.T) {
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(dir, 0700); err != nil {
		t.Fatal(err)
	}
	var out, diagnostic bytes.Buffer
	if code := runInit([]string{"-state-home", filepath.Join(dir, "state"), "-key-backend", "file", "-trust-listen", availableAddress(t)}, &out, &diagnostic); code != 0 {
		t.Fatal(code, diagnostic.String())
	}
	var initialized bootstrap.Report
	if err := json.Unmarshal(out.Bytes(), &initialized); err != nil {
		t.Fatal(err)
	}
	id := strings.Repeat("e", 32)
	out.Reset()
	diagnostic.Reset()
	args := []string{"-cluster-state-dir", initialized.AuthorityDir, "-principal-id", id, "-revocation-reason", "test", "-discover", "off", "-daemon-address", "127.0.0.1:1"}
	if code := runRevoke(args, &out, &diagnostic); code != 1 || !strings.Contains(diagnostic.String(), "revocation committed") {
		t.Fatal(code, diagnostic.String())
	}
	var receipt struct {
		Generation uint64               `json:"generation"`
		Pushed     bool                 `json:"pushed"`
		Delivery   []revocationDelivery `json:"delivery"`
	}
	if err := json.Unmarshal(out.Bytes(), &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Generation == 0 || receipt.Pushed || len(receipt.Delivery) != 1 || receipt.Delivery[0].Delivered {
		t.Fatal("incorrect delivery receipt", out.String())
	}
	p, err := statehome.Discover(statehome.Options{ExactDir: initialized.AuthorityDir}, statehome.Context{Kind: statehome.Authority})
	if err != nil {
		t.Fatal(err)
	}
	a, err := clusterapp.OpenAuthority(context.Background(), p, time.Now())
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = a.Close() }()
	s, err := a.Snapshots.Load(time.Now())
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range s.Payload.Revocations {
		if r.Kind == "principal" && r.ID == id && r.FirstGeneration == receipt.Generation {
			return
		}
	}
	t.Fatal("delivery failure lost the committed revocation")
}
