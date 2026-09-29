//go:build darwin || linux

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/recruitment"
	"github.com/mrothroc/mixlab/trust/bootstrap"
)

func TestSubmitValidationAndHelp(t *testing.T) {
	for _, args := range [][]string{
		nil, {"-pool", "remote"}, {"-workers", "1"},
		{"-abort", "-attempt-state-dir", "/tmp/unused", "-config", "changed.json"},
		{"-abort", "-attempt-state-dir", "/tmp/unused", "-node", "node.local:7445"},
		{"-node", "https://node:443/path"}, {"-discover", "auto"},
		{"-fetch", "-abort", "-attempt-state-dir", "/tmp/unused"},
		{"-fetch", "-config", "changed.json", "-attempt-state-dir", "/tmp/unused"},
		{"-fetch", "-resume-from", "/tmp/old", "-attempt-state-dir", "/tmp/unused"},
		{"-abort", "-checkpoint-at", "2", "-attempt-state-dir", "/tmp/unused"},
		{"-checkpoint-at", "2147483648", "-attempt-state-dir", "/tmp/unused"},
	} {
		var out, diagnostic bytes.Buffer
		if code := runSubmitContext(context.Background(), args, &out, &diagnostic); code != 2 {
			t.Fatal(args, code, diagnostic.String())
		}
	}
	var out, diagnostic bytes.Buffer
	if code := run([]string{"submit", "-help"}, &out, &diagnostic); code != 0 {
		t.Fatal(code, diagnostic.String())
	}
	for _, flag := range []string{"-workers", "-config", "-worker-binary", "-train", "-dataset-id", "-abort", "-fetch", "-checkpoint-at", "-resume-from", "-attempt-state-dir", "-principal-state-dir", "-cluster-state-dir"} {
		if !strings.Contains(diagnostic.String(), flag) {
			t.Fatal("missing help", flag)
		}
	}
}

func TestResumeSelectionKeepsOrderedNodes(t *testing.T) {
	source := nodejob.Manifest{Members: []nodejob.Member{{Node: "a", Rank: 0}, {Node: "b", Rank: 1}}}
	for _, tc := range []struct {
		nodes []string
		ok    bool
	}{{[]string{"a", "b"}, true}, {[]string{"b", "a"}, false}, {[]string{"a", "c"}, false}, {[]string{"a"}, false}} {
		var s recruitment.Selection
		for _, node := range tc.nodes {
			s.Selected = append(s.Selected, recruitment.Candidate{Capabilities: nodeagent.Capabilities{Node: node}})
		}
		if err := validateResumeSelection(source, s); (err == nil) != tc.ok {
			t.Fatal(tc, err)
		}
	}
	s := recruitment.Selection{Requirements: recruitment.Requirements{DatasetID: "changed"}}
	if err := validateResumeSelection(source, s); err == nil || !strings.Contains(err.Error(), "dataset") {
		t.Fatal("dataset mismatch not rejected before reservation", err)
	}
}

func TestNodesTransientAuthorityLifecycle(t *testing.T) {
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(dir, 0700); err != nil {
		t.Fatal(err)
	}
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	address := l.Addr().String()
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}
	var out, diagnostic bytes.Buffer
	args := []string{"init", "-state-home", filepath.Join(dir, "state"), "-key-backend", "file", "-trust-listen", address, "-controller-principal-state-dir", filepath.Join(dir, "controller")}
	if code := run(args, &out, &diagnostic); code != 0 {
		t.Fatal(code, diagnostic.String())
	}
	var report bootstrap.Report
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	diagnostic.Reset()
	args = []string{"nodes", "-principal-state-dir", filepath.Join(dir, "controller"), "-cluster-state-dir", report.AuthorityDir, "-node", "127.0.0.1:1"}
	if code := run(args, &out, &diagnostic); code != 0 {
		t.Fatal(code, diagnostic.String())
	}
	if !strings.Contains(out.String(), "connection_refused") {
		t.Fatal(out.String())
	}
	l, err = net.Listen("tcp", address)
	if err != nil {
		t.Fatal("temporary authority listener leaked", err)
	}
	if err := l.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestSubmissionPortsAvoidEphemeralRange(t *testing.T) {
	ports, err := submissionLoopbackPorts(8)
	if err != nil {
		t.Fatal(err)
	}
	seen := map[int]bool{}
	for _, port := range ports {
		if port < 20000 || port >= 30000 || seen[port] {
			t.Fatal("invalid or duplicate port", ports)
		}
		seen[port] = true
	}
	if _, err := submissionLoopbackPorts(0); err == nil {
		t.Fatal("empty world accepted")
	}
}
