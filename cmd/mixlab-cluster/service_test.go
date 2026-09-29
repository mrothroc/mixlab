package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestServiceAttemptHonorsParentStop(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	var log bytes.Buffer
	code := runServiceRetry(ctx, &log, func() int {
		t.Fatal("started after stop")
		return 1
	})
	if code != 0 || log.Len() != 0 {
		t.Fatalf("code=%d log=%s", code, log.String())
	}
}

func TestSignedServiceRequirement(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("macOS signing requirement")
	}
	binary := os.Getenv("MIXLAB_SIGNED_SERVICE_TEST_BINARY")
	if binary == "" {
		t.Skip("set MIXLAB_SIGNED_SERVICE_TEST_BINARY to a signed candidate")
	}
	if err := verifyServiceSignature(context.Background(), binary); err != nil {
		t.Fatal(err)
	}
	if err := verifyServiceSignature(context.Background(), "/usr/bin/true"); err == nil {
		t.Fatal("unrelated Apple binary accepted as Mixlab")
	}
}

func TestServiceCLIHelpAndValidation(t *testing.T) {
	for _, args := range [][]string{{"agent", "install", "-help"}, {"authority", "install", "-help"}, {"agent", "reapprove", "-help"}, {"doctor", "-help"}} {
		var out, err bytes.Buffer
		if code := run(args, &out, &err); code != 0 {
			t.Fatal(args, code, err.String())
		}
	}
	for _, args := range [][]string{{"agent", "install"}, {"agent", "stop", "-agent-listen", "127.0.0.1:7445"}, {"agent", "reapprove"}, {"doctor", "extra"}} {
		var out, err bytes.Buffer
		if code := run(args, &out, &err); code != 2 {
			t.Fatal(args, code, err.String())
		}
	}
}

func TestDoctorWithoutStateIsHonestAndReadOnly(t *testing.T) {
	home, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("HOME", home)
	var out, stderr bytes.Buffer
	if code := run([]string{"doctor"}, &out, &stderr); code != 0 {
		t.Fatal(code, stderr.String())
	}
	var response struct {
		Checks []doctorCheck `json:"checks"`
	}
	if err := json.Unmarshal(out.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, c := range response.Checks {
		if c.Check == "trust" {
			found = true
			if c.Status != "unknown" {
				t.Fatal(c)
			}
		}
	}
	if !found {
		t.Fatal(out.String())
	}
	entries, err := os.ReadDir(home)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatal("doctor created state", entries)
	}
	if strings.Contains(out.String(), "private-key") {
		t.Fatal(out.String())
	}
}
