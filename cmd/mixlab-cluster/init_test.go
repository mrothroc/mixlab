package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/mrothroc/mixlab/trust/bootstrap"
)

func TestInitAndRecoverCLI(t *testing.T) {
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(dir, 0700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("MIXLAB_STATE_HOME", filepath.Join(dir, "ignored"))
	var out, errs bytes.Buffer
	args := []string{"init", "-state-home", filepath.Join(dir, "state"), "-key-backend", "file", "-trust-listen", "localhost:7443", "-controller-principal-state-dir", filepath.Join(dir, "controller")}
	if code := run(args, &out, &errs); code != 0 || errs.Len() > 0 {
		t.Fatalf("code=%d err=%s", code, &errs)
	}
	var report bootstrap.Report
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if len(report.Principals) != 3 || report.Principals[1].Directory != filepath.Join(dir, "controller") {
		t.Fatal(report)
	}
	if !strings.HasPrefix(report.AuthorityDir, filepath.Join(dir, "state", "clusters")+"/") {
		t.Fatal(report.AuthorityDir)
	}
	if _, err := os.Lstat(filepath.Join(dir, "ignored")); !os.IsNotExist(err) {
		t.Fatal("ignored env was used", err)
	}
	out.Reset()
	errs.Reset()
	if code := run([]string{"init", "-recover", "-cluster-state-dir", report.AuthorityDir}, &out, &errs); code != 0 || errs.Len() > 0 {
		t.Fatalf("code=%d err=%s", code, &errs)
	}
	var recovered bootstrap.Report
	if err := json.Unmarshal(out.Bytes(), &recovered); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(report, recovered) {
		t.Fatal("recovered different identity")
	}
	out.Reset()
	errs.Reset()
	if code := run([]string{"init", "-cluster-state-dir", report.AuthorityDir, "-state-home", filepath.Join(dir, "state"), "-key-backend", "file"}, &out, &errs); code != 1 || out.Len() != 0 || !strings.Contains(errs.String(), "-recover") {
		t.Fatalf("code=%d out=%s err=%s", code, &out, &errs)
	}
}

func TestInitCLIValidationAndHelp(t *testing.T) {
	for _, args := range [][]string{
		{"-recover"}, {"-recover", "-cluster-state-dir", "/tmp/unused", "-key-backend", "file"},
		{"-trust-advertise", "mdns"}, {"-trust-listen", "0.0.0.0:7443"},
		{"-trust-listen", ":7443"}, {"-trust-listen", "localhost:0"}, {"-trust-listen", "localhost:65536"},
		{"-trust-listen", "https://localhost:7443"}, {"positional"},
	} {
		var out, errs bytes.Buffer
		if code := run(append([]string{"init"}, args...), &out, &errs); code != 2 || out.Len() > 0 || errs.Len() == 0 {
			t.Fatalf("args=%v code=%d out=%s err=%s", args, code, &out, &errs)
		}
	}
	var out, errs bytes.Buffer
	if code := run([]string{"init", "-help"}, &out, &errs); code != 0 {
		t.Fatalf("%d: %s", code, &errs)
	}
	for _, flag := range []string{"-state-home", "-cluster-state-dir", "-controller-principal-state-dir", "-coordinator-principal-state-dir", "-trust-listen", "-trust-advertise", "-key-backend", "-recover"} {
		if !strings.Contains(errs.String(), flag) {
			t.Fatal("missing help", flag)
		}
	}
	if endpoint, err := bootstrapEndpoint("[::1]:7443"); err != nil || endpoint != "https://[::1]:7443" {
		t.Fatal(endpoint, err)
	}
}
