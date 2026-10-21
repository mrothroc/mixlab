package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"syscall"
	"testing"

	"github.com/mrothroc/mixlab/internal/clusterdiagnostic"
)

func TestServiceInstalledHostingAdvice(t *testing.T) {
	for _, role := range []string{"authority", "agent", "coordinator"} {
		for _, platform := range []string{"darwin", "linux"} {
			t.Run(role+"/"+platform, func(t *testing.T) {
				var out bytes.Buffer
				reportServiceInstalled(&out, role, platform)
				want := []string{role + " service installed", role + " status", "doctor"}
				if platform == "darwin" {
					want = append(want, "logged-in GUI session", "Local Network allowlist", "does not make a LaunchAgent run while logged out", "foreground commands and file keys", "docs/cluster-headless.md", "/"+role+"/service.log")
				} else {
					want = append(want, "systemd lingering", "installation does not change it")
					if strings.Contains(out.String(), "LaunchAgent") {
						t.Fatal("Linux received macOS-only advice")
					}
				}
				for _, text := range want {
					if !strings.Contains(out.String(), text) {
						t.Errorf("missing %q in %s", text, &out)
					}
				}
			})
		}
	}
}

func TestLocalNetworkAdvicePreservesUncertainty(t *testing.T) {
	for _, text := range []string{"No supported BSD-socket permission query", "logged-in GUI session", "system-wide Local Network allowlist", "restart", "file keys", "docs/cluster-headless.md", "routing problem", "no settings changed"} {
		if !strings.Contains(macOSLocalNetworkAdvice, text) {
			t.Errorf("missing %q", text)
		}
	}
	f := clusterdiagnostic.Classify(syscall.EHOSTUNREACH)
	if f.Reason != "route_or_permission_denied" || !strings.Contains(f.Hint, "allowlist") || !strings.Contains(f.Hint, "does not prove permission denial") {
		t.Fatalf("network failure overclaims evidence or hides alternate path: %+v", f)
	}
}

func TestDoctorMacOSHostingAdviceIsUnknownAndReadOnly(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("macOS-specific doctor observation")
	}
	home := t.TempDir()
	t.Setenv("HOME", home)
	var out, stderr bytes.Buffer
	if code := runDoctor(nil, &out, &stderr); code != 0 {
		t.Fatalf("doctor %d: %s", code, &stderr)
	}
	var response struct {
		Checks []doctorCheck `json:"checks"`
	}
	if err := json.Unmarshal(out.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, check := range response.Checks {
		if check.Check == "local_network_permission" {
			found = true
			if check.Status != "unknown" || check.Detail != macOSLocalNetworkAdvice {
				t.Fatalf("unexpected permission claim: %+v", check)
			}
		}
	}
	if !found {
		t.Fatal("missing local_network_permission")
	}
	entries, err := os.ReadDir(home)
	if err != nil || len(entries) != 0 {
		t.Fatalf("read-only doctor created local state: %v %v", entries, err)
	}
}

func readHostingDoc(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("../../docs", name))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestHeadlessDocumentationContract(t *testing.T) {
	for _, name := range []string{"cluster-quickstart.md", "cluster-agent.md", "cluster-services.md", "feature-matrix.md"} {
		doc := readHostingDoc(t, name)
		for _, text := range []string{"LaunchAgent", "allowlist", "cluster-headless.md", "cluster-network-evidence.md"} {
			if !strings.Contains(doc, text) {
				t.Errorf("%s missing %q", name, text)
			}
		}
	}
	doc := readHostingDoc(t, "cluster-headless.md")
	contract := strings.Join(strings.Fields(doc), " ")
	for _, text := range []string{
		"tn3179-understanding-local-network-privacy", "sudo defaults read", "sudo defaults delete",
		"AllowedEthernetLocalNetworkAddresses -array", "AllowedWiFiLocalNetworkAddresses -array",
		"Restart the Mac", "replaces", "system-wide for every program", "outgoing", "mTLS",
		"not been verified on a physical Mac", "before first login", "FileVault", "sleep/wake",
		"Serving commands do not accept `-key-backend`", "init", "enroll", "not rotated or size-bounded",
		"PID note alone never authorizes", "not readiness", "installed services only",
	} {
		if !strings.Contains(contract, text) {
			t.Errorf("headless guide missing %q", text)
		}
	}
	for _, name := range []string{"cluster-headless.md", "cluster-network-evidence.md"} {
		text := readHostingDoc(t, name)
		if strings.Contains(text, "mixlab-planning") || strings.Contains(text, "/private/tmp/") {
			t.Errorf("%s exposes private evidence paths", name)
		}
		for _, ip := range regexp.MustCompile(`\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b`).FindAllString(text, -1) {
			if ip != "192.168.1.10" && ip != "192.168.1.20" {
				t.Errorf("%s contains a non-placeholder address %s", name, ip)
			}
		}
	}
}

func TestHeadlessDocumentedStartCommands(t *testing.T) {
	doc := readHostingDoc(t, "cluster-headless.md")
	// Check shell syntax without executing policy writes, SSH or process signals.
	if bash, err := exec.LookPath("bash"); err == nil {
		for _, block := range regexp.MustCompile("(?s)```bash\n(.*?)\n```").FindAllStringSubmatch(doc, -1) {
			cmd := exec.Command(bash, "-n")
			cmd.Stdin = strings.NewReader(block[1])
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("invalid shell example: %v %s", err, out)
			}
		}
	} else {
		t.Log("bash unavailable; shell syntax check omitted")
	}
	seen := map[string]bool{}
	for _, line := range strings.Split(strings.ReplaceAll(doc, "\\\n", " "), "\n") {
		if !strings.HasPrefix(line, `nohup "$bin" `) {
			continue
		}
		fields := strings.Fields(line)
		role := fields[2]
		command := []string{role}
		if role != "agent" {
			if fields[3] != "serve" {
				t.Fatalf("invalid foreground command: %s", line)
			}
			command = append(command, "serve")
		}
		var out, stderr bytes.Buffer
		if code := run(append(command, "-help"), &out, &stderr); code != 0 {
			t.Fatalf("help %v: %d %s", command, code, &stderr)
		}
		for _, field := range fields {
			if strings.HasPrefix(field, "-") && !strings.Contains(stderr.String(), "  "+field+" ") && !strings.Contains(stderr.String(), "  "+field+"\n") {
				t.Errorf("%v does not accept documented flag %s", command, field)
			}
		}
		for _, text := range []string{"</dev/null", "2>&1", " &"} {
			if !strings.Contains(line, text) {
				t.Errorf("%s does not detach all streams: %s", role, line)
			}
		}
		seen[role] = true
	}
	for _, role := range []string{"authority", "agent", "coordinator"} {
		if !seen[role] {
			t.Errorf("missing detached %s example", role)
		}
	}
}
