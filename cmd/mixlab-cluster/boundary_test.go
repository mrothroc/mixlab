package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const modulePath = "github.com/mrothroc/mixlab/"

func goCommand(t *testing.T, cgo string, args ...string) []byte {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "go", args...)
	cmd.Dir = "../.."
	cmd.Env = append(os.Environ(), "CGO_ENABLED="+cgo, "GOFLAGS=")
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("go %v (CGO_ENABLED=%s): %v\n%s\n%s", args, cgo, err, out, stderr.String())
	}
	return out
}

type dependency struct {
	ImportPath string
	Standard   bool
	Imports    []string
}

// These pinned modules implement the fixed HPKE suite. Application packages
// may use our adapter, never select algorithms from CIRCL directly.
func hpkeDependency(path string) bool {
	if path == modulePath+"internal/credentialcrypto" || strings.HasPrefix(path, "github.com/cloudflare/circl/") {
		return true
	}
	switch path {
	case "golang.org/x/crypto/cryptobyte", "golang.org/x/crypto/cryptobyte/asn1",
		"golang.org/x/crypto/internal/alias", "golang.org/x/crypto/chacha20",
		"golang.org/x/crypto/internal/poly1305", "golang.org/x/crypto/chacha20poly1305",
		"golang.org/x/crypto/hkdf", "golang.org/x/sys/cpu":
		return true
	}
	return false
}

func dependencies(t *testing.T, cgo, tags, target string) []dependency {
	t.Helper()
	out := goCommand(t, cgo, "list", "-deps", "-json", "-tags="+tags, target)
	d := json.NewDecoder(bytes.NewReader(out))
	var result []dependency
	for {
		var pkg dependency
		if err := d.Decode(&pkg); err != nil {
			if err == io.EOF {
				break
			}
			t.Fatalf("decode go list: %v", err)
		}
		result = append(result, pkg)
	}
	return result
}

func hasComponent(path string, names ...string) bool {
	for _, component := range strings.Split(strings.TrimPrefix(path, modulePath), "/") {
		for _, name := range names {
			if component == name {
				return true
			}
		}
	}
	return false
}

func TestExecDependencyBoundaries(t *testing.T) {
	for _, tags := range []string{"", "mlx"} {
		for _, cgo := range []string{"0", "1"} {
			t.Run("tags="+tags+"/cgo="+cgo, func(t *testing.T) {
				for _, pkg := range dependencies(t, cgo, tags, "./cmd/mixlab-cluster") {
					if strings.HasPrefix(pkg.ImportPath, modulePath) && hasComponent(pkg.ImportPath, "train", "gpu", "learner") {
						t.Errorf("cluster must not link training/MLX components: %s", pkg.ImportPath)
					}
				}
				for _, pkg := range dependencies(t, cgo, tags, "./internal/buildinfo") {
					if !pkg.Standard && pkg.ImportPath != modulePath+"internal/buildinfo" && pkg.ImportPath != modulePath+"workercontrol" && pkg.ImportPath != modulePath+"internal/strictjson" {
						t.Errorf("build identity must remain application-independent: %s", pkg.ImportPath)
					}
				}
				for _, pkg := range dependencies(t, cgo, tags, "./cmd/mixlab") {
					if hpkeDependency(pkg.ImportPath) {
						t.Errorf("training binary imports credential crypto: %s", pkg.ImportPath)
					}
					if strings.HasPrefix(pkg.ImportPath, modulePath) && hasComponent(pkg.ImportPath,
						"trust", "securekeys", "nodecredentials", "transport", "enrollment", "discovery", "nodeagent", "node-agent", "node_agent", "cluster", "mixlab-cluster",
						"admission", "recovery", "coordinator") {
						t.Errorf("training binary imports cluster/trust component: %s", pkg.ImportPath)
					}
				}
				for _, pkg := range dependencies(t, cgo, tags, "./statehome") {
					if strings.HasPrefix(pkg.ImportPath, modulePath) && hasComponent(pkg.ImportPath,
						"trust", "securekeys", "cluster", "admission", "recovery", "coordinator", "learner", "train", "gpu") {
						t.Errorf("state home imports its consumers: %s", pkg.ImportPath)
					}
				}
				for _, target := range []string{"./workerhost", "./workerjob"} {
					for _, pkg := range dependencies(t, cgo, tags, target) {
						if strings.HasPrefix(pkg.ImportPath, modulePath) && hasComponent(pkg.ImportPath,
							"train", "gpu", "arch", "trust", "securekeys", "nodecredentials", "enrollment", "coordinator", "discovery") {
							t.Errorf("worker contract/hosting imports application authority or numerical runtime: %s", pkg.ImportPath)
						}
					}
				}
				for _, pkg := range dependencies(t, cgo, tags, "./nodeagent") {
					if pkg.ImportPath == modulePath+"workerhost" || (strings.HasPrefix(pkg.ImportPath, modulePath) && hasComponent(pkg.ImportPath, "train", "gpu", "arch", "clusterapp", "securekeys", "enrollment", "bootstrap", "authority")) {
						t.Errorf("node management imports process/key/numerical implementation instead of ports: %s", pkg.ImportPath)
					}
				}
				for _, pkg := range dependencies(t, cgo, tags, "./discovery") {
					if strings.HasPrefix(pkg.ImportPath, modulePath) && pkg.ImportPath != modulePath+"discovery" {
						t.Errorf("discovery imports application policy: %s", pkg.ImportPath)
					}
				}
				for _, target := range []string{"./grouptransport", "./transport/ringproxy"} {
					for _, pkg := range dependencies(t, cgo, tags, target) {
						if strings.HasPrefix(pkg.ImportPath, modulePath) && hasComponent(pkg.ImportPath, "train", "gpu", "arch", "workerhost", "workerjob", "nodeagent", "clusterapp", "securekeys", "enrollment", "bootstrap", "authority") {
							t.Errorf("ring plan/adapter imports execution or authority implementation: %s", pkg.ImportPath)
						}
					}
				}
				for _, pkg := range dependencies(t, cgo, tags, "./workercontrol/local") {
					if !pkg.Standard && pkg.ImportPath != modulePath+"workercontrol/local" &&
						pkg.ImportPath != modulePath+"workercontrol" && pkg.ImportPath != modulePath+"statehome" && pkg.ImportPath != modulePath+"internal/strictjson" {
						t.Errorf("local worker adapter imports an application context: %s", pkg.ImportPath)
					}
				}
				for _, pkg := range dependencies(t, cgo, tags, "./trust") {
					if !pkg.Standard && pkg.ImportPath != modulePath+"trust" &&
						pkg.ImportPath != modulePath+"trust/internal/certificates" && !hpkeDependency(pkg.ImportPath) {
						t.Errorf("trust rules import an application context or unreviewed crypto dependency: %s", pkg.ImportPath)
					}
				}
				for _, target := range []string{"./trust/keylifecycle", "./trust/bootstrap", "./trust/enrollment", "./trust/authority", "./trust/workload", "./nodecredentials"} {
					for _, pkg := range dependencies(t, cgo, tags, target) {
						if strings.HasPrefix(pkg.ImportPath, modulePath) && hasComponent(pkg.ImportPath, "train", "gpu", "arch", "workerhost", "workerjob", "workercontrol") {
							t.Errorf("credential lifecycle imports execution components: %s", pkg.ImportPath)
						}
					}
				}
				for _, pkg := range dependencies(t, cgo, tags, "./trust/workload") {
					if strings.HasPrefix(pkg.ImportPath, modulePath) && hasComponent(pkg.ImportPath, "nodeagent", "nodejob", "recruitment", "grouptransport", "clusterapp", "distributed") {
						t.Errorf("workload issuance imports application admission semantics: %s", pkg.ImportPath)
					}
				}
				for _, target := range []string{"./transport/managedtls", "./transport/enrollmenttls", "./transport/snapshottls"} {
					for _, pkg := range dependencies(t, cgo, tags, target) {
						if strings.HasPrefix(pkg.ImportPath, modulePath) && hasComponent(pkg.ImportPath,
							"train", "gpu", "arch", "securekeys", "statehome", "bootstrap", "authority", "enrollment", "keylifecycle", "nodecredentials", "workerhost", "workerjob", "workercontrol") {
							t.Errorf("TLS adapter imports application state or execution context: %s", pkg.ImportPath)
						}
					}
				}
				for _, pkg := range dependencies(t, cgo, tags, "./securekeys") {
					if !pkg.Standard && pkg.ImportPath != modulePath+"securekeys" && pkg.ImportPath != modulePath+"statehome" && !hpkeDependency(pkg.ImportPath) {
						t.Errorf("key-store adapter imports trust policy or application context: %s", pkg.ImportPath)
					}
					if pkg.ImportPath == modulePath+"securekeys" {
						for _, imported := range pkg.Imports {
							if hpkeDependency(imported) && imported != modulePath+"internal/credentialcrypto" {
								t.Errorf("key store bypasses fixed-suite adapter: %s", imported)
							}
						}
					}
				}
				for _, pkg := range dependencies(t, cgo, tags, "./internal/credentialcrypto") {
					if !pkg.Standard && !hpkeDependency(pkg.ImportPath) {
						t.Errorf("crypto adapter imports application policy: %s", pkg.ImportPath)
					}
					if pkg.ImportPath == modulePath+"internal/credentialcrypto" {
						for _, imported := range pkg.Imports {
							if strings.Contains(imported, ".") && imported != "github.com/cloudflare/circl/hpke" {
								t.Errorf("fixed-suite adapter imports an unreviewed implementation: %s", imported)
							}
						}
					}
				}
			})
		}
	}
}

func TestPureBuildAndBinaryIdentity(t *testing.T) {
	dir := t.TempDir()
	var reports []string
	for _, binary := range []string{"mixlab", "mixlab-cluster"} {
		path := filepath.Join(dir, binary)
		goCommand(t, "0", "build", "-o", path, "./cmd/"+binary)
		out, err := exec.Command(path, "-version").CombinedOutput()
		if err != nil {
			t.Fatalf("%s -version: %v\n%s", binary, err, out)
		}
		lines := strings.Split(strings.TrimSpace(string(out)), "\n")
		if len(lines) < 2 || !strings.HasPrefix(lines[0], binary+" ") || lines[1] != "worker_protocol mixlab_worker_control_v1" {
			t.Fatalf("%s identity output: %q", binary, out)
		}
		reports = append(reports, strings.TrimPrefix(lines[0], binary+" ")+"\n"+lines[1])
	}
	if reports[0] != reports[1] {
		t.Fatalf("same-source binary identities differ: %q vs %q", reports[0], reports[1])
	}
	// Selecting MLX must not change the cluster's ability to build without cgo.
	goCommand(t, "0", "build", "-tags=mlx", "-o", filepath.Join(dir, "cluster-mlx"), "./cmd/mixlab-cluster")
}
