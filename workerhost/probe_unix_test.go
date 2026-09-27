//go:build darwin || linux

package workerhost

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workerjob"
	"github.com/mrothroc/mixlab/workerprobe"
)

func helperProbe() int {
	if os.Getenv("WORKERHOST_SECRET") != "" {
		return 40
	}
	mode, _ := os.ReadFile("probe-case")
	if string(mode) == "hang" {
		time.Sleep(30 * time.Second)
		return 41
	}
	if string(mode) == "overflow" {
		fmt.Println(strings.Repeat("x", workerprobe.MaxBytes*2))
		return 0
	}
	if string(mode) == "error" {
		fmt.Fprintln(os.Stderr, "probe initialization failed")
		return 42
	}
	exe, err := os.Executable()
	if err != nil {
		return 43
	}
	hash, err := workerjob.FileDigest(exe)
	if err != nil {
		return 44
	}
	if string(mode) == "wrong-build" {
		hash = strings.Repeat("a", 64)
	}
	r := workerprobe.Report{Version: workerprobe.Version, BuildID: hash, BuildVersion: "helper", WorkerProtocol: workercontrol.Version, OS: runtime.GOOS, Arch: runtime.GOARCH, DeviceKind: "none", Backends: []string{}, DTypes: []string{}, CustomOps: []string{}}
	if err := json.NewEncoder(os.Stdout).Encode(r); err != nil {
		return 45
	}
	return 0
}

func TestApprovedWorkerProbe(t *testing.T) {
	t.Setenv("WORKERHOST_SECRET", "must-not-reach-probe")
	for _, mode := range []string{"success", "error", "wrong-build", "overflow", "hang"} {
		t.Run(mode, func(t *testing.T) {
			s, p := plan(t, "unused")
			if err := os.WriteFile(filepath.Join(p.Directory.Dir(), "probe-case"), []byte(mode), 0600); err != nil {
				t.Fatal(err)
			}
			// Race-instrumented children sleep for one second on clean exit.
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			start := time.Now()
			r, err := s.Probe(ctx, p.Directory)
			if (err == nil) != (mode == "success") {
				t.Fatal(mode, r, err)
			}
			if mode == "success" && (r.Available || r.DeviceKind != "none" || r.BuildID != s.build) {
				t.Fatal("invalid no-MLX probe", r)
			}
			if mode == "error" && !strings.Contains(err.Error(), "probe initialization failed") {
				t.Fatal("lost bounded diagnostic", err)
			}
			if time.Since(start) > 4*time.Second {
				t.Fatal("probe not bounded")
			}
		})
	}
}
