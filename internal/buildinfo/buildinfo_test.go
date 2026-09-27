package buildinfo_test

import (
	"runtime/debug"
	"testing"

	"github.com/mrothroc/mixlab/internal/buildinfo"
	"github.com/mrothroc/mixlab/workercontrol"
)

func TestWorkerProtocolGolden(t *testing.T) {
	const golden = "mixlab_worker_control_v1"
	if buildinfo.WorkerProtocol != golden || workercontrol.Version != golden {
		t.Fatalf("worker protocol drift: build=%q transport=%q want=%q", buildinfo.WorkerProtocol, workercontrol.Version, golden)
	}
}

func TestFormatReport(t *testing.T) {
	for _, binary := range []string{"mixlab", "mixlab-cluster"} {
		t.Run(binary, func(t *testing.T) {
			info := &debug.BuildInfo{
				Main: debug.Module{Version: "v1.2.3"},
				Settings: []debug.BuildSetting{
					{Key: "vcs.revision", Value: "0123456789abcdef"},
					{Key: "vcs.time", Value: "2026-09-25T00:00:00Z"},
					{Key: "vcs.modified", Value: "true"},
				},
			}
			want := binary + " v1.2.3 (0123456789ab-dirty, 2026-09-25T00:00:00Z)\nworker_protocol mixlab_worker_control_v1"
			if got := buildinfo.FormatReport(binary, info); got != want {
				t.Fatalf("report = %q, want %q", got, want)
			}
			if got, want := buildinfo.FormatReport(binary, nil), binary+" unknown\nworker_protocol mixlab_worker_control_v1"; got != want {
				t.Fatalf("missing metadata report = %q, want %q", got, want)
			}
		})
	}
}

func TestReportUsesLinkerMetadata(t *testing.T) {
	info, _ := debug.ReadBuildInfo()
	if got, want := buildinfo.Report("mixlab-cluster"), buildinfo.FormatReport("mixlab-cluster", info); got != want {
		t.Fatalf("report = %q, want %q", got, want)
	}
}
