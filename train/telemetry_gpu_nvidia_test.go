package train

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestParseNvidiaSMIGPUUtilPercent(t *testing.T) {
	for _, tc := range []struct {
		name, output string
		want         *float64
	}{
		{"busy", "97\n", ptrFloat64(97)},
		{"idle", "0\n", ptrFloat64(0)},
		{"full", "100\n", ptrFloat64(100)},
		{"multiple", "12\n87\n23\n", ptrFloat64(87)},
		{"whitespace", "\n 42.5 \r\n\n", ptrFloat64(42.5)},
		{"quoted", "\"52\"\n", ptrFloat64(52)},
		{"unsupported_device", "[N/A]\n", nil},
		{"mixed_devices", "[N/A]\n0\n[N/A]\n", ptrFloat64(0)},
		{"empty", "", nil},
		{"invalid", "not a number\n", nil},
		{"nonfinite", "NaN\nInf\n-Inf\n", nil},
		{"out_of_range", "-1\n101\n", nil},
		{"wrong_columns", "0, 97\n", nil},
		{"malformed_csv", "\"97\n", nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := parseNvidiaSMIGPUUtilPercent(tc.output)
			if tc.want == nil {
				if got != nil {
					t.Fatalf("got %g, want unavailable", *got)
				}
			} else if got == nil || *got != *tc.want {
				t.Fatalf("got %v, want %g", got, *tc.want)
			}
		})
	}
}

func TestNvidiaSMISamplerCommand(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("test executable uses a POSIX shell")
	}
	for _, tc := range []struct {
		name, script string
		want         *float64
	}{
		{"success", `test "$#" = 2 && test "$1" = '--query-gpu=utilization.gpu' && test "$2" = '--format=csv,noheader,nounits' || exit 2
printf '12\n97\n'
`, ptrFloat64(97)},
		{"failure_with_output", "printf '97\\n'\nexit 1\n", nil},
		{"unsupported", "printf '[N/A]\\n'\n", nil},
		{"missing_executable", "", nil},
		{"timeout", "while :; do :; done\n", nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			if tc.script != "" {
				if err := os.WriteFile(filepath.Join(dir, "nvidia-smi"), []byte("#!/bin/sh\n"+tc.script), 0o700); err != nil {
					t.Fatal(err)
				}
			}
			t.Setenv("PATH", dir)
			timeout := 5 * time.Second
			if tc.name == "timeout" {
				timeout = 200 * time.Millisecond
			}
			ctx, cancel := context.WithTimeout(context.Background(), timeout)
			defer cancel()
			got := sampleNvidiaSMIGPUUtilPercent(ctx)
			if tc.want == nil {
				if got != nil {
					t.Fatalf("got %g, want unavailable", *got)
				}
				if line := formatTelemetryLine(telemetrySnapshot{GPUUtilPercent: got}); !strings.Contains(line, "gpu_util=n/a") {
					t.Fatalf("missing unavailable marker: %s", line)
				}
			} else if got == nil || *got != *tc.want {
				t.Fatalf("got %v, want %g", got, *tc.want)
			}
			if tc.name == "success" && runtime.GOOS == "linux" {
				if got := sampleGPUUtilPercent(); got == nil || *got != *tc.want {
					t.Fatalf("Linux telemetry wrapper got %v, want %g", got, *tc.want)
				}
			}
			if tc.name == "timeout" && ctx.Err() != context.DeadlineExceeded {
				t.Fatalf("expected timeout, got %v", ctx.Err())
			}
		})
	}
}
