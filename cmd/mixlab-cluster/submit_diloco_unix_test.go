//go:build darwin || linux

package main

import (
	"bufio"
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/admission"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust"
)

func dilocoSubmissionArgs(root string) []string {
	return []string{
		"-principal-state-dir", filepath.Join(root, "identity-must-not-be-opened"),
		"-coordinator-endpoint", "https://127.0.0.1:1",
		"-coordinator-id", strings.Repeat("a", 32),
		"-config", filepath.Join(root, "config-must-not-be-read.json"),
		"-worker-binary", filepath.Join(root, "worker-must-not-run"),
		"-train", "train", "-dataset-id", strings.Repeat("b", 64),
		"-validation", "validation", "-validation-id", strings.Repeat("c", 64),
		"-workers", "2", "-node", "127.0.0.1:1",
	}
}

func dilocoReplaceArg(args []string, name, value string) []string {
	out := append([]string{}, args...)
	for i := 0; i+1 < len(out); i += 2 {
		if out[i] == name {
			out[i+1] = value
			return out
		}
	}
	return append(out, name, value)
}

func dilocoRemoveArg(args []string, name string) []string {
	var out []string
	for i := 0; i < len(args); i += 2 {
		if i+1 == len(args) {
			return append(out, args[i])
		}
		if args[i] != name {
			out = append(out, args[i:i+2]...)
		}
	}
	return out
}

func assertDiLoCoUsageFailure(t *testing.T, args []string, want string) {
	t.Helper()
	// Any attempt to open the absent identity would return operational code 1,
	// not usage code 2. Cancellation must not supersede argument validation.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	var out, diagnostic bytes.Buffer
	if code := runSubmitContext(ctx, append([]string{"diloco"}, args...), &out, &diagnostic); code != 2 || out.Len() != 0 || !strings.Contains(diagnostic.String(), want) {
		t.Fatalf("args=%v code=%d out=%s diagnostic=%s; want usage failure %q", args, code, &out, &diagnostic, want)
	}
	for _, unexpected := range []string{"identity-must-not-be-opened", "config-must-not-be-read", "controller trust refresh:", "DiLoCo run_id="} {
		if strings.Contains(diagnostic.String(), unexpected) {
			t.Fatalf("usage validation reached operational work: %s", &diagnostic)
		}
	}
}

func TestSubmitDiLoCoStatusCancelValidationBeforeIdentity(t *testing.T) {
	root := t.TempDir()
	t.Setenv("HOME", root)
	base := dilocoSubmissionArgs(root)[:6]
	for _, operation := range []string{"-status", "-cancel"} {
		t.Run(operation, func(t *testing.T) {
			retained := append(append([]string{}, base...), "-run-id", strings.Repeat("d", 32), operation)
			assertDiLoCoUsageFailure(t, dilocoRemoveArg(retained, "-run-id"), "status/cancel requires run-id")
			assertDiLoCoUsageFailure(t, dilocoReplaceArg(retained, "-run-id", ""), "status/cancel requires run-id")
			for _, required := range []string{"-principal-state-dir", "-coordinator-endpoint", "-coordinator-id"} {
				assertDiLoCoUsageFailure(t, dilocoRemoveArg(retained, required), "explicit controller identity")
			}
			// Explicit defaults are still forbidden: recovery cannot change the
			// retained run's training, discovery, model or resource policy.
			for _, option := range [][2]string{
				{"-config", "changed.json"}, {"-worker-binary", "/unused/worker"},
				{"-train", "changed"}, {"-dataset-id", strings.Repeat("e", 64)},
				{"-validation", "changed"}, {"-validation-id", strings.Repeat("f", 64)},
				{"-validation-batches", "1"}, {"-target-loss", "0"}, {"-initial-model", ""},
				{"-workers", "2"}, {"-discover", "off"}, {"-node", "127.0.0.1:1"},
				{"-max-runtime-seconds", "3600"}, {"-max-cpu-seconds", "7200"},
				{"-max-memory-bytes", "4294967296"}, {"-max-disk-bytes", "4294967296"}, {"-max-log-bytes", "16777216"},
			} {
				t.Run(option[0], func(t *testing.T) {
					assertDiLoCoUsageFailure(t, append(append([]string{}, retained...), option[:]...), "unexpected flag \""+strings.TrimPrefix(option[0], "-")+"\"")
				})
			}
		})
	}
	assertDiLoCoUsageFailure(t, append(append([]string{}, base...), "-run-id", strings.Repeat("d", 32), "-status", "-cancel"), "one operation required")
	for _, operation := range []string{"-status=invalid", "-cancel=invalid"} {
		assertDiLoCoUsageFailure(t, append(append([]string{}, base...), operation), "invalid boolean value")
	}
	entries, err := os.ReadDir(root)
	if err != nil || len(entries) != 0 {
		t.Fatalf("invalid recovery created state: %v %v", entries, err)
	}
}

func TestSubmitDiLoCoNewRunValidationBeforeIdentity(t *testing.T) {
	root := t.TempDir()
	t.Setenv("HOME", root)
	base := dilocoSubmissionArgs(root)
	for _, required := range []string{"-principal-state-dir", "-coordinator-endpoint", "-coordinator-id", "-config", "-worker-binary", "-train", "-dataset-id", "-validation", "-validation-id", "-workers", "-node"} {
		t.Run("missing"+required, func(t *testing.T) {
			want := "new run requires"
			if required == "-principal-state-dir" || required == "-coordinator-endpoint" || required == "-coordinator-id" {
				want = "explicit controller identity"
			}
			assertDiLoCoUsageFailure(t, dilocoRemoveArg(base, required), want)
		})
	}
	for _, option := range [][2]string{
		{"-workers", "0"}, {"-workers", "1"}, {"-workers", "65"},
		{"-worker-binary", "relative-worker"}, {"-config", ""}, {"-train", ""}, {"-dataset-id", ""},
		{"-validation", ""}, {"-validation-id", ""}, {"-validation-batches", "0"},
		{"-discover", "auto"}, {"-run-id", strings.Repeat("d", 32)},
	} {
		t.Run(option[0]+"="+option[1], func(t *testing.T) {
			assertDiLoCoUsageFailure(t, dilocoReplaceArg(base, option[0], option[1]), "new run requires")
		})
	}
	for _, value := range []string{"NaN", "+Inf", "-Inf", "-0.1"} {
		assertDiLoCoUsageFailure(t, dilocoReplaceArg(base, "-target-loss", value), "finite and non-negative")
	}
	for _, option := range [][2]string{
		{"-max-runtime-seconds", "0"}, {"-max-cpu-seconds", "0"}, {"-max-memory-bytes", "0"}, {"-max-disk-bytes", "0"}, {"-max-log-bytes", "0"},
	} {
		assertDiLoCoUsageFailure(t, dilocoReplaceArg(base, option[0], option[1]), "submit diloco:")
	}
	for _, tail := range [][]string{{"extra"}, {"-unknown-option"}, {"-resume", "/unused"}, {"-cluster-state-dir", "/unused"}, {"-workers", "not-an-integer"}, {"-validation-batches", "-1"}, {"-config"}} {
		assertDiLoCoUsageFailure(t, append(append([]string{}, base...), tail...), "")
	}
	assertDiLoCoUsageFailure(t, dilocoReplaceArg(base, "-node", "https://node:443/path"), "invalid value")
	entries, err := os.ReadDir(root)
	if err != nil || len(entries) != 0 {
		t.Fatalf("invalid submission created state: %v %v", entries, err)
	}
}

func TestSubmitDiLoCoHelpWithoutIdentity(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	var out, diagnostic bytes.Buffer
	if code := run([]string{"submit", "diloco", "-help"}, &out, &diagnostic); code != 0 || out.Len() != 0 {
		t.Fatalf("help %d: %s %s", code, &out, &diagnostic)
	}
	for _, name := range []string{"state-home", "principal-state-dir", "authority-endpoint", "coordinator-endpoint", "coordinator-id", "status", "cancel", "run-id", "config", "worker-binary", "train", "dataset-id", "validation", "validation-id", "validation-batches", "target-loss", "initial-model", "workers", "discover", "node", "max-runtime-seconds", "max-cpu-seconds", "max-memory-bytes", "max-disk-bytes", "max-log-bytes"} {
		if !strings.Contains(diagnostic.String(), "-"+name) {
			t.Errorf("help missing -%s", name)
		}
	}
	for _, obsolete := range []string{"-attempt-state-dir", "-cluster-state-dir", "-resume-from", "-fetch", "-abort"} {
		if strings.Contains(diagnostic.String(), obsolete) {
			t.Errorf("DiLoCo help advertised DDP option %s", obsolete)
		}
	}
}

func TestSubmitDiLoCoRejectsPortableWorkerBeforeRecruitment(t *testing.T) {
	root, serviceArgs, report := coordinatorCLIFixture(t)
	binary := filepath.Join(root, "mixlab-stub")
	goCommand(t, "0", "build", "-tags=", "-o", binary, "./cmd/mixlab")
	serviceArgs[len(serviceArgs)-1] = availableAddress(t)
	var initOut, initErr bytes.Buffer
	if code := runCoordinatorContext(context.Background(), "init", serviceArgs, &initOut, &initErr); code != 0 {
		t.Fatalf("init %d: %s", code, &initErr)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	reader, writer := io.Pipe()
	defer func() { _ = reader.Close() }()
	done := make(chan int, 1)
	go func() {
		done <- runCoordinatorContext(ctx, "serve", serviceArgs, writer, io.Discard)
		_ = writer.Close()
	}()
	defer func() {
		cancel()
		if code := <-done; code != 0 {
			t.Errorf("serve exit %d", code)
		}
	}()
	if line, err := bufio.NewReader(reader).ReadString('\n'); err != nil || !strings.Contains(line, "coordinator ready") {
		t.Fatalf("coordinator readiness: %q %v", line, err)
	}
	controller := fixturePrincipal(report, trust.Controller)
	coordinator := fixturePrincipal(report, trust.Coordinator)
	base := dilocoSubmissionArgs(root)
	for _, option := range [][2]string{
		{"-principal-state-dir", controller.Directory}, {"-coordinator-endpoint", "https://" + serviceArgs[len(serviceArgs)-1]},
		{"-coordinator-id", coordinator.Principal}, {"-worker-binary", binary},
	} {
		base = dilocoReplaceArg(base, option[0], option[1])
	}
	for _, mode := range []string{"diloco", "ddp"} {
		t.Run(mode, func(t *testing.T) {
			config := filepath.Join(root, mode+".json")
			distributed := `{"mode":"diloco","inner_steps":2,"outer_optimizer":"nesterov","outer_lr":1,"outer_momentum":0,"aggregation":"uniform"}`
			if mode == "ddp" {
				distributed = `{"mode":"ddp","backend":"ring"}`
			}
			b := []byte(`{"model_dim":8,"vocab_size":16,"seq_len":4,"blocks":[{"type":"plain","heads":2}],"training":{"objective":"causal","optimizer":"adamw","steps":8,"batch_tokens":8,"lr":0.001,"distributed":` + distributed + `}}`)
			if err := os.WriteFile(config, b, 0600); err != nil {
				t.Fatal(err)
			}
			_, plan, probe, err := inspectSubmission(ctx, config, binary)
			if err != nil || probe.DeviceKind != "none" || (plan.Learner != nil) != (mode == "diloco") {
				t.Fatalf("portable fixture: learner=%v device=%s err=%v", plan.Learner, probe.DeviceKind, err)
			}
			var out, diagnostic bytes.Buffer
			code := runSubmitDiLoCoContext(ctx, dilocoReplaceArg(base, "-config", config), &out, &diagnostic)
			if code != 1 || out.Len() != 0 || !strings.Contains(diagnostic.String(), "R2 requires a DiLoCo config and Metal worker") || strings.Contains(diagnostic.String(), "DiLoCo run_id=") {
				t.Fatalf("submit %d: out=%s diagnostic=%s", code, &out, &diagnostic)
			}
		})
	}
	path, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(root, "admission-state-dir")}, statehome.Context{Kind: statehome.Principal})
	if err != nil {
		t.Fatal(err)
	}
	inbox, err := admission.OpenInbox(path)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := inbox.IDs(ctx)
	if err != nil || len(ids) != 0 {
		t.Fatalf("unsupported worker accepted submissions: %v %v", ids, err)
	}
}
