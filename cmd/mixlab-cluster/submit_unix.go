//go:build darwin || linux

package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math/big"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/mrothroc/mixlab/discovery"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/recruitment"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust/principal"
	"github.com/mrothroc/mixlab/workerhost"
	"github.com/mrothroc/mixlab/workerjob"
	"github.com/mrothroc/mixlab/workerprobe"
)

func runSubmit(args []string, stdout, stderr io.Writer) int {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	return runSubmitContext(ctx, args, stdout, stderr)
}

func runSubmitContext(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	f := flag.NewFlagSet("mixlab-cluster submit", flag.ContinueOnError)
	f.SetOutput(stderr)
	home := f.String("state-home", "", "Mixlab state root")
	principalDir := f.String("principal-state-dir", "", "enrolled controller identity")
	ca := f.String("cluster-state-dir", "", "local authority to serve temporarily; omit for an existing service")
	endpoint := f.String("authority-endpoint", "", "root-signed authority URL; defaults to the sole signed URL")
	pool := f.String("pool", "local", "local only; no coordinator or elastic workers")
	discover := f.String("discover", "off", "mdns or off; explicit node endpoints work without multicast")
	count := f.Int("workers", 0, "fixed number of workers, 2..64")
	config := f.String("config", "", "training config inspected by the approved local worker")
	worker := f.String("worker-binary", "", "administrator-approved absolute local mixlab executable")
	train := f.String("train", "", "node-local logical dataset selector, never a path")
	dataset := f.String("dataset-id", "", "expected dataset SHA256 shared by all selected nodes")
	journal := f.String("attempt-state-dir", "", "new private durable attempt directory; retain for recovery")
	abort := f.Bool("abort", false, "compensate retained attempt only; never restart training")
	fetch := f.Bool("fetch", false, "retrieve weights or checkpoint from a successful retained attempt; never restart training")
	checkpointAt := f.Uint64("checkpoint-at", 0, "stop after this optimizer attempt and export an exact-resume bundle; keeps schedule horizon")
	resumeFrom := f.String("resume-from", "", "successful checkpoint attempt directory; same ordered nodes, new attempt-state-dir")
	runtime := f.Int("max-runtime-seconds", 3600, "requested wall-time ceiling")
	cpu := f.Int64("max-cpu-seconds", 7200, "requested monitored aggregate CPU budget")
	memory := f.Uint64("max-memory-bytes", 4<<30, "requested monitored resident memory budget")
	disk := f.Uint64("max-disk-bytes", 4<<30, "requested monitored private disk budget")
	logs := f.Uint64("max-log-bytes", 16<<20, "requested bounded retained logs")
	var nodes endpointFlags
	f.Var(&nodes, "node", "explicit agent host:port (repeatable)")
	if err := f.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	invalid := func(err error) int { _, _ = fmt.Fprintln(stderr, "submit:", err); return 2 }
	failed := func(err error) int { _, _ = fmt.Fprintln(stderr, "submit:", err); return 1 }
	if f.NArg() != 0 || *pool != "local" || *journal == "" || (*discover != "off" && *discover != "mdns") {
		return invalid(fmt.Errorf("local pool, attempt-state-dir and mdns/off discovery required; no positional arguments"))
	}
	if *abort && *fetch {
		return invalid(fmt.Errorf("abort and fetch are mutually exclusive"))
	}
	if *checkpointAt > 1<<31-1 {
		return invalid(fmt.Errorf("checkpoint-at exceeds supported optimizer attempt range"))
	}
	if *abort || *fetch {
		var forbidden string
		f.Visit(func(f *flag.Flag) {
			switch f.Name {
			case "state-home", "principal-state-dir", "cluster-state-dir", "authority-endpoint", "attempt-state-dir", "abort", "fetch":
			default:
				forbidden = f.Name
			}
		})
		if forbidden != "" {
			return invalid(fmt.Errorf("abort/fetch uses the retained attempt, not -%s", forbidden))
		}
	} else if *count < 2 || *count > 64 || *config == "" || !filepath.IsAbs(*worker) || *train == "" || *dataset == "" || (*discover == "off" && len(nodes) == 0) {
		return invalid(fmt.Errorf("workers [2,64], config, absolute worker-binary, train selector, dataset-id and node endpoints/discovery required"))
	}
	opts, err := stateOptions(*home, *journal)
	if err != nil {
		return invalid(err)
	}
	path, err := statehome.Resolve(opts, statehome.Context{Kind: statehome.Principal})
	if err != nil {
		return invalid(err)
	}
	limits := nodejob.Limits{RuntimeSeconds: *runtime, CPUSeconds: *cpu, MemoryBytes: *memory, DiskBytes: *disk, LogBytes: *logs}
	if err := limits.Validate(); err != nil {
		return invalid(err)
	}
	err = withController(ctx, *home, *principalDir, *ca, *endpoint, stderr, func(ctx context.Context, p *principal.Store, authority string) error {
		if *fetch {
			return fetchSubmissionOutput(ctx, p, path, stdout)
		}
		if *abort {
			launch, err := recruitment.OpenLaunch(path)
			if err != nil {
				return err
			}
			return launch.Abort(ctx, clusterapp.RecruitmentAbortPorts(p, time.Now))
		}
		b, plan, probe, err := inspectSubmission(ctx, *config, *worker)
		if err != nil {
			return err
		}
		identity, _, err := p.Active(time.Now())
		if err != nil {
			return err
		}
		r := recruitment.Requirements{Cluster: identity.Cluster, Count: *count, BuildID: plan.BuildID, MLXVersion: probe.MLXVersion, DeviceKind: probe.DeviceKind, DType: plan.DType, CustomOps: probe.CustomOps, DatasetSelector: *train, DatasetID: *dataset, Limits: limits, ProbeMaxAge: 10 * time.Minute}
		if err := r.Validate(); err != nil {
			return err
		}
		lookup, err := clusterapp.NodeLookup(p, time.Now)
		if err != nil {
			return err
		}
		hints, err := nodeHints(ctx, *discover, nodes, discovery.MDNS{})
		if err != nil {
			return err
		}
		selection, err := recruitment.Select(ctx, hints, r, lookup, time.Now)
		if err != nil {
			return fmt.Errorf("selection: %w; decisions: %v", err, selection.Decisions)
		}
		ports, err := submissionLoopbackPorts(*count)
		if err != nil {
			return err
		}
		operationsOptions := clusterapp.RecruitmentOptions{Principal: p, Config: b, NumericalPlan: plan, AuthorityEndpoint: authority, LoopbackPorts: ports, Clock: time.Now, CheckpointAt: *checkpointAt}
		run, err := submissionID()
		if err != nil {
			return err
		}
		if *resumeFrom != "" {
			run, err = applySubmissionResume(ctx, *resumeFrom, identity.Principal, &operationsOptions)
			if err != nil {
				return err
			}
			if err := validateResumeSelection(*operationsOptions.ResumeSource, selection); err != nil {
				return err
			}
		}
		operations, err := clusterapp.RecruitmentPorts(operationsOptions)
		if err != nil {
			return err
		}
		attempt, err := submissionID()
		if err != nil {
			return err
		}
		launch, err := recruitment.BeginLaunch(path, recruitment.LaunchPlan{Cluster: identity.Cluster, Controller: identity.Principal, Run: run, Attempt: attempt, Selection: selection, TTLSeconds: 300})
		if err != nil {
			return err
		}
		if err := json.NewEncoder(stdout).Encode(struct {
			Run, Attempt, Journal string
			Selection             recruitment.Selection
		}{run, attempt, path.Dir(), selection}); err != nil {
			return err
		}
		if err := launch.Run(ctx, operations); err != nil {
			return fmt.Errorf("attempt retained at %s; use submit -abort if cleanup is incomplete: %w", path.Dir(), err)
		}
		return fetchSubmissionOutput(ctx, p, path, stdout)
	})
	if err != nil {
		return failed(err)
	}
	return 0
}

func submissionID() (string, error) {
	var b [16]byte
	_, err := rand.Read(b[:])
	return hex.EncodeToString(b[:]), err
}

func inspectSubmission(ctx context.Context, config, binary string) ([]byte, workerprobe.Plan, workerprobe.Report, error) {
	var plan workerprobe.Plan
	var probe workerprobe.Report
	f, err := os.Open(config)
	if err != nil {
		return nil, plan, probe, err
	}
	b, readErr := io.ReadAll(io.LimitReader(f, (256<<10)+1))
	closeErr := f.Close()
	if readErr != nil {
		return nil, plan, probe, readErr
	}
	if closeErr != nil {
		return nil, plan, probe, closeErr
	}
	b, err = nodejob.CanonicalConfig(b)
	if err != nil {
		return nil, plan, probe, err
	}
	binary, err = filepath.EvalSymlinks(binary)
	if err != nil {
		return nil, plan, probe, err
	}
	hash, err := workerjob.FileDigest(binary)
	if err != nil {
		return nil, plan, probe, err
	}
	host, err := workerhost.New(binary, hash)
	if err != nil {
		return nil, plan, probe, err
	}
	dir, err := os.MkdirTemp("", "mixlab-submit-probe-")
	if err != nil {
		return nil, plan, probe, err
	}
	defer func() { _ = os.RemoveAll(dir) }()
	dir, err = filepath.EvalSymlinks(dir)
	if err != nil {
		return nil, plan, probe, err
	}
	path, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Worker})
	if err != nil {
		return nil, plan, probe, err
	}
	plan, err = host.Plan(ctx, path, b)
	if err != nil {
		return nil, plan, probe, err
	}
	probe, err = host.Probe(ctx, path)
	return b, plan, probe, err
}

func submissionLoopbackPorts(count int) ([]int, error) {
	if count < 2 || count > 64 {
		return nil, fmt.Errorf("fixed world requires 2..64 ports")
	}
	var listeners []net.Listener
	defer func() {
		for _, l := range listeners {
			_ = l.Close()
		}
	}()
	ports := make([]int, 0, count)
	// Stay outside macOS's ephemeral client range: a connecting rank must not
	// accidentally claim another rank's not-yet-bound listener port.
	for attempts := 0; len(ports) < count && attempts < count*16; attempts++ {
		n, err := rand.Int(rand.Reader, big.NewInt(10000))
		if err != nil {
			return nil, err
		}
		l, err := net.Listen("tcp4", fmt.Sprintf("127.0.0.1:%d", 20000+n.Int64()))
		if err != nil {
			continue
		}
		listeners = append(listeners, l)
		ports = append(ports, l.Addr().(*net.TCPAddr).Port)
	}
	if len(ports) != count {
		return nil, fmt.Errorf("unable to allocate distinct local ring ports")
	}
	return ports, nil
}
