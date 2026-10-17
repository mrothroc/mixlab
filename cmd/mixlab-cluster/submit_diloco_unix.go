//go:build darwin || linux

package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"time"

	"github.com/mrothroc/mixlab/admission"
	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/coordinator"
	runcontract "github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/discovery"
	"github.com/mrothroc/mixlab/internal/clusterapp"
	"github.com/mrothroc/mixlab/nodeagent"
	"github.com/mrothroc/mixlab/nodejob"
	"github.com/mrothroc/mixlab/recruitment"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust/principal"
	"github.com/mrothroc/mixlab/workerhost"
)

func runSubmitDiLoCoContext(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	f := flag.NewFlagSet("mixlab-cluster submit diloco", flag.ContinueOnError)
	f.SetOutput(stderr)
	home := f.String("state-home", "", "Mixlab state root")
	identity := f.String("principal-state-dir", "", "enrolled submitting controller")
	authority := f.String("authority-endpoint", "", "root-signed authority URL")
	endpoint := f.String("coordinator-endpoint", "", "installed coordinator HTTPS origin")
	coordinatorID := f.String("coordinator-id", "", "exact enrolled coordinator principal ID")
	runID := f.String("run-id", "", "retained run ID for status or pre-commit cancellation")
	status := f.Bool("status", false, "read durable admission/run status")
	cancel := f.Bool("cancel", false, "request bounded pre-commit compensation; cannot cancel committed runs")
	config := f.String("config", "", "DiLoCo training config")
	worker := f.String("worker-binary", "", "administrator-approved absolute Mixlab executable")
	train := f.String("train", "", "node-local logical training dataset selector")
	dataset := f.String("dataset-id", "", "training dataset SHA256")
	validation := f.String("validation", "", "designated learner's logical validation dataset selector")
	validationID := f.String("validation-id", "", "validation dataset SHA256")
	batches := f.Uint64("validation-batches", 1, "fixed validation batch count at every global round")
	target := f.Float64("target-loss", 0, "optional global validation target for early stopping")
	initial := f.String("initial-model", "", "optional float32 model artifact for a new weights-only run; not exact resume")
	count := f.Int("workers", 0, "fixed cohort size, 2..64")
	discover := f.String("discover", "off", "mdns or off")
	runtime := f.Int("max-runtime-seconds", 3600, "requested learner wall-time ceiling")
	cpu := f.Int64("max-cpu-seconds", 7200, "requested aggregate CPU budget per learner")
	memory := f.Uint64("max-memory-bytes", 4<<30, "requested resident memory budget per learner")
	disk := f.Uint64("max-disk-bytes", 4<<30, "requested private disk budget per learner")
	logs := f.Uint64("max-log-bytes", 16<<20, "requested retained logs per learner")
	var nodes endpointFlags
	f.Var(&nodes, "node", "explicit agent host:port (repeatable)")
	if err := f.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	invalid := func(err error) int { _, _ = fmt.Fprintln(stderr, "submit diloco:", err); return 2 }
	if f.NArg() != 0 || *endpoint == "" || *coordinatorID == "" || *identity == "" || (*status && *cancel) {
		return invalid(fmt.Errorf("explicit controller identity, coordinator endpoint/ID and one operation required"))
	}
	var targetLoss *float64
	f.Visit(func(v *flag.Flag) {
		if v.Name == "target-loss" {
			targetLoss = target
		}
	})
	if targetLoss != nil && (math.IsNaN(*targetLoss) || math.IsInf(*targetLoss, 0) || *targetLoss < 0) {
		return invalid(fmt.Errorf("target-loss must be finite and non-negative"))
	}
	if *status || *cancel {
		var forbidden string
		f.Visit(func(v *flag.Flag) {
			switch v.Name {
			case "state-home", "principal-state-dir", "authority-endpoint", "coordinator-endpoint", "coordinator-id", "run-id", "status", "cancel":
			default:
				forbidden = v.Name
			}
		})
		if *runID == "" || forbidden != "" {
			return invalid(fmt.Errorf("status/cancel requires run-id and retained service identity only; unexpected flag %q", forbidden))
		}
	} else if *runID != "" || *count < 2 || *count > 64 || *config == "" || !filepath.IsAbs(*worker) || *train == "" || *dataset == "" || *validation == "" || *validationID == "" || *batches == 0 || (*discover != "off" && *discover != "mdns") || (*discover == "off" && len(nodes) == 0) {
		return invalid(fmt.Errorf("new run requires config, absolute worker-binary, training/validation selectors and hashes, workers [2,64] and node endpoints/discovery"))
	}
	limits := nodejob.Limits{RuntimeSeconds: *runtime, CPUSeconds: *cpu, MemoryBytes: *memory, DiskBytes: *disk, LogBytes: *logs}
	if err := limits.Validate(); err != nil {
		return invalid(err)
	}
	err := withController(ctx, *home, *identity, "", *authority, stderr, func(ctx context.Context, p *principal.Store, _ string) error {
		client, err := clusterapp.NewDiLoCoClient(p, *endpoint, *coordinatorID, time.Now)
		if err != nil {
			return err
		}
		if *status || *cancel {
			v, err := client.Status(ctx, *runID, *cancel)
			if err != nil {
				return err
			}
			if err := json.NewEncoder(stdout).Encode(v); err != nil {
				return err
			}
			if v.RunPhase == coordinator.Failed || v.Admission == "ABORTED" {
				return fmt.Errorf("run ended unsuccessfully; see durable status")
			}
			return nil
		}
		info, err := client.Info(ctx)
		if err != nil {
			return err
		}
		b, plan, probe, err := inspectSubmission(ctx, *config, *worker)
		if err != nil {
			return err
		}
		if plan.Learner == nil || probe.DeviceKind != "metal" {
			return fmt.Errorf("R2 requires a DiLoCo config and Metal worker")
		}
		s, _, err := p.Active(time.Now())
		if err != nil {
			return err
		}
		if info.Cluster != s.Cluster {
			return fmt.Errorf("coordinator cluster mismatch")
		}
		r := recruitment.Requirements{Cluster: s.Cluster, Count: *count, BuildID: plan.BuildID, MLXVersion: probe.MLXVersion, DeviceKind: probe.DeviceKind, DType: plan.DType, CustomOps: probe.CustomOps, DatasetSelector: *train, DatasetID: *dataset, Limits: limits, ProbeMaxAge: 10 * time.Minute}
		lookup, err := clusterapp.NodeLookup(p, time.Now)
		if err != nil {
			return err
		}
		hints, err := nodeHints(ctx, *discover, nodes, discovery.MDNS{})
		if err != nil {
			return err
		}
		selection, err := recruitment.SelectLearners(ctx, hints, r, lookup, time.Now)
		if err != nil {
			return fmt.Errorf("selection: %w; decisions: %v", err, selection.Decisions)
		}
		// The first ordered learner is the designated evaluator. Selection and
		// its dataset identity are immutable after durable acceptance.
		if !slices.Contains(selection.Selected[0].Capabilities.Datasets, nodeagent.Dataset{Selector: *validation, ID: *validationID}) {
			return fmt.Errorf("first selected learner lacks the requested validation dataset")
		}
		id, err := submissionID()
		if err != nil {
			return err
		}
		attempt, err := submissionID()
		if err != nil {
			return err
		}
		if _, err := fmt.Fprintf(stderr, "DiLoCo run_id=%s; retain this ID to query an ambiguous submission response\n", id); err != nil {
			return err
		}
		model, err := uploadDiLoCoInitialModel(ctx, client, *initial, *worker, b)
		if err != nil {
			return err
		}
		now := time.Now().Unix()
		q := admission.Submission{Owner: s.Principal, Launch: recruitment.LaunchPlan{Cluster: s.Cluster, Controller: info.Controller, Run: id, Attempt: attempt, Selection: selection, TTLSeconds: 900}, Deadline: now + 1800,
			Specification: admission.Specification{
				Config: b, CreatedAt: now, BaseModel: model, Outer: plan.Learner.Outer, Endpoint: info.Endpoint, ArtifactStoreID: "coordinator-" + info.Coordinator,
				Coordinator: runcontract.Coordinator{ClusterID: s.Cluster, PrincipalID: info.Coordinator, Role: "coordinator", Audience: "learner-run"},
				Contract:    runcontract.Contract{ConfigHash: plan.ConfigHash, ProgramHash: plan.ProgramHash, WeightLayoutHash: plan.WeightLayoutHash, InnerOptimizerHash: plan.OptimizerHash, DatasetHash: *dataset, BuildHash: plan.BuildID},
				Policy:      coordinator.EvaluationPolicy{WorkerID: clusterapp.DiLoCoWorkerID(id, attempt, selection.Selected[0].Capabilities.Node), ValidationHash: *validationID, Split: *validation, BatchTokens: uint64(plan.Learner.BatchTokens), SequenceLength: uint64(plan.Learner.SequenceLength), Batches: *batches, TargetLoss: targetLoss, TimeoutSeconds: 300},
			},
		}
		out, err := client.Submit(ctx, q)
		if err != nil {
			return fmt.Errorf("run %s may have been accepted; query -status before resubmitting: %w", id, err)
		}
		return json.NewEncoder(stdout).Encode(out)
	})
	if err != nil {
		_, _ = fmt.Fprintln(stderr, "submit diloco:", err)
		return 1
	}
	return 0
}

func uploadDiLoCoInitialModel(ctx context.Context, client *clusterapp.DiLoCoClient, file, binary string, config []byte) (runcontract.Blob, error) {
	if file == "" {
		var b []byte
		err := withSubmissionWorker(binary, func(h *workerhost.Supervisor, p statehome.Path) error {
			var err error
			b, err = h.InitialLearnerModel(ctx, p, config)
			return err
		})
		if err != nil {
			return runcontract.Blob{}, err
		}
		h := sha256.Sum256(b)
		return client.UploadInitialModel(ctx, artifact.Ref{SHA256: hex.EncodeToString(h[:]), Bytes: uint64(len(b))}, bytes.NewReader(b))
	}
	f, err := os.Open(file)
	if err != nil {
		return runcontract.Blob{}, err
	}
	defer func() { _ = f.Close() }()
	s, err := f.Stat()
	if err != nil {
		return runcontract.Blob{}, err
	}
	if !s.Mode().IsRegular() || s.Size() <= 0 || uint64(s.Size()) > artifact.MaxBytes {
		return runcontract.Blob{}, fmt.Errorf("bounded regular float32 model file required")
	}
	h := sha256.New()
	n, err := io.Copy(h, io.LimitReader(f, s.Size()+1))
	if err != nil {
		return runcontract.Blob{}, err
	}
	if n != s.Size() {
		return runcontract.Blob{}, fmt.Errorf("initial model changed while hashing")
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return runcontract.Blob{}, err
	}
	return client.UploadInitialModel(ctx, artifact.Ref{SHA256: hex.EncodeToString(h.Sum(nil)), Bytes: uint64(n)}, f)
}
