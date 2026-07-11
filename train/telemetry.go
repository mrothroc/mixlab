package train

import (
	"context"
	"encoding/json"
	"expvar"
	"fmt"
	"net"
	"net/http"
	"net/http/pprof"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/mrothroc/mixlab/gpu"
)

type telemetryRuntime struct {
	state  *telemetryState
	file   *os.File
	enc    *json.Encoder
	server *http.Server
	ln     net.Listener
	mu     sync.Mutex
}

type telemetryState struct {
	mu sync.RWMutex
	s  telemetryRunState
}

type telemetryRunState struct {
	Model                 string             `json:"model,omitempty"`
	Step                  int                `json:"step"`
	TotalSteps            int                `json:"total_steps"`
	Loss                  *float64           `json:"loss,omitempty"`
	ValLoss               *float64           `json:"val_loss,omitempty"`
	LR                    float64            `json:"lr"`
	Objective             string             `json:"objective,omitempty"`
	SeqLen                int                `json:"seq_len"`
	BatchTokens           int                `json:"batch_tokens"`
	ElapsedSeconds        float64            `json:"elapsed_seconds"`
	SteadyElapsedSeconds  float64            `json:"steady_elapsed_seconds"`
	TokensPerSec          float64            `json:"tokens_per_sec"`
	Timing                *telemetryTiming   `json:"timing,omitempty"`
	UpdatedAt             string             `json:"updated_at,omitempty"`
	ComponentLosses       map[string]float64 `json:"component_losses,omitempty"`
	Extra                 map[string]float64 `json:"extra,omitempty"`
	OptimizerSteps        uint64             `json:"optimizer_steps"`
	SkippedOptimizerSteps uint64             `json:"skipped_optimizer_steps"`
	ConsecutiveSkipped    uint64             `json:"consecutive_skipped_optimizer_steps"`
	OptimizerStepSkipped  bool               `json:"optimizer_step_skipped,omitempty"`
}

type telemetryTiming struct {
	DataMS       float64 `json:"data_ms"`
	GPUMS        float64 `json:"gpu_ms"`
	ValidationMS float64 `json:"validation_ms"`
	LogMS        float64 `json:"log_ms"`
}

type telemetryUpdate struct {
	Model                 string
	Step                  int
	TotalSteps            int
	Loss                  float64
	HasLoss               bool
	ValLoss               float64
	HasValLoss            bool
	LR                    float32
	Objective             string
	SeqLen                int
	BatchTokens           int
	Elapsed               time.Duration
	SteadyElapsed         time.Duration
	TokensPerSec          float64
	Timing                telemetryTiming
	HasTiming             bool
	ComponentLosses       map[string]float64
	Extra                 map[string]float64
	OptimizerSteps        uint64
	SkippedOptimizerSteps uint64
	ConsecutiveSkipped    uint64
	OptimizerStepSkipped  bool
}

type telemetrySnapshot struct {
	telemetryRunState
	MLX            telemetryMLX  `json:"mlx"`
	Host           telemetryHost `json:"host"`
	GPUUtilPercent *float64      `json:"gpu_util_percent,omitempty"`
}

type telemetryMLX struct {
	ActiveBytes uint64 `json:"active_bytes"`
	CacheBytes  uint64 `json:"cache_bytes"`
	PeakBytes   uint64 `json:"peak_bytes"`
}

type telemetryHost struct {
	RSSBytes uint64 `json:"rss_bytes"`
}

var (
	telemetryExpvarState atomic.Value
	telemetryExpvarOnce  sync.Once
)

func withTelemetryRuntime(opts TrainOptions, fn func(TrainOptions) error) error {
	if opts.PProfAddr == "" && opts.TelemetryOut == "" {
		return fn(opts)
	}
	rt, err := newTelemetryRuntime(opts.PProfAddr, opts.TelemetryOut)
	if err != nil {
		return err
	}
	defer rt.Close()
	opts.telemetry = rt
	return fn(opts)
}

func newTelemetryRuntime(pprofAddr, telemetryOut string) (*telemetryRuntime, error) {
	rt := &telemetryRuntime{state: newTelemetryState()}
	registerTelemetryExpvar(rt.state)
	if telemetryOut != "" {
		f, err := os.Create(telemetryOut)
		if err != nil {
			return nil, fmt.Errorf("create telemetry output %q: %w", telemetryOut, err)
		}
		rt.file = f
		rt.enc = json.NewEncoder(f)
	}
	if pprofAddr != "" {
		if err := rt.startDebugServer(pprofAddr); err != nil {
			rt.Close()
			return nil, err
		}
	}
	return rt, nil
}

func newTelemetryState() *telemetryState {
	return &telemetryState{}
}

func (rt *telemetryRuntime) Close() {
	if rt == nil {
		return
	}
	if rt.server != nil {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		_ = rt.server.Shutdown(ctx)
		cancel()
	}
	if rt.file != nil {
		_ = rt.file.Close()
	}
}

func (rt *telemetryRuntime) startDebugServer(addr string) error {
	mux := rt.debugMux()
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("start pprof debug server on %s: %w", addr, err)
	}
	rt.ln = ln
	rt.server = &http.Server{Handler: mux}
	go func() {
		if err := rt.server.Serve(ln); err != nil && err != http.ErrServerClosed {
			fmt.Fprintf(os.Stderr, "mixlab debug server stopped: %v\n", err)
		}
	}()
	fmt.Fprintf(os.Stderr, "debug server listening on http://%s/debug/pprof/ and http://%s/debug/mixlab/telemetry\n", addr, addr)
	return nil
}

func (rt *telemetryRuntime) debugMux() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/debug/pprof/", pprof.Index)
	mux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
	mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
	mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
	mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
	mux.Handle("/debug/vars", expvar.Handler())
	mux.HandleFunc("/debug/mixlab/telemetry", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		snap := rt.state.snapshot(true)
		if err := json.NewEncoder(w).Encode(snap); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	})
	return mux
}

func registerTelemetryExpvar(state *telemetryState) {
	telemetryExpvarState.Store(state)
	telemetryExpvarOnce.Do(func() {
		expvar.Publish("mixlab_telemetry", expvar.Func(func() any {
			v := telemetryExpvarState.Load()
			if v == nil {
				return map[string]string{"status": "no telemetry state"}
			}
			state, ok := v.(*telemetryState)
			if !ok || state == nil {
				return map[string]string{"status": "invalid telemetry state"}
			}
			return state.snapshot(false)
		}))
	})
}

func (rt *telemetryRuntime) writeSnapshot(sampleGPU bool) error {
	if rt == nil || rt.enc == nil {
		return nil
	}
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if err := rt.enc.Encode(rt.state.snapshot(sampleGPU)); err != nil {
		return fmt.Errorf("write telemetry snapshot: %w", err)
	}
	return nil
}

func (s *telemetryState) update(u telemetryUpdate) {
	if s == nil {
		return
	}
	next := telemetryRunState{
		Model:                 u.Model,
		Step:                  u.Step,
		TotalSteps:            u.TotalSteps,
		LR:                    float64(u.LR),
		Objective:             u.Objective,
		SeqLen:                u.SeqLen,
		BatchTokens:           u.BatchTokens,
		ElapsedSeconds:        u.Elapsed.Seconds(),
		SteadyElapsedSeconds:  u.SteadyElapsed.Seconds(),
		TokensPerSec:          u.TokensPerSec,
		UpdatedAt:             time.Now().UTC().Format(time.RFC3339Nano),
		ComponentLosses:       cloneTelemetryValues(u.ComponentLosses),
		Extra:                 cloneTelemetryValues(u.Extra),
		OptimizerSteps:        u.OptimizerSteps,
		SkippedOptimizerSteps: u.SkippedOptimizerSteps,
		ConsecutiveSkipped:    u.ConsecutiveSkipped,
		OptimizerStepSkipped:  u.OptimizerStepSkipped,
	}
	if u.HasLoss {
		loss := u.Loss
		next.Loss = &loss
	}
	if u.HasValLoss {
		valLoss := u.ValLoss
		next.ValLoss = &valLoss
	}
	if u.HasTiming {
		timing := u.Timing
		next.Timing = &timing
	}
	s.mu.Lock()
	s.s = next
	s.mu.Unlock()
}

func cloneTelemetryValues(values map[string]float64) map[string]float64 {
	if len(values) == 0 {
		return nil
	}
	cloned := make(map[string]float64, len(values))
	for name, value := range values {
		cloned[name] = value
	}
	return cloned
}

func (s *telemetryState) snapshot(sampleGPU bool) telemetrySnapshot {
	if s == nil {
		return telemetrySnapshot{}
	}
	s.mu.RLock()
	run := s.s
	s.mu.RUnlock()
	mem := gpu.MemoryStatsSnapshot()
	snap := telemetrySnapshot{
		telemetryRunState: run,
		MLX: telemetryMLX{
			ActiveBytes: mem.ActiveBytes,
			CacheBytes:  mem.CacheBytes,
			PeakBytes:   mem.PeakBytes,
		},
		Host: telemetryHost{RSSBytes: hostRSSBytes()},
	}
	if sampleGPU {
		snap.GPUUtilPercent = sampleGPUUtilPercent()
	}
	return snap
}

func formatTelemetryLine(s telemetrySnapshot) string {
	gpuUtil := "n/a"
	if s.GPUUtilPercent != nil {
		gpuUtil = fmt.Sprintf("%.0f%%", *s.GPUUtilPercent)
	}
	line := fmt.Sprintf("[telemetry] step %d/%d tok/s=%.0f gpu_util=%s mlx_active=%s mlx_cache=%s mlx_peak=%s rss=%s",
		s.Step, s.TotalSteps, s.TokensPerSec, gpuUtil,
		formatMiB(s.MLX.ActiveBytes), formatMiB(s.MLX.CacheBytes), formatMiB(s.MLX.PeakBytes),
		formatMiB(s.Host.RSSBytes))
	if s.SkippedOptimizerSteps > 0 {
		line += fmt.Sprintf(
			" optimizer_steps=%d skipped_optimizer_steps=%d consecutive_skips=%d",
			s.OptimizerSteps, s.SkippedOptimizerSteps, s.ConsecutiveSkipped)
		if s.OptimizerStepSkipped {
			line += " optimizer_step_skipped=true"
		}
	}
	return line
}
