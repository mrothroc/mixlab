package integration

// Opt-in LAN measurement harness. The server creates disposable test trust and
// synthetic artifacts only. Its bundle is private test material, never a real
// cluster credential. Transfer it over the operator-approved SSH channel.
import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/coordinator/artifactapi"
	"github.com/mrothroc/mixlab/transport/managedtls"
	"github.com/mrothroc/mixlab/trust"
)

type fillReader byte

func (f fillReader) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = byte(f)
	}
	return len(p), nil
}
func filledRef(size int64, fill byte) artifact.Ref {
	h := sha256.New()
	_, _ = io.CopyN(h, fillReader(fill), size)
	return artifact.Ref{SHA256: hex.EncodeToString(h.Sum(nil)), Bytes: uint64(size)}
}

type lanIdentity struct {
	Chain [][]byte
	Key   []byte
}
type lanCase struct {
	Bytes        int64
	Concurrency  int
	Trial        int
	Fill         []byte
	Upload, Read []artifactapi.Request
}
type lanBundle struct {
	Root          []byte
	Fingerprint   string
	Snapshot      trust.SignedSnapshot
	Now           time.Time
	ServerID, URL string
	Clients       []lanIdentity
	Cases         []lanCase
}

func TestArtifactLAN(t *testing.T) {
	mode := os.Getenv("MIXLAB_P3_LAN_MODE")
	if mode == "" {
		t.Skip("opt-in bounded LAN benchmark")
	}
	if mode == "server" {
		artifactLANServer(t)
		return
	}
	if mode == "client" {
		artifactLANClient(t)
		return
	}
	t.Fatal("invalid LAN mode")
}
func artifactLANServer(t *testing.T) {
	address := os.Getenv("MIXLAB_P3_LAN_ADDR")
	bundlePath := os.Getenv("MIXLAB_P3_LAN_BUNDLE")
	if address == "" || bundlePath == "" {
		t.Fatal("explicit address and private bundle path required")
	}
	large := filledRef(128<<20, 0x35)
	small := filledRef(16<<20, 0x36)
	a := newArtifactFixture(t, large)
	check(t, a.g.Store.PutVerified(context.Background(), large, io.LimitReader(fillReader(0x35), int64(large.Bytes))))
	check(t, a.g.Store.PutVerified(context.Background(), small, io.LimitReader(fillReader(0x36), int64(small.Bytes))))
	a.s.Revision++
	a.s.Momentum.Ref = small
	for i := range a.s.Assignments {
		a.s.Assignments[i].OuterState = a.s.Momentum
	}
	check(t, a.g.Repository.Save(context.Background(), a.s.Plan.Cohort.RunID, 1, coordinator.Change{Next: a.s}))
	handler, e := a.g.Handler()
	check(t, e)
	listener, e := net.Listen("tcp", address)
	check(t, e)
	server := &http.Server{Handler: handler, ReadHeaderTimeout: 5 * time.Second, IdleTimeout: 5 * time.Second, MaxHeaderBytes: 16 << 10}
	done := make(chan error, 1)
	go func() { done <- server.Serve(tls.NewListener(listener, a.g.Policy.ServerConfig())) }()
	b := lanBundle{Root: a.f.a.DER(), Fingerprint: a.f.a.Fingerprint(), Snapshot: a.f.snapshot, Now: a.f.now, ServerID: a.f.serverID, URL: "https://" + listener.Addr().String()}
	for _, c := range a.clients {
		b.Clients = append(b.Clients, lanIdentity{c.Chain, []byte(c.Key.(ed25519.PrivateKey))})
	}
	sequence := 0
	for _, size := range []int64{16 << 20, 128 << 20} {
		for _, concurrency := range []int{1, 2} {
			for trial := range 3 {
				c := lanCase{Bytes: size, Concurrency: concurrency, Trial: trial}
				for i := range concurrency {
					fill := byte(0x60 + sequence)
					sequence++
					ref := filledRef(size, fill)
					up := a.request(t, i, "create", fmt.Sprint("upload-", sequence)).Grant.Body
					up.Artifact.Ref = ref
					read := a.request(t, i, "read", fmt.Sprint("read-", sequence)).Grant.Body
					if size == 16<<20 {
						read.Artifact = a.s.Momentum
					}
					c.Fill = append(c.Fill, fill)
					c.Upload = append(c.Upload, a.sign(t, up))
					c.Read = append(c.Read, a.sign(t, read))
				}
				b.Cases = append(b.Cases, c)
			}
		}
	}
	encoded, e := json.Marshal(b)
	check(t, e)
	check(t, os.WriteFile(bundlePath, encoded, 0600))
	t.Logf("ready address=%s synthetic_disk_bound_mib=1440", listener.Addr())
	var before, after syscall.Rusage
	check(t, syscall.Getrusage(syscall.RUSAGE_SELF, &before))
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, os.Interrupt)
	defer stop()
	select {
	case <-ctx.Done():
	case <-time.After(8 * time.Minute):
	}
	check(t, server.Close())
	e = <-done
	if e != nil && e != http.ErrServerClosed {
		t.Fatal(e)
	}
	check(t, syscall.Getrusage(syscall.RUSAGE_SELF, &after))
	t.Logf("server_cpu_user_seconds=%.3f server_cpu_system_seconds=%.3f peak_rss_bytes=%d", usageSeconds(after.Utime)-usageSeconds(before.Utime), usageSeconds(after.Stime)-usageSeconds(before.Stime), after.Maxrss)
	check(t, os.Remove(bundlePath))
}
func usageSeconds(t syscall.Timeval) float64 { return float64(t.Sec) + float64(t.Usec)/1e6 }
func artifactLANClient(t *testing.T) {
	encoded, e := os.ReadFile(os.Getenv("MIXLAB_P3_LAN_BUNDLE"))
	check(t, e)
	var b lanBundle
	check(t, json.Unmarshal(encoded, &b))
	anchor, e := trust.PinRoot(b.Root, b.Fingerprint, b.Now)
	check(t, e)
	view, e := trust.VerifySnapshot(anchor, b.Snapshot, b.Now)
	check(t, e)
	clients := make([]*http.Client, len(b.Clients))
	for i, c := range b.Clients {
		p, e := managedtls.New(managedtls.Identity{Chain: c.Chain, Key: ed25519.PrivateKey(c.Key)}, func(chain [][]byte, now time.Time) (trust.AuthenticatedPrincipal, error) {
			peer, e := trust.AuthenticatePrincipal(anchor, view, chain, now)
			if e == nil && (peer.Role != trust.Coordinator || peer.Principal != b.ServerID) {
				e = fmt.Errorf("wrong benchmark coordinator")
			}
			return peer, e
		}, func() time.Time { return b.Now })
		check(t, e)
		clients[i], e = p.HTTPClient(time.Minute)
		check(t, e)
		defer clients[i].CloseIdleConnections()
	}
	for _, c := range b.Cases {
		for _, operation := range []string{"upload", "download"} {
			start := time.Now()
			results := make(chan error, c.Concurrency)
			var wg sync.WaitGroup
			for i := range c.Concurrency {
				wg.Add(1)
				go func() {
					defer wg.Done()
					q, route := c.Read[i], artifactapi.ReadRoute
					if operation == "upload" {
						q, route = c.Upload[i], artifactapi.CreateRoute
					}
					var envelope bytes.Buffer
					if e := artifactapi.WriteRequest(&envelope, q); e != nil {
						results <- e
						return
					}
					var body io.Reader = &envelope
					if operation == "upload" {
						body = io.MultiReader(&envelope, io.LimitReader(fillReader(c.Fill[i]), c.Bytes))
					}
					resp, e := clients[i].Post(b.URL+route, "application/octet-stream", body)
					if e != nil {
						results <- e
						return
					}
					defer func() { _ = resp.Body.Close() }()
					switch {
					case operation == "download" && resp.StatusCode == http.StatusOK:
						e = artifact.Copy(context.Background(), io.Discard, resp.Body, q.Grant.Body.Artifact.Ref)
					case operation == "upload" && resp.StatusCode == http.StatusNoContent:
						_, e = io.Copy(io.Discard, resp.Body)
					default:
						out, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
						e = fmt.Errorf("HTTP %d: %s", resp.StatusCode, out)
					}
					results <- e
				}()
			}
			wg.Wait()
			close(results)
			for e := range results {
				check(t, e)
			}
			seconds := time.Since(start).Seconds()
			line, _ := json.Marshal(map[string]any{"operation": operation, "bytes_each": c.Bytes, "concurrency": c.Concurrency, "trial": c.Trial, "seconds": seconds, "aggregate_mib_s": float64(c.Bytes*int64(c.Concurrency)) / (1 << 20) / seconds})
			t.Log(string(line))
		}
	}
}
