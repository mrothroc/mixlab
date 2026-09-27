//go:build darwin || linux

package train

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/artifact"
	artifactlocal "github.com/mrothroc/mixlab/artifact/local"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/workerhost"
	"github.com/mrothroc/mixlab/workerhost/contract"
	"github.com/mrothroc/mixlab/workerjob"
	"io"
)

// Unlike the direct supervisor smoke, this crosses the production cluster
// helper executable, durable attempt journal and physical cleanup receipt.
func TestManagedGuardianCLITraining(t *testing.T) {
	cli, cluster := os.Getenv("MIXLAB_MANAGED_CLI"), os.Getenv("MIXLAB_CLUSTER_CLI")
	if cli == "" || cluster == "" {
		t.Skip("set MIXLAB_MANAGED_CLI and MIXLAB_CLUSTER_CLI to matched local builds")
	}
	a := managedFixture(t)
	a.OutputMaxBytes = 1 << 20
	a.JobID, a.AttemptID = strings.Repeat("1", 32), strings.Repeat("2", 32)
	a.View.LaunchAttemptID = a.AttemptID
	build, err := workerjob.FileDigest(cli)
	if err != nil {
		t.Fatal(err)
	}
	guardianBuild, err := workerjob.FileDigest(cluster)
	if err != nil {
		t.Fatal(err)
	}
	a.BuildID = build
	for i, port := range managedRingPorts(t) {
		a.RingAddresses[i][0] = fmt.Sprintf("127.0.0.1:%d", port)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	type result struct {
		out  contract.Outcome
		err  error
		dir  string
		rank int
	}
	results := make(chan result, 2)
	for rank := range 2 {
		b := a
		b.View.LocalRank = rank
		b.View.LocalMemberID = b.View.Membership.OrderedMembers[rank].MemberID
		base := managedPrivateDir(t)
		root, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(base.Dir(), "runtime")}, statehome.Context{Kind: statehome.Worker})
		if err != nil {
			t.Fatal(err)
		}
		store, err := workerhost.InitializeRuntimeStore(root)
		if err != nil {
			t.Fatal(err)
		}
		runtime, err := workerhost.NewGuardianRuntime(store, cli, build, cluster, guardianBuild, 60*time.Second, 3*time.Second)
		if err != nil {
			t.Fatal(err)
		}
		dir, err := store.Attempt(ctx, a.JobID, a.AttemptID)
		if err != nil {
			t.Fatal(err)
		}
		host, guardian, err := runtime.Open(ctx, a.JobID, a.AttemptID)
		if err != nil {
			t.Fatal(err)
		}
		approval := contract.Approved{Version: contract.Version, ManifestHash: strings.Repeat("a", 64), TransportHash: strings.Repeat("b", 64), Assignment: b,
			Limits: contract.Limits{CPUSeconds: 3600, MemoryBytes: 8 << 30, DiskBytes: 1 << 30, LogBytes: 1 << 20}}
		go func() {
			out, err := host.Run(ctx, approval, guardian)
			if err == nil {
				again, e := host.Run(ctx, approval, guardian)
				if e != nil || again != out {
					err = fmt.Errorf("terminal retry changed: %v", e)
				}
			}
			results <- result{out, err, dir.Dir(), rank}
		}()
	}
	for range 2 {
		r := <-results
		if r.err != nil || r.out.Kind != contract.Exited || !r.out.NoChild || r.out.PID <= 0 {
			cancel()
			log, _ := os.ReadFile(filepath.Join(r.dir, "worker.log"))
			t.Errorf("guarded native worker: %+v %v\n%s", r.out, r.err, log)
			continue
		}
		if r.rank == 0 {
			p, err := statehome.Resolve(statehome.Options{ExactDir: r.dir}, statehome.Context{Kind: statehome.Worker})
			if err != nil {
				t.Fatal(err)
			}
			b, err := p.ReadFile(workerjob.OutputReceiptFile)
			if err != nil {
				t.Fatal("native output receipt missing", err)
			}
			var ref artifact.Ref
			if err := json.Unmarshal(b, &ref); err != nil {
				t.Fatal(err)
			}
			store, err := artifactlocal.Open(p)
			if err != nil {
				t.Fatal(err)
			}
			if err := store.Copy(context.Background(), ref, io.Discard); err != nil {
				t.Fatal("native output verification", err)
			}
		}
	}
}
