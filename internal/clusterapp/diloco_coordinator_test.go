package clusterapp

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net"
	"path/filepath"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/artifact/tensors"
	"github.com/mrothroc/mixlab/diloco/optimizer"
	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/trust/principal"
)

func TestInstalledDiLoCoRestartRefreshAndInitialModelUpload(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	al, err := net.Listen("tcp", "127.0.0.1:0")
	check(t, err)
	ap := initializedAt(t, "https://"+al.Addr().String())
	check(t, InitializeAuthority(ctx, ap, testNow))
	root := filepath.Dir(ap.Dir())
	path := func(name string) statehome.Path {
		p, e := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(root, name)}, statehome.Context{Kind: statehome.Principal})
		check(t, e)
		return p
	}
	l, err := net.Listen("tcp", "127.0.0.1:0")
	check(t, err)
	address := l.Addr().String()
	o := DiLoCoCoordinatorOptions{Principal: path("coordinator"), AdmissionPrincipal: path("controller"), State: path("run-state"), Admission: path("admissions"), Checkpoints: path("checkpoints"), Artifacts: path("artifacts"), Endpoint: "https://" + address, Clock: func() time.Time { return testNow }}
	check(t, InitializeDiLoCoCoordinator(o))
	// Only public trust is stale, not the certificate. Serving must refresh
	// before accepting requests instead of becoming stuck in a restart loop.
	now := testNow.Add(time.Hour)
	o.Clock = func() time.Time { return now }
	a, err := OpenAuthority(ctx, ap, now)
	check(t, err)
	defer func() { check(t, a.Close()) }()
	adone := make(chan error, 1)
	go func() { adone <- ServeAuthority(ctx, al, a, o.Clock) }()
	defer func() { cancel(); check(t, <-adone) }()
	p, err := principal.Open(o.AdmissionPrincipal, now)
	check(t, err)
	defer func() { check(t, p.Close()) }()
	identity, err := principal.Open(o.Principal, now)
	check(t, err)
	s, err := identity.View(now)
	check(t, err)
	check(t, identity.Close())
	client, err := NewDiLoCoClient(p, o.Endpoint, s.Principal, o.Clock)
	check(t, err)
	var payload bytes.Buffer
	check(t, tensors.Write(&payload, []optimizer.Tensor{{Name: "w0_weight", Shape: []int{2}, Values: []float32{1, 2}}}))
	h := sha256.Sum256(payload.Bytes())
	ref := artifact.Ref{SHA256: hex.EncodeToString(h[:]), Bytes: uint64(payload.Len())}
	var previous string
	for attempt := 0; attempt < 2; attempt++ {
		if attempt > 0 {
			l, err = net.Listen("tcp", address)
			check(t, err)
		}
		life, stop := context.WithCancel(ctx)
		ready := make(chan struct{})
		done := make(chan error, 1)
		o.Ready = func() { close(ready) }
		go func() { done <- ServeInstalledDiLoCo(life, l, o) }()
		select {
		case <-ready:
		case e := <-done:
			stop()
			t.Fatal("startup", e)
		case <-ctx.Done():
			stop()
			t.Fatal(ctx.Err())
		}
		func() {
			defer func() { stop(); check(t, <-done) }()
			info, e := client.Info(ctx)
			check(t, e)
			if info.Coordinator != s.Principal {
				t.Fatal("service identity changed")
			}
			model, e := client.UploadInitialModel(ctx, ref, bytes.NewReader(payload.Bytes()))
			check(t, e)
			if previous != "" && previous != model.Ref.SHA256 {
				t.Fatal("restart changed model")
			}
			previous = model.Ref.SHA256
			wrong := ref
			wrong.SHA256 = hex.EncodeToString(make([]byte, 32))
			if _, e = client.UploadInitialModel(ctx, wrong, bytes.NewReader(payload.Bytes())); e == nil {
				t.Fatal("uploaded mismatched model hash")
			}
			bad := []byte("not safetensors")
			hash := sha256.Sum256(bad)
			if _, e = client.UploadInitialModel(ctx, artifact.Ref{SHA256: hex.EncodeToString(hash[:]), Bytes: uint64(len(bad))}, bytes.NewReader(bad)); e == nil {
				t.Fatal("uploaded malformed model")
			}
		}()
	}
}
