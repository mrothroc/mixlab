//go:build darwin || linux

package workerhost

import (
	"context"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/mrothroc/mixlab/statehome"
	wc "github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workerjob"
)

func guardianPrivateDirectory(q guardianBegin) (statehome.Path, error) {
	if !filepath.IsAbs(q.LearnerPrivateDirectory) || filepath.Clean(q.LearnerPrivateDirectory) != q.LearnerPrivateDirectory || pathsOverlap(q.Directory, q.LearnerPrivateDirectory) {
		return statehome.Path{}, fmt.Errorf("separate canonical learner private directory required")
	}
	return statehome.Resolve(statehome.Options{ExactDir: q.LearnerPrivateDirectory}, statehome.Context{Kind: statehome.Worker})
}

// An inherited socket is a local capability held only by the guardian and its
// parent. This lane carries opaque owner RPCs; the supervision lane remains
// available to acknowledge startup and cancel/reap a stuck worker.
func serveGuardianRelay(ctx context.Context, c net.Conn, a workerjob.LearnerAssignment, relay Relay) {
	defer func() { _ = c.Close() }()
	stop := context.AfterFunc(ctx, func() { _ = c.Close() })
	defer stop()
	var sequence uint64
	var bytes uint64
	for count := 0; count < 16_000_000; count++ {
		q, err := wc.ReadFrame(c, wc.MaxFrameBytes)
		if err != nil || q.JobID != a.JobID || q.AttemptID != a.AttemptID || q.Sequence <= sequence {
			return
		}
		sequence = q.Sequence
		call, cancel := context.WithTimeout(ctx, time.Duration(a.RuntimeSeconds)*time.Second)
		out, err := relay(call, q)
		cancel()
		if err != nil {
			return
		}
		out.Version, out.JobID, out.AttemptID = wc.Version, a.JobID, a.AttemptID
		out.Sequence, out.CorrelationID = q.Sequence, q.CorrelationID
		bytes += uint64(len(q.Payload) + len(out.Payload) + 2048)
		if bytes > 1<<40 || c.SetWriteDeadline(time.Now().Add(time.Duration(a.RuntimeSeconds)*time.Second)) != nil || wc.WriteFrame(c, out, wc.MaxFrameBytes) != nil {
			return
		}
	}
}

func inheritedGuardianRelay(a workerjob.LearnerAssignment) (Relay, func(), error) {
	f := os.NewFile(4, "guardian-learner-relay")
	c, err := net.FileConn(f)
	_ = f.Close()
	if err != nil {
		return nil, nil, err
	}
	if _, ok := c.(*net.UnixConn); !ok {
		_ = c.Close()
		return nil, nil, fmt.Errorf("inherited Unix learner relay required")
	}
	var mu sync.Mutex
	relay := func(ctx context.Context, q wc.Envelope) (wc.Envelope, error) {
		mu.Lock()
		defer mu.Unlock()
		if err := ctx.Err(); err != nil {
			return wc.Envelope{}, err
		}
		if q.JobID != a.JobID || q.AttemptID != a.AttemptID {
			return wc.Envelope{}, wc.ErrBinding
		}
		deadline := time.Now().Add(time.Duration(a.RuntimeSeconds) * time.Second)
		if d, ok := ctx.Deadline(); ok && d.Before(deadline) {
			deadline = d
		}
		if err := c.SetDeadline(deadline); err != nil {
			return wc.Envelope{}, err
		}
		stop := context.AfterFunc(ctx, func() { _ = c.Close() })
		defer stop()
		if err := wc.WriteFrame(c, q, wc.MaxFrameBytes); err != nil {
			return wc.Envelope{}, err
		}
		out, err := wc.ReadFrame(c, wc.MaxFrameBytes)
		if err == nil && (out.JobID != a.JobID || out.AttemptID != a.AttemptID || out.Sequence != q.Sequence || out.CorrelationID != q.CorrelationID) {
			err = wc.ErrBinding
		}
		return out, err
	}
	return relay, func() { _ = c.Close() }, nil
}
