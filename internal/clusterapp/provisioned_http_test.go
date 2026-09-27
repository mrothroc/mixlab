package clusterapp

import (
	"context"
	"net"
	"os"
	"testing"
	"time"

	"github.com/mrothroc/mixlab/trust/enrollment"
	"github.com/mrothroc/mixlab/trust/principal"
)

func TestProvisionedEnrollmentOverTLS(t *testing.T) {
	for _, trailing := range []bool{false, true} {
		t.Run(map[bool]string{false: "origin", true: "trailing-slash"}[trailing], func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
			defer cancel()
			l, err := net.Listen("tcp", "127.0.0.1:0")
			check(t, err)
			defer func() { _ = l.Close() }()
			endpoint := "https://" + l.Addr().String()
			if trailing {
				endpoint += "/"
			}
			p := initializedAt(t, endpoint)
			check(t, InitializeAuthority(ctx, p, testNow))
			a, err := OpenAuthority(ctx, p, testNow)
			check(t, err)
			defer func() { _ = a.Close() }()
			v, err := a.Current(ctx, testNow)
			check(t, err)
			i, err := a.Enrollment.Invite(ctx, enrollment.NodeEnrollment, endpoint, "authority", time.Minute, v, testNow)
			check(t, err)
			defer i.Clear()
			done := make(chan error, 1)
			go func() { done <- ServeInvitation(ctx, l, a, i, func() time.Time { return testNow }) }()
			stage, final := enrolleePaths(t, p)
			s, err := EnrollProvisioned(ctx, i, stage, final, "file", func() time.Time { return testNow })
			check(t, err)
			if s.Principal == "" || s.EnvelopeKey == nil {
				t.Fatal("node was not installed")
			}
			check(t, enrollment.WriteInvitation(p, "consumed.json", i, testNow))
			check(t, enrollment.DestroyInvitation(p, "consumed.json", i))
			if _, err := os.Stat(stage.Dir()); !os.IsNotExist(err) {
				t.Fatal("enrollment staging retained", err)
			}
			installed, err := principal.Open(final, testNow)
			check(t, err)
			check(t, installed.Close())
			select {
			case err := <-done:
				check(t, err)
			case <-time.After(5 * time.Second):
				t.Fatal("temporary invitation server did not stop")
			}
		})
	}
}
