package nodejob

import (
	"testing"
	"time"

	"github.com/mrothroc/mixlab/trust"
)

func TestWorkloadDeadline(t *testing.T) {
	const created = 1800000000
	m := Manifest{Created: created, Expires: created + 60, Limits: Limits{RuntimeSeconds: 3 * 24 * 3600}}
	got, err := m.WorkloadDeadline()
	if err != nil || got.Unix() != m.Expires+int64(m.Limits.RuntimeSeconds)+300 {
		t.Fatal(got, err)
	}
	m.Limits.RuntimeSeconds = int(trust.MaxWorkloadLifetime/time.Second) - 60 - 300
	got, err = m.WorkloadDeadline()
	if err != nil || got.Sub(time.Unix(created, 0)) != trust.MaxWorkloadLifetime {
		t.Fatal(got, err)
	}
	m.Limits.RuntimeSeconds++
	if _, err := m.WorkloadDeadline(); err == nil {
		t.Fatal("silently truncated an overlong runtime")
	}
	for _, runtime := range []int{-1, 0, int(^uint(0) >> 1)} {
		m.Limits.RuntimeSeconds = runtime
		if _, err := m.WorkloadDeadline(); err == nil {
			t.Fatal("accepted invalid runtime", runtime)
		}
	}
}
