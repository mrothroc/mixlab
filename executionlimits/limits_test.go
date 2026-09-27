package executionlimits

import "testing"

func TestLimitsBoundsAndPolicy(t *testing.T) {
	base := Limits{RuntimeSeconds: 60, CPUSeconds: 120, MemoryBytes: 1 << 30, DiskBytes: 1 << 30, LogBytes: 1 << 20}
	if err := base.Validate(); err != nil {
		t.Fatal(err)
	}
	if !base.Within(base) {
		t.Fatal("equal budget rejected")
	}
	for _, mutate := range []func(*Limits){
		func(l *Limits) { l.RuntimeSeconds = 0 }, func(l *Limits) { l.RuntimeSeconds = 7*24*3600 + 1 },
		func(l *Limits) { l.CPUSeconds = -1 }, func(l *Limits) { l.CPUSeconds = 64*7*24*3600 + 1 },
		func(l *Limits) { l.MemoryBytes = 0 }, func(l *Limits) { l.MemoryBytes = 1<<40 + 1 },
		func(l *Limits) { l.DiskBytes = 0 }, func(l *Limits) { l.DiskBytes = 1<<40 + 1 },
		func(l *Limits) { l.LogBytes = 0 }, func(l *Limits) { l.LogBytes = 1<<30 + 1 },
		func(l *Limits) { l.LogBytes = l.DiskBytes + 1 },
	} {
		l := base
		mutate(&l)
		if l.Validate() == nil {
			t.Fatalf("accepted %+v", l)
		}
	}
	for _, mutate := range []func(*Limits){func(l *Limits) { l.RuntimeSeconds++ }, func(l *Limits) { l.CPUSeconds++ }, func(l *Limits) { l.MemoryBytes++ }, func(l *Limits) { l.DiskBytes++ }, func(l *Limits) { l.LogBytes++ }} {
		l := base
		mutate(&l)
		if l.Within(base) {
			t.Fatalf("exceeded policy: %+v", l)
		}
	}
}
