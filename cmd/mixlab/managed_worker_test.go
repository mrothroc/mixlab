package main

import "testing"

func TestWorkerProbeFlags(t *testing.T) {
	if err := validateProbeFlags(map[string]bool{"mode": true}, nil); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"config", "train", "worker-session-fd", "pprof-addr", "cpuprofile"} {
		if err := validateProbeFlags(map[string]bool{"mode": true, name: true}, nil); err == nil {
			t.Fatal("probe accepted", name)
		}
	}
	if err := validateProbeFlags(map[string]bool{"mode": true}, []string{"extra"}); err == nil {
		t.Fatal("probe accepted positional arguments")
	}
}

func TestManagedWorkerFlags(t *testing.T) {
	valid := map[string]bool{"mode": true, "worker-control-socket": true, "worker-session-fd": true}
	if err := validateManagedFlags(valid, nil); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"config", "train", "cpuprofile", "pprof-addr", "resume", "telemetry-out"} {
		valid[name] = true
		if err := validateManagedFlags(valid, nil); err == nil {
			t.Errorf("accepted %s", name)
		}
		delete(valid, name)
	}
	if err := validateManagedFlags(valid, []string{"arbitrary"}); err == nil {
		t.Fatal("accepted positional arguments")
	}
	delete(valid, "worker-session-fd")
	if err := validateManagedFlags(valid, nil); err == nil {
		t.Fatal("accepted missing descriptor")
	}
}
