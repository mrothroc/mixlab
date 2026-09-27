package main

import "fmt"

func validateProbeFlags(flags map[string]bool, args []string) error {
	if len(args) != 0 {
		return fmt.Errorf("worker-probe does not accept positional arguments")
	}
	for name := range flags {
		if name != "mode" {
			return fmt.Errorf("worker-probe rejects -%s", name)
		}
	}
	return nil
}

func validateManagedFlags(flags map[string]bool, args []string) error {
	if len(args) != 0 {
		return fmt.Errorf("managed-worker does not accept positional arguments")
	}
	for name := range flags {
		if name != "mode" && name != "worker-control-socket" && name != "worker-session-fd" {
			return fmt.Errorf("managed-worker rejects -%s; training inputs come only from authenticated assignment", name)
		}
	}
	if !flags["worker-control-socket"] || !flags["worker-session-fd"] {
		return fmt.Errorf("managed-worker requires hosting socket and inherited descriptor")
	}
	return nil
}
