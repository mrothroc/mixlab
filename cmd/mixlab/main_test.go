package main

import "testing"

func TestRequestedHelpMode(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
	}{
		{name: "space", args: []string{"-mode", "export-hf", "-h"}, want: "export-hf"},
		{name: "equals", args: []string{"-mode=prepare", "-h"}, want: "prepare"},
		{name: "double_dash_equals", args: []string{"--mode=parity", "-h"}, want: "parity"},
		{name: "none", args: []string{"-h"}, want: ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := requestedHelpMode(tt.args); got != tt.want {
				t.Fatalf("requestedHelpMode(%v)=%q want %q", tt.args, got, tt.want)
			}
		})
	}
}

func TestAliasedStringFlagValue(t *testing.T) {
	tests := []struct {
		name    string
		primary string
		alias   string
		flags   map[string]bool
		want    string
		wantErr bool
	}{
		{name: "primary", primary: "legacy", flags: map[string]bool{"output": true}, want: "legacy"},
		{name: "alias", primary: "", alias: "new", flags: map[string]bool{"export-dir": true}, want: "new"},
		{name: "same", primary: "same", alias: "same", flags: map[string]bool{"output": true, "export-dir": true}, want: "same"},
		{name: "conflict", primary: "old", alias: "new", flags: map[string]bool{"output": true, "export-dir": true}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := aliasedStringFlagValue(tt.primary, "export-dir", tt.alias, tt.flags)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tt.want {
				t.Fatalf("got %q want %q", got, tt.want)
			}
		})
	}
}

func TestTelemetryFlagsInTrainingHelpGroups(t *testing.T) {
	for _, mode := range []string{"arch", "arch_race"} {
		groups := modeFlagGroups[mode]
		for _, flagName := range []string{"pprof-addr", "telemetry-out"} {
			if !flagGroupContains(groups, flagName) {
				t.Fatalf("%s help groups missing %s", mode, flagName)
			}
		}
	}
}

func TestPreparePairsFlagsInHelpGroup(t *testing.T) {
	groups := modeFlagGroups["prepare-pairs"]
	for _, flagName := range []string{"pair-in", "pair-out", "vocab-size", "pair-max-len"} {
		if !flagGroupContains(groups, flagName) {
			t.Fatalf("prepare-pairs help groups missing %s", flagName)
		}
	}
}

func TestMinimalPairFactoryFlagsInPrepareHelpGroup(t *testing.T) {
	groups := modeFlagGroups["prepare"]
	for _, flagName := range []string{
		"minimal-pair-weights",
		"minimal-pair-morphology",
		"minimal-pair-report-out",
		"minimal-pair-sample-out",
		"minimal-pair-sample-count",
	} {
		if !flagGroupContains(groups, flagName) {
			t.Fatalf("prepare help groups missing %s", flagName)
		}
	}
}

func TestScoreEBMPLLFlagsInHelpGroup(t *testing.T) {
	groups := modeFlagGroups["score-ebm"]
	for _, flagName := range []string{"score-pll-aggregation", "score-pll-window", "score-pll-attribution-dump", "score-pll-skip-token-ids", "score-position-batch", "score-emit-token-energy"} {
		if !flagGroupContains(groups, flagName) {
			t.Fatalf("score-ebm help groups missing %s", flagName)
		}
	}
}

func flagGroupContains(groups []flagGroup, name string) bool {
	for _, group := range groups {
		for _, got := range group.Names {
			if got == name {
				return true
			}
		}
	}
	return false
}
