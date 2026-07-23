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

func TestValidateModeHasConfigHelp(t *testing.T) {
	if !flagGroupContains(modeFlagGroups["validate"], "config") {
		t.Fatal("validate help groups missing config")
	}
}

func TestResumeFlagInArchCheckpointHelpGroup(t *testing.T) {
	if !flagGroupContains(modeFlagGroups["arch"], "resume") {
		t.Fatal("arch help groups missing resume")
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

func TestFASTAFlagsInPrepareHelpGroup(t *testing.T) {
	groups := modeFlagGroups["prepare"]
	for _, flagName := range []string{"input-format", "nucleotide-alphabet", "nucleotide-ambiguous-symbols", "nucleotide-invalid-symbol-policy"} {
		if !flagGroupContains(groups, flagName) {
			t.Fatalf("prepare help groups missing %s", flagName)
		}
	}
}

func TestPerRecordFramingFlagsInPrepareHelpGroup(t *testing.T) {
	groups := modeFlagGroups["prepare"]
	for _, flagName := range []string{"frame-per-record", "record-seq-len", "record-pad-id", "record-bos-id", "record-eos-id", "record-overflow"} {
		if !flagGroupContains(groups, flagName) {
			t.Fatalf("prepare help groups missing %s", flagName)
		}
	}
}

func TestSequenceVocabularyFlagsInNativeIOHelpGroups(t *testing.T) {
	for _, mode := range []string{"generate", "score-ebm"} {
		if !flagGroupContains(modeFlagGroups[mode], "sequence-vocab") {
			t.Fatalf("%s help groups missing sequence-vocab", mode)
		}
	}
}

func TestBulkGenerationFlagsInHelpGroup(t *testing.T) {
	groups := modeFlagGroups["generate"]
	for _, flagName := range []string{"num-samples", "gen-batch", "gen-seed", "eos-token-id", "generate-out"} {
		if !flagGroupContains(groups, flagName) {
			t.Fatalf("generate help groups missing %s", flagName)
		}
	}
}

func TestHFExportSpecialTokenFlagsInHelpGroup(t *testing.T) {
	groups := modeFlagGroups["export-hf"]
	for _, flagName := range []string{"bos-token-id", "eos-token-id", "pad-token-id"} {
		if !flagGroupContains(groups, flagName) {
			t.Fatalf("export-hf help groups missing %s", flagName)
		}
	}
}

func TestExplicitIntFlag(t *testing.T) {
	provided := map[string]bool{"eos-token-id": true}
	if got := explicitIntFlag("bos-token-id", -1, provided); got != nil {
		t.Fatalf("unprovided flag returned %v", *got)
	}
	got := explicitIntFlag("eos-token-id", 0, provided)
	if got == nil || *got != 0 {
		t.Fatalf("explicit zero = %v, want pointer to 0", got)
	}
}

func TestConstrainedGenerationFlagsInHelpGroup(t *testing.T) {
	groups := modeFlagGroups["generate"]
	for _, flagName := range []string{"grammar-table", "grammar", "grammar-string", "grammar-prompt-mode", "grammar-on-incomplete", "grammar-max-attempts", "tokenizer-path"} {
		if !flagGroupContains(groups, flagName) {
			t.Fatalf("generate help groups missing %s", flagName)
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
