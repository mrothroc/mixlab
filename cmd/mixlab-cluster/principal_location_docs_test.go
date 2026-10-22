package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestHeadlessPrincipalLocationAndCleanupDocumented(t *testing.T) {
	var out, stderr bytes.Buffer
	if code := runInit([]string{"-help"}, &out, &stderr); code != 0 {
		t.Fatal(code)
	}
	for _, want := range []string{"does not relocate principals", "staging/enrollment", "-state-home"} {
		if !strings.Contains(stderr.String(), want) {
			t.Errorf("init help missing %q", want)
		}
	}
	doc := readHostingDoc(t, "cluster-headless.md")
	for _, want := range []string{"authority-principal-state-dir", "principals/<cluster>", "staging/enrollment", "authority_dir", "principals[].directory", "-state-home"} {
		if !strings.Contains(doc, want) {
			t.Errorf("setup/teardown missing %q", want)
		}
	}
}
