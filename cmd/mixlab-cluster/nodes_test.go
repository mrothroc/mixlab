package main

import (
	"bytes"
	"context"
	"reflect"
	"testing"

	"github.com/mrothroc/mixlab/discovery"
)

func TestNodeHintsExplicitDoesNotBrowse(t *testing.T) {
	got, err := nodeHints(context.Background(), "off", []string{"node.local:7445"}, nil)
	if err != nil || len(got) != 1 || got[0].Endpoint != "node.local:7445" {
		t.Fatal(got, err)
	}
	more := []discovery.Hint{{Service: discovery.Node, Endpoint: "other.local:7445"}}
	both, err := nodeHints(context.Background(), "mdns", []string{"node.local:7445"}, hintProvider{hints: more})
	if err != nil || !reflect.DeepEqual(both, append(got, more...)) {
		t.Fatal(both, err)
	}
}

func TestNodesCLIValidationAndHelp(t *testing.T) {
	for _, test := range []struct {
		args []string
		code int
	}{
		{[]string{"-help"}, 0}, {nil, 2}, {[]string{"-discover", "auto"}, 2},
		{[]string{"-node", "https://host:1/path"}, 2},
		{[]string{"-node", "127.0.0.1:7445", "-controller-state-dir", t.TempDir() + "/missing"}, 1},
	} {
		var out, err bytes.Buffer
		if got := runNodes(test.args, &out, &err); got != test.code {
			t.Fatal(test.args, got, err.String())
		}
	}
}
