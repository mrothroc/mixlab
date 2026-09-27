package data

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestDistributedIdentityCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := DistributedDatasetIdentityContext(ctx, "/not-opened/*.bin"); !errors.Is(err, context.Canceled) {
		t.Fatal("canceled identity read touched dataset", err)
	}
	ctx, cancel = context.WithCancel(context.Background())
	r := datasetIdentityReader{ctx, strings.NewReader("contents")}
	b := make([]byte, 1)
	if n, err := r.Read(b); n != 1 || err != nil || b[0] != 'c' {
		t.Fatal(n, err)
	}
	cancel()
	if _, err := r.Read(b); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}
