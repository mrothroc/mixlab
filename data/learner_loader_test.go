package data

import (
	"fmt"
	"path/filepath"
	"reflect"
	"testing"
)

func TestLearnerPartitionsAndDirectRestore(t *testing.T) {
	dir := t.TempDir()
	tokens := make([]uint16, 129)
	for i := range tokens {
		tokens[i] = uint16(i)
	}
	writeShard(t, dir, "train.bin", tokens)
	seen := map[int]bool{}
	for i := 0; i < 2; i++ {
		id := LearnerIdentity{RunID: "r", WorkerID: fmt.Sprint(i), PartitionID: fmt.Sprint(i), Index: i, Count: 2, Seed: 7}
		l, e := NewLearnerLoader(filepath.Join(dir, "train.bin"), id, 4, 8, 256)
		if e != nil {
			t.Fatal(e)
		}
		for range 16 {
			b, e := l.NextBatch(4)
			if e != nil {
				t.Fatal(e)
			}
			if seen[b.X[0]] {
				t.Fatal("overlapping partition")
			}
			seen[b.X[0]] = true
		}
		saved := l.State()
		a, e := l.NextBatch(12)
		if e != nil {
			t.Fatal(e)
		}
		if e = l.Restore(saved); e != nil {
			t.Fatal(e)
		}
		b, e := l.NextBatch(12)
		if e != nil || !reflect.DeepEqual(a, b) {
			t.Fatal("direct replay mismatch", e)
		}
		saved.Identity.Index = 1 - i
		before := l.State()
		if e = l.Restore(saved); e == nil || l.State() != before {
			t.Fatal("cross-worker restore accepted/mutated")
		}
	}
	id := LearnerIdentity{RunID: "r", WorkerID: "w", PartitionID: "p", Count: 1}
	if _, e := NewLearnerLoader(filepath.Join(dir, "train.bin"), id, 4, 4, 256); e != nil {
		t.Fatal("single learner", e)
	}
}
