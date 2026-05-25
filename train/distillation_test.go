package train

import "testing"

func TestDistillationTeacherProbBufferReuseAndClear(t *testing.T) {
	ensemble := &distillationEnsemble{vocabSize: 4}
	first := ensemble.teacherProbBuffer(3)
	if len(first) != 12 {
		t.Fatalf("first buffer len=%d want 12", len(first))
	}
	for i := range first {
		first[i] = 1
	}

	second := ensemble.teacherProbBuffer(3)
	if len(second) != 12 {
		t.Fatalf("second buffer len=%d want 12", len(second))
	}
	if &first[0] != &second[0] {
		t.Fatal("teacher probability buffer was not reused for same-sized batches")
	}
	for i, v := range second {
		if v != 0 {
			t.Fatalf("second[%d]=%g want cleared zero", i, v)
		}
	}

	larger := ensemble.teacherProbBuffer(5)
	if len(larger) != 20 {
		t.Fatalf("larger buffer len=%d want 20", len(larger))
	}
}
