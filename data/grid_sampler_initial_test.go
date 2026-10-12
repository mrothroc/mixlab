package data

import (
	"reflect"
	"testing"
)

func TestGridSamplerInitialRecovery(t *testing.T) {
	a, _ := NewGridSampler(7, 42)
	state := a.Snapshot()
	b, _ := NewGridSampler(7, 42)
	if err := b.Restore(state, 0, 3); err != nil {
		t.Fatal(err)
	}
	for step := 0; step < 10; step++ {
		x, ex, px, err := a.Next(3)
		if err != nil {
			t.Fatal(err)
		}
		y, ey, py, err := b.Next(3)
		if err != nil || !reflect.DeepEqual(x, y) || ex != ey || px != py {
			t.Fatal("initial restore changed shuffle", step, err)
		}
	}
	for _, kind := range []string{"cursor", "epoch", "seed", "negative"} {
		t.Run(kind, func(t *testing.T) {
			s, _ := NewGridSampler(7, 42)
			v := s.Snapshot()
			step := 0
			switch kind {
			case "cursor":
				v.Cursor = 1
			case "epoch":
				v.Epoch = 1
			case "seed":
				v.Order[0], v.Order[1] = v.Order[1], v.Order[0]
			case "negative":
				step = -1
			}
			if err := s.Restore(v, step, 3); err == nil {
				t.Fatal("accepted invalid initial state")
			}
		})
	}
}
