package train

import (
	"testing"
	"time"

	"github.com/mrothroc/mixlab/arch"
)

func TestGridLoaderThroughput(t *testing.T) {
	var s gridThroughput
	s.observe(16, time.Second, 30*time.Second, 31*time.Second, true)
	s.observe(16, time.Second, time.Second, 2500*time.Millisecond, false)
	c, tr, w, wait := s.rates(40 * time.Second)
	if c != 16 || tr != 6.4 || w != .8 || wait != 40 {
		t.Fatal(c, tr, w, wait)
	}
	if c, tr, w, wait = (gridThroughput{}).rates(0); c != 0 || tr != 0 || w != 0 || wait != 0 {
		t.Fatal("nonzero empty rates")
	}
}

func TestGridLoaderResumeHashIgnoresConcurrency(t *testing.T) {
	c, err := LoadArchConfig("../examples/grid_regression_tiny.json")
	if err != nil {
		t.Fatal(err)
	}
	a, err := resumeConfigHash(c)
	if err != nil {
		t.Fatal(err)
	}
	depth := 0
	c.Training.GridLoader = &arch.GridLoaderSpec{PrefetchBatches: &depth, ReadWorkers: 1}
	b, err := resumeConfigHash(c)
	if err != nil || a != b {
		t.Fatal("loader changed resume identity", err)
	}
}
