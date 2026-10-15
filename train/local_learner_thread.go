//go:build mlx && cgo && (darwin || linux)

package train

import (
	"runtime"
	"sync"

	"github.com/mrothroc/mixlab/gpu"
)

// All numerical operations, including construction/destruction and restore,
// execute on the same OS thread. A session mutex alone cannot guarantee this.
type threadedLocalLearner struct {
	requests chan func(localLearnerEngine)
	done     chan struct{}
	once     sync.Once
}

func localLearnerOnThread(create func() (localLearnerEngine, error)) (localLearnerEngine, error) {
	t := &threadedLocalLearner{requests: make(chan func(localLearnerEngine)), done: make(chan struct{})}
	ready := make(chan error, 1)
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		defer close(t.done)
		e, err := create()
		ready <- err
		if err != nil {
			return
		}
		defer e.close()
		for f := range t.requests {
			f(e)
		}
	}()
	if e := <-ready; e != nil {
		<-t.done
		return nil, e
	}
	return t, nil
}

func (t *threadedLocalLearner) call(f func(localLearnerEngine)) {
	done := make(chan struct{})
	t.requests <- func(e localLearnerEngine) { f(e); close(done) }
	<-done
}
func (t *threadedLocalLearner) step(b objectiveBatch, s int, lr float32) (v float32, err error) {
	t.call(func(e localLearnerEngine) { v, err = e.step(b, s, lr) })
	return
}
func (t *threadedLocalLearner) weights() (v [][]float32, err error) {
	t.call(func(e localLearnerEngine) { v, err = e.weights() })
	return
}
func (t *threadedLocalLearner) state() (v gpu.TrainerStateSnapshot, err error) {
	t.call(func(e localLearnerEngine) { v, err = e.state() })
	return
}
func (t *threadedLocalLearner) stats() (v gpu.TrainerOptimizerStats, err error) {
	t.call(func(e localLearnerEngine) { v, err = e.stats() })
	return
}
func (t *threadedLocalLearner) install(w [][]float32) (err error) {
	t.call(func(e localLearnerEngine) { err = e.install(w) })
	return
}
func (t *threadedLocalLearner) restore(v gpu.TrainerStateSnapshot, s int) (err error) {
	t.call(func(e localLearnerEngine) { err = e.restore(v, s) })
	return
}
func (t *threadedLocalLearner) evaluate(b objectiveBatch) (v float32, err error) {
	t.call(func(e localLearnerEngine) { v, err = e.evaluate(b) })
	return
}
func (t *threadedLocalLearner) close() { t.once.Do(func() { close(t.requests); <-t.done }) }
