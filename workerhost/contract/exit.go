package contract

import "fmt"

const (
	ProcessNotStarted = "not_started"
	ProcessExited     = "exited"
	ProcessSignaled   = "signaled"
)

// ProcessExit records OS wait evidence, independently of the worker protocol's
// success or failure. The zero value means unknown (including legacy receipts),
// not exit code zero. Scalar fields keep Outcome comparable.
type ProcessExit struct {
	Kind   string `json:"kind"`
	Code   int    `json:"code,omitempty"`
	Signal int    `json:"signal,omitempty"`
}

func (e ProcessExit) Validate() error {
	switch e.Kind {
	case "", ProcessNotStarted:
		if e.Code == 0 && e.Signal == 0 {
			return nil
		}
	case ProcessExited:
		if e.Code >= 0 && e.Code <= 255 && e.Signal == 0 {
			return nil
		}
	case ProcessSignaled:
		if e.Code == 0 && e.Signal > 0 && e.Signal <= 255 {
			return nil
		}
	}
	return fmt.Errorf("invalid OS process exit evidence")
}
