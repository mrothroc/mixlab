package clusterapp

import "context"

// A timeout cannot make a joined authorization or validation failure retryable.
func contextOnlyFailure(err error) bool {
	if err == nil {
		return false
	}
	if joined, ok := err.(interface{ Unwrap() []error }); ok {
		causes := joined.Unwrap()
		if len(causes) == 0 {
			return false
		}
		for _, cause := range causes {
			if !contextOnlyFailure(cause) {
				return false
			}
		}
		return true
	}
	if wrapped, ok := err.(interface{ Unwrap() error }); ok {
		return contextOnlyFailure(wrapped.Unwrap())
	}
	return err == context.DeadlineExceeded || err == context.Canceled
}
