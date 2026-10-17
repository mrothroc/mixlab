package local

import "time"

// LifetimeOptions is for locally supervised sessions whose owner already
// enforces a finite runtime and process cleanup. Scheduler stalls or intentional
// suspension must not corrupt partially read/written frames. Queues stay bounded;
// the owning context still cancels all I/O on revocation or process loss.
func LifetimeOptions(lifetime time.Duration) DispatcherOptions {
	return DispatcherOptions{ReadTimeout: lifetime, WriteTimeout: lifetime,
		RequestTimeout: lifetime, EnqueueTimeout: lifetime}
}
