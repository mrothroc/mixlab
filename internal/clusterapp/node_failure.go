package clusterapp

import "fmt"

// Preserve node identity through wrapped/joined errors without parsing URLs or
// error text. Public diagnostics classify Err and never expose its raw string.
type nodeFailure struct {
	node string
	err  error
}

func (e *nodeFailure) Error() string  { return fmt.Sprintf("node %s: %v", e.node, e.err) }
func (e *nodeFailure) Unwrap() error  { return e.err }
func (e *nodeFailure) NodeID() string { return e.node }
