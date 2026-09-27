//go:build !darwin && !linux

package statehome

import (
	"fmt"
	"os"
)

func unsupported() error {
	return fmt.Errorf("%w: filesystem operations require macOS or Linux", ErrUnsafe)
}
func checkOwner(os.FileInfo, bool) error    { return unsupported() }
func checkAncestor(os.FileInfo) error       { return unsupported() }
func openNoFollow(string) (*os.File, error) { return nil, unsupported() }
func lockDirectory(*os.File) error          { return unsupported() }
func unlockDirectory(*os.File)              {}
func tryProcessLock(*os.File) (bool, error) { return false, unsupported() }
func checkPathACL(string) error             { return unsupported() }
func checkFileACL(*os.File) error           { return unsupported() }
