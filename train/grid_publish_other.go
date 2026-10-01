//go:build !darwin && !linux

package train

import "fmt"

func publishGridDirectory(from, to string) error {
	return fmt.Errorf("atomic grid prediction publication is supported on macOS and Linux only")
}
