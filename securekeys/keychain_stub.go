//go:build !darwin || !cgo

package securekeys

// OpenKeychain is unavailable without the native macOS Security adapter.
func OpenKeychain(scope string) (*Store, error) { return nil, ErrUnavailable }
