package securekeys

import (
	"fmt"
	"runtime"

	"github.com/mrothroc/mixlab/statehome"
)

// OpenSelected resolves an explicit backend (or the platform default for NEW
// state). Reopened identities must pass their persisted backend, not an empty
// choice. A Keychain failure is never retried with file storage.
func OpenSelected(backend string, path statehome.Path, scope string) (*Store, error) {
	var err error
	backend, err = ResolveBackend(backend)
	if err != nil {
		return nil, err
	}
	switch backend {
	case "keychain":
		return OpenKeychain(scope)
	case "file":
		return OpenFile(path, scope)
	default:
		return nil, fmt.Errorf("unsupported key backend %q", backend)
	}
}

// ResolveBackend selects a name for NEW state without touching a key store.
// Persist this choice before publication; recovery never chooses a fallback.
func ResolveBackend(backend string) (string, error) {
	if backend == "" {
		if runtime.GOOS == "darwin" {
			backend = "keychain"
		} else {
			backend = "file"
		}
	}
	switch backend {
	case "keychain", "file":
		return backend, nil
	default:
		return "", fmt.Errorf("unsupported key backend %q", backend)
	}
}
