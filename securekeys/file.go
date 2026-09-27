package securekeys

import (
	"errors"
	"os"

	"github.com/mrothroc/mixlab/statehome"
)

// OpenFile uses an existing validated context directory. It neither creates an
// identity nor repairs insecure permissions. Explicitly select it as the macOS
// fallback; errors opening an existing Keychain identity never invoke it.
func OpenFile(path statehome.Path, scope string) (*Store, error) {
	if err := path.Validate(); err != nil {
		return nil, err
	}
	return newStore(fileBackend{path: path}, "file", scope)
}

type fileBackend struct{ path statehome.Path }

func keyName(id string) string { return "key-" + id + ".json" }
func (f fileBackend) create(id string, b []byte) error {
	err := f.path.CompareAndSwap(keyName(id), nil, b)
	if errors.Is(err, statehome.ErrConflict) {
		return ErrExists
	}
	return err
}
func (f fileBackend) read(id string) ([]byte, error) {
	b, err := f.path.ReadFileLimit(keyName(id), 1024)
	if errors.Is(err, os.ErrNotExist) {
		return nil, ErrMissing
	}
	return b, err
}
func (f fileBackend) remove(id string, expected []byte) error {
	return f.path.CompareAndSwap(keyName(id), expected, nil)
}
