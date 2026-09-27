package statehome

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// PublishStream creates one new immutable protected file without buffering its
// body. The callback must honor its owning operation's deadline and finish all
// validation before returning. Existing targets are never replaced. The caller
// must not reenter this directory's state operations from the callback.
func (p Path) PublishStream(name string, limit int64, write func(io.Writer) error) error {
	if err := segment(name); err != nil {
		return err
	}
	if limit <= 0 || write == nil {
		return fmt.Errorf("bounded stream writer required")
	}
	dir, err := p.lock()
	if err != nil {
		return err
	}
	defer release(dir)
	if err := absent(filepath.Join(p.dir, name)); err != nil {
		return err
	}
	return p.writeStreamLocked(dir, name, func(w io.Writer) error {
		bounded := &streamWriter{writer: w, remaining: limit}
		return errors.Join(write(bounded), bounded.err)
	})
}

type streamWriter struct {
	writer    io.Writer
	remaining int64
	err       error
}

func (w *streamWriter) Write(b []byte) (int, error) {
	if w.err != nil {
		return 0, w.err
	}
	if int64(len(b)) > w.remaining {
		w.err = fmt.Errorf("protected stream exceeds byte limit")
		return 0, w.err
	}
	n, err := w.writer.Write(b)
	w.remaining -= int64(n)
	if err == nil && n != len(b) {
		err = io.ErrShortWrite
	}
	w.err = err
	return n, err
}

// OpenRead validates a protected regular inode and returns a read-only handle.
// The caller owns closure and bounded reading. It does not keep the directory
// locked while consuming a potentially large immutable file.
func (p Path) OpenRead(name string) (*os.File, error) {
	if err := segment(name); err != nil {
		return nil, err
	}
	dir, err := p.lock()
	if err != nil {
		return nil, err
	}
	defer release(dir)
	return openProtected(filepath.Join(p.dir, name))
}
