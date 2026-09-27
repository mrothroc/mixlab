package checkpoint

import (
	"bytes"
	"context"
	"encoding/binary"
	"io"
	"testing"
)

func fixture(t *testing.T) []byte {
	t.Helper()
	var b bytes.Buffer
	err := Write(context.Background(), &b, [3]Member{{2, bytes.NewReader([]byte("{}"))}, {5, bytes.NewReader([]byte("model"))}, {5, bytes.NewReader([]byte("state"))}})
	if err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}
func TestCheckpointBundleRoundTripAndClosedShape(t *testing.T) {
	b := fixture(t)
	var got []string
	err := Read(context.Background(), bytes.NewReader(b), func(name string, n uint64, r io.Reader) error {
		value, err := io.ReadAll(r)
		if err != nil {
			return err
		}
		if uint64(len(value)) != n {
			t.Fatal("size")
		}
		got = append(got, name+":"+string(value))
		return nil
	})
	if err != nil || len(got) != 3 || got[0] != Manifest+":{}" || got[1] != Model+":model" || got[2] != State+":state" {
		t.Fatal(got, err)
	}
	for _, which := range []string{"magic", "zero", "oversized", "truncated", "extra", "unconsumed"} {
		t.Run(which, func(t *testing.T) {
			bad := bytes.Clone(b)
			switch which {
			case "magic":
				bad[0] = 0
			case "zero":
				binary.LittleEndian.PutUint64(bad[8:], 0)
			case "oversized":
				binary.LittleEndian.PutUint64(bad[8:], 1<<63)
			case "truncated":
				bad = bad[:len(bad)-1]
			case "extra":
				bad = append(bad, 0)
			}
			err := Read(context.Background(), bytes.NewReader(bad), func(_ string, _ uint64, r io.Reader) error {
				if which == "unconsumed" {
					return nil
				}
				_, err := io.Copy(io.Discard, r)
				return err
			})
			if err == nil {
				t.Fatal("invalid container accepted")
			}
		})
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if Read(ctx, bytes.NewReader(b), func(_ string, _ uint64, r io.Reader) error { _, err := io.Copy(io.Discard, r); return err }) == nil {
		t.Fatal("cancellation ignored")
	}
}

func TestCheckpointBundleWriteRejectsChangedMembers(t *testing.T) {
	for _, n := range []uint64{0, 1, 3, 1 << 63} {
		err := Write(context.Background(), io.Discard, [3]Member{{n, bytes.NewReader([]byte("{}"))}, {1, bytes.NewReader([]byte("m"))}, {1, bytes.NewReader([]byte("s"))}})
		if err == nil {
			t.Fatal("invalid size accepted", n)
		}
	}
}
