//go:build darwin || linux

package workerhost

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/mrothroc/mixlab/artifact"
	"github.com/mrothroc/mixlab/statehome"
	wc "github.com/mrothroc/mixlab/workercontrol"
	"github.com/mrothroc/mixlab/workerjob"
)

func TestOutputArtifactStagesOnlyCompleteVerifiedStream(t *testing.T) {
	for _, mode := range []string{"valid", "checksum", "offset", "identity", "begin", "truncated", "unapproved", "oversize", "terminal"} {
		t.Run(mode, func(t *testing.T) {
			dir, err := filepath.EvalSymlinks(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			if err := os.Chmod(dir, 0700); err != nil {
				t.Fatal(err)
			}
			p, err := statehome.Resolve(statehome.Options{ExactDir: dir}, statehome.Context{Kind: statehome.Worker})
			if err != nil {
				t.Fatal(err)
			}
			body := []byte("final weights")
			h := sha256.Sum256(body)
			ref := artifact.Ref{SHA256: hex.EncodeToString(h[:]), Bytes: uint64(len(body))}
			binding := wc.Binding{JobID: "job", AttemptID: "attempt"}
			var seq uint64
			frame := func(kind wc.Kind, value any) wc.Envelope {
				seq++
				e, err := workerjob.Envelope(binding, seq, kind, value)
				if err != nil {
					t.Fatal(err)
				}
				return e
			}
			first := frame(wc.KindArtifactWrite, workerjob.ArtifactMessage{Phase: "begin", Ref: ref})
			chunk := workerjob.ArtifactMessage{Phase: "chunk", Ref: ref, Data: body}
			limit := ref.Bytes
			switch mode {
			case "checksum":
				chunk.Data = []byte("wrong weights")
			case "offset":
				chunk.Offset = 1
			case "identity":
				chunk.Ref.SHA256 = hex.EncodeToString(make([]byte, 32))
			case "begin":
				chunk.Phase, chunk.Data = "begin", nil
			case "unapproved":
				limit = 0
			case "oversize":
				limit--
			}
			messages := []wc.Envelope{frame(wc.KindHeartbeat, workerjob.Event{}), frame(wc.KindArtifactWrite, chunk)}
			if mode != "truncated" {
				messages = append(messages, frame(wc.KindArtifactWrite, workerjob.ArtifactMessage{Phase: "end", Ref: ref, Offset: ref.Bytes}))
			}
			if mode == "terminal" {
				messages[1] = frame(wc.KindTerminalOutcome, workerjob.Event{})
			}
			got, err := receiveOutputArtifact(context.Background(), p, limit, first, func() (wc.Envelope, error) {
				if len(messages) == 0 {
					return wc.Envelope{}, io.EOF
				}
				e := messages[0]
				messages = messages[1:]
				return e, nil
			})
			if mode == "valid" {
				if err != nil || got != ref {
					t.Fatal(got, err)
				}
				b, err := p.ReadFile(workerjob.OutputReceiptFile)
				if err != nil {
					t.Fatal(err)
				}
				var receipt artifact.Ref
				if err := json.Unmarshal(b, &receipt); err != nil || receipt != ref {
					t.Fatal(receipt, err)
				}
			} else {
				if err == nil {
					t.Fatal("invalid artifact accepted")
				}
				entries, err := os.ReadDir(dir)
				if err != nil || len(entries) != 0 {
					t.Fatal("partial publication", entries, err)
				}
			}
		})
	}
}
