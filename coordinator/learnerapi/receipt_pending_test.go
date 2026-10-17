package learnerapi

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"testing"

	"github.com/mrothroc/mixlab/coordinator"
	"github.com/mrothroc/mixlab/diloco/authenticated"
	private "github.com/mrothroc/mixlab/diloco/learner"
	"github.com/mrothroc/mixlab/diloco/run"
	"github.com/mrothroc/mixlab/trust"
)

type receiptLookupRepository struct {
	coordinator.Repository
	err     error
	receipt private.Receipt
}

func (r receiptLookupRepository) Receipt(context.Context, string, string) (private.Receipt, error) {
	return r.receipt, r.err
}

func TestReceiptPendingDispatchRequiresMissingReceipt(t *testing.T) {
	var update private.Update
	b, err := os.ReadFile("../../diloco/testdata/update.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &update); err != nil {
		t.Fatal(err)
	}
	hash, err := run.Digest(update)
	if err != nil {
		t.Fatal(err)
	}
	receipt := private.Receipt{Format: "mixlab_update_receipt_v1", Binding: update.Binding, UpdateID: update.ID, UpdateHash: hash, PrivateHash: update.PrivateHash}
	q := Request{Workload: trust.WorkloadBinding{Run: update.Binding.RunID, Worker: update.WorkerID, Participant: update.ParticipantID}, Call: private.RelayRequest{Operation: "receipt", Update: &update}}
	for _, tc := range []struct {
		name    string
		err     error
		pending bool
	}{
		{"missing", coordinator.ErrNotFound, true},
		{"storage-error", io.ErrUnexpectedEOF, false},
		{"conflict", coordinator.ErrConflict, false},
		{"canceled", context.Canceled, false},
		{"found", nil, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, err := (Gateway{}).dispatch(context.Background(), coordinator.Model{Repository: receiptLookupRepository{err: tc.err, receipt: receipt}}, coordinator.State{}, q)
			if errors.Is(err, private.ErrReceiptPending) != tc.pending {
				t.Fatal("non-missing lookup became pending", err)
			}
			if tc.err == nil {
				if err != nil || out.Reply.Receipt == nil || *out.Reply.Receipt != receipt {
					t.Fatal("committed receipt was lost", err)
				}
			} else if !tc.pending && !errors.Is(err, tc.err) {
				t.Fatal("lookup failure was hidden", err)
			}
		})
	}
}

func TestReceiptPendingResponseStrictDecode(t *testing.T) {
	for _, body := range []string{`{}`, `{"code":"other"}`, `{"code":"receipt_pending","extra":true}`, `{"code":"receipt_pending","code":"receipt_pending"}`, `{"Code":"receipt_pending"}`} {
		var out ReceiptPendingResponse
		if err := authenticated.Decode([]byte(body), &out); err == nil {
			t.Fatalf("accepted malformed pending response %s", body)
		}
	}
	var out ReceiptPendingResponse
	if err := authenticated.Decode([]byte(`{"code":"receipt_pending"}`), &out); err != nil {
		t.Fatal(err)
	}
}
