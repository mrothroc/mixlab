package clusterapp

import (
	"net/http"

	"github.com/mrothroc/mixlab/trust"
)

func (h *nodeHTTP) output(w http.ResponseWriter, r *http.Request, actor trust.AuthenticatedPrincipal, id string) {
	o, ok := h.jobs.(nodeOutputOperations)
	if !ok {
		http.NotFound(w, r)
		return
	}
	switch r.Method {
	case http.MethodGet:
		if r.ContentLength != 0 || len(r.TransferEncoding) != 0 {
			http.Error(w, "body not allowed", http.StatusBadRequest)
			return
		}
		ref, _, err := o.Output(r.Context(), actor, id, nil)
		nodeResponse(w, ref, err)
	case http.MethodPost:
		var q NodeOutputRequest
		if !h.decode(w, r, &q, 1024, &actor) {
			return
		}
		ref, b, err := o.Output(r.Context(), actor, id, &q)
		nodeResponse(w, NodeOutputChunk{Ref: ref, Offset: q.Offset, Data: b}, err)
	default:
		http.NotFound(w, r)
	}
}
