package clusterapp

import (
	"context"
	"errors"
	"io"
	"log"
	"net"
	"net/http"
	"time"
)

// newManagedHTTPServer keeps the TLS envelope separate from each owner's
// body-read budget. net/http bounds TLS by the smallest positive header/read/
// write timeout, so none of those fields can retain a shorter body budget.
// Owners still control connection limits, authentication, and shutdown.
func newManagedHTTPServer(ctx context.Context, handler http.Handler, bodyReadTimeout time.Duration) *http.Server {
	h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Handler entry is after TLS and headers. Bodyless requests already
		// have net/http's background disconnect read; a body deadline there
		// would cancel the operation itself. Keep route/context bounds intact.
		if r.ContentLength != 0 || len(r.TransferEncoding) != 0 {
			err := http.NewResponseController(w).SetReadDeadline(time.Now().Add(bodyReadTimeout))
			if err != nil && !errors.Is(err, http.ErrNotSupported) {
				http.Error(w, "request read deadline unavailable", http.StatusServiceUnavailable)
				return
			}
		}
		handler.ServeHTTP(w, r)
	})
	return &http.Server{
		Handler:           h,
		ReadHeaderTimeout: 55 * time.Second,
		ReadTimeout:       55 * time.Second,
		WriteTimeout:      55 * time.Second,
		IdleTimeout:       30 * time.Second,
		MaxHeaderBytes:    8 << 10,
		BaseContext:       func(net.Listener) context.Context { return ctx },
		ErrorLog:          log.New(io.Discard, "", 0),
	}
}
