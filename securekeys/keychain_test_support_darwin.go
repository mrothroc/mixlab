//go:build darwin && cgo && keychaintest

package securekeys

/*
#cgo CFLAGS: -Wno-deprecated-declarations
#include <Security/Security.h>
#include <stdlib.h>
*/
import "C"

import (
	"crypto/rand"
	"unsafe"
)

// Compiled only for opt-in disposable-Keychain integration tests. Never opens
// the login Keychain, changes the default, or handles user credentials.
func disposableKeychain(path, scope string) (*Store, func() error, error) {
	password := make([]byte, 32)
	defer clear(password)
	if _, err := rand.Read(password); err != nil {
		return nil, nil, err
	}
	p := C.CString(path)
	defer C.free(unsafe.Pointer(p))
	var ref C.SecKeychainRef
	status := C.SecKeychainCreate(p, C.UInt32(len(password)), unsafe.Pointer(&password[0]), 0, 0, &ref)
	if err := keychainError(status); err != nil {
		return nil, nil, err
	}
	b := &keychainBackend{ref: ref, scope: scope}
	// Cleanup owns a reference even if the store is closed during the test.
	C.CFRetain(C.CFTypeRef(ref))
	s, err := newStore(b, "keychain", scope)
	cleanup := func() error {
		b.mu.Lock()
		defer b.mu.Unlock()
		err := keychainError(C.SecKeychainDelete(ref))
		C.CFRelease(C.CFTypeRef(ref))
		if b.ref != 0 {
			C.CFRelease(C.CFTypeRef(b.ref))
			b.ref = 0
		}
		return err
	}
	return s, cleanup, err
}

func lockTestKeychain(s *Store) error {
	b := s.backend.(*keychainBackend)
	b.mu.Lock()
	defer b.mu.Unlock()
	return keychainError(C.SecKeychainLock(b.ref))
}
