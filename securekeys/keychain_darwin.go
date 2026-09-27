//go:build darwin && cgo

package securekeys

/*
#cgo LDFLAGS: -framework Security -framework CoreFoundation
#cgo CFLAGS: -Wno-deprecated-declarations
#include <Security/Security.h>
#include <CoreFoundation/CoreFoundation.h>
#include <stdlib.h>
#include <string.h>

static CFMutableDictionaryRef key_query(SecKeychainRef kc, const char *scope, const char *id, int add) {
    CFMutableDictionaryRef q = CFDictionaryCreateMutable(NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    CFStringRef service = CFStringCreateWithFormat(NULL, NULL, CFSTR("org.mixlab.signing.v1.%s"), scope);
    CFStringRef account = CFStringCreateWithCString(NULL, id, kCFStringEncodingUTF8);
    CFDictionarySetValue(q, kSecClass, kSecClassGenericPassword);
    CFDictionarySetValue(q, kSecAttrService, service);
    CFDictionarySetValue(q, kSecAttrAccount, account);
    CFDictionarySetValue(q, kSecUseAuthenticationUI, kSecUseAuthenticationUIFail);
    if (add) {
        CFDictionarySetValue(q, kSecUseKeychain, kc);
    } else {
        CFArrayRef search = CFArrayCreate(NULL, (const void **)&kc, 1, &kCFTypeArrayCallBacks);
        CFDictionarySetValue(q, kSecMatchSearchList, search);
        CFRelease(search);
    }
    CFRelease(service);
    CFRelease(account);
    return q;
}

static OSStatus key_add(SecKeychainRef kc, const char *scope, const char *id, const void *value, int length) {
    CFMutableDictionaryRef q = key_query(kc, scope, id, 1);
    CFDataRef data = CFDataCreate(NULL, value, length);
    CFDictionarySetValue(q, kSecValueData, data);
    OSStatus result = SecItemAdd(q, NULL);
    CFRelease(data);
    CFRelease(q);
    return result;
}

static SecKeychainRef key_default(OSStatus *status) {
    SecKeychainRef ref = NULL;
    *status = SecKeychainCopyDefault(&ref);
    return ref;
}

static OSStatus key_read(SecKeychainRef kc, const char *scope, const char *id, void *value, int *length) {
    CFMutableDictionaryRef q = key_query(kc, scope, id, 0);
    CFDictionarySetValue(q, kSecReturnData, kCFBooleanTrue);
    CFDictionarySetValue(q, kSecMatchLimit, kSecMatchLimitOne);
    CFTypeRef result = NULL;
    OSStatus status = SecItemCopyMatching(q, &result);
    if (status == errSecSuccess) {
        if (!result || CFGetTypeID(result) != CFDataGetTypeID() || CFDataGetLength(result) > *length) {
            status = errSecDecode;
        } else {
            *length = (int)CFDataGetLength(result);
            memcpy(value, CFDataGetBytePtr(result), *length);
        }
    }
    if (result) CFRelease(result);
    CFRelease(q);
    return status;
}

static OSStatus key_delete(SecKeychainRef kc, const char *scope, const char *id) {
    CFMutableDictionaryRef q = key_query(kc, scope, id, 0);
    OSStatus result = SecItemDelete(q);
    CFRelease(q);
    return result;
}
*/
import "C"

import (
	"bytes"
	"fmt"
	"sync"
	"unsafe"
)

// OpenKeychain uses only the current default Keychain, never the global search
// list or synchronizable cloud items. It neither enumerates nor changes other
// applications' items. Operations fail rather than opening authentication UI.
func OpenKeychain(scope string) (*Store, error) {
	if !validHex(scope, 32) {
		return nil, fmt.Errorf("invalid key-store context scope")
	}
	var status C.OSStatus
	ref := C.key_default(&status)
	if err := keychainError(status); err != nil {
		return nil, err
	}
	return newStore(&keychainBackend{ref: ref, scope: scope}, "keychain", scope)
}

type keychainBackend struct {
	mu    sync.Mutex
	ref   C.SecKeychainRef
	scope string
}

func keychainError(status C.OSStatus) error {
	switch status {
	case C.errSecSuccess:
		return nil
	case C.errSecItemNotFound:
		return ErrMissing
	case C.errSecDuplicateItem:
		return ErrExists
	default:
		return fmt.Errorf("%w (Security status %d)", ErrUnavailable, int(status))
	}
}

func (b *keychainBackend) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.ref != 0 {
		C.CFRelease(C.CFTypeRef(b.ref))
		b.ref = 0
	}
	return nil
}

func cstrings(scope, id string) (*C.char, *C.char, func()) {
	s, i := C.CString(scope), C.CString(id)
	return s, i, func() { C.free(unsafe.Pointer(s)); C.free(unsafe.Pointer(i)) }
}

func (b *keychainBackend) create(id string, value []byte) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.ref == 0 {
		return ErrUnavailable
	}
	s, i, done := cstrings(b.scope, id)
	defer done()
	return keychainError(C.key_add(b.ref, s, i, unsafe.Pointer(&value[0]), C.int(len(value))))
}

func (b *keychainBackend) readLocked(id string) ([]byte, error) {
	if b.ref == 0 {
		return nil, ErrUnavailable
	}
	s, i, done := cstrings(b.scope, id)
	defer done()
	value := make([]byte, 1024)
	length := C.int(len(value))
	if err := keychainError(C.key_read(b.ref, s, i, unsafe.Pointer(&value[0]), &length)); err != nil {
		clear(value)
		return nil, err
	}
	return value[:int(length)], nil
}

func (b *keychainBackend) read(id string) ([]byte, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.readLocked(id)
}

func (b *keychainBackend) remove(id string, expected []byte) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	got, err := b.readLocked(id)
	if err != nil {
		return err
	}
	defer clear(got)
	if !bytes.Equal(got, expected) {
		return fmt.Errorf("private key changed before deletion")
	}
	s, i, done := cstrings(b.scope, id)
	defer done()
	return keychainError(C.key_delete(b.ref, s, i))
}
