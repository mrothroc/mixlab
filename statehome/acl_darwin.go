package statehome

import (
	"fmt"
	"os"
	"runtime"
	"syscall"
	"unsafe"
)

type darwinAttrList struct {
	BitmapCount                           uint16
	Reserved                              uint16
	Common, Volume, Directory, File, Fork uint32
}

func checkPathACL(path string) error {
	p, err := syscall.BytePtrFromString(path)
	if err != nil {
		return fmt.Errorf("%w: ACL path: %v", ErrUnsafe, err)
	}
	attrs := darwinAttrList{BitmapCount: 5, Common: 0x00400000}
	var buf [darwinACLBufferSize]byte
	// ATTR_CMN_EXTENDED_SECURITY, FSOPT_NOFOLLOW | FSOPT_REPORT_FULLSIZE.
	_, _, errno := syscall.Syscall6(syscall.SYS_GETATTRLIST,
		uintptr(unsafe.Pointer(p)), uintptr(unsafe.Pointer(&attrs)),
		uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)), 1|4, 0)
	runtime.KeepAlive(p)
	if errno != 0 {
		return fmt.Errorf("%w: cannot inspect Darwin ACL on %s: %v", ErrUnsafe, path, errno)
	}
	return parseDarwinACL(buf[:])
}

func checkFileACL(f *os.File) error {
	attrs := darwinAttrList{BitmapCount: 5, Common: 0x00400000}
	var buf [darwinACLBufferSize]byte
	_, _, errno := syscall.Syscall6(syscall.SYS_FGETATTRLIST,
		f.Fd(), uintptr(unsafe.Pointer(&attrs)), uintptr(unsafe.Pointer(&buf[0])),
		uintptr(len(buf)), 4, 0)
	runtime.KeepAlive(f)
	if errno != 0 {
		return fmt.Errorf("%w: cannot inspect Darwin ACL on %s: %v", ErrUnsafe, f.Name(), errno)
	}
	return parseDarwinACL(buf[:])
}
