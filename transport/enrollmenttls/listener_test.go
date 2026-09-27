package enrollmenttls

import (
	"net"
	"testing"
)

func TestEnrollmentLocalAddressRestriction(t *testing.T) {
	l, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = l.Close() }()
	ip, err := address(l.Addr())
	if err != nil {
		t.Fatal(err)
	}
	iface, err := localAddressInterface(ip)
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateListener(l, iface); err != nil {
		t.Fatal(err)
	}
	if err := ValidateListener(l, "not-the-local-interface"); err == nil {
		t.Fatal("accepted interface/address mismatch")
	}
	wildcard, err := net.Listen("tcp4", "0.0.0.0:0")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = wildcard.Close() }()
	if err := ValidateListener(wildcard, ""); err == nil {
		t.Fatal("accepted wildcard enrollment listener")
	}
}
