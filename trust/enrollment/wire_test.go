package enrollment

import (
	"bytes"
	"testing"
)

func TestProtectedInvitationWire(t *testing.T) {
	f := setup(t)
	i := f.invite(t, NodeEnrollment)
	defer i.Clear()
	raw, err := EncodeInvitation(i, now)
	check(t, err)
	defer clear(raw)
	copy, err := DecodeInvitation(raw, i.Endpoint, i.Audience, now)
	check(t, err)
	if !bytes.Equal(copy.Secret, i.Secret) || copy.ID != i.ID {
		t.Fatal("provisioning file changed")
	}
	copy.Clear()
	if len(i.Secret) != 32 || bytes.Equal(i.Secret, make([]byte, 32)) {
		t.Fatal("decoded secret aliases source")
	}
	for _, bad := range [][]byte{
		append(bytes.Clone(raw), '\n'),
		append(bytes.Clone(raw[:len(raw)-1]), []byte(`,"use_limit":1}`)...),
		append(bytes.Clone(raw[:len(raw)-1]), []byte(`,"unknown":0}`)...),
		[]byte(`{"secret":"broken`),
		make([]byte, maxInvitationBytes+1),
	} {
		if got, err := DecodeInvitation(bad, i.Endpoint, i.Audience, now); err == nil || len(got.Secret) != 0 {
			t.Fatal("invalid file returned secret")
		}
		clear(bad)
	}
}
