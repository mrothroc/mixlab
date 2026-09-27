package enrollment

import (
	"bytes"
	"encoding/json"
	"fmt"
	"time"
)

const maxInvitationBytes = 32 << 10

// EncodeInvitation produces secret-bearing bytes for protected file output,
// never log output. The caller must clear them after writing the file.
func EncodeInvitation(i Invitation, now time.Time) ([]byte, error) {
	if _, err := i.ValidateTarget(i.Endpoint, i.Audience, now); err != nil {
		return nil, err
	}
	b, err := json.Marshal(i)
	if err != nil || len(b) > maxInvitationBytes {
		clear(b)
		return nil, fmt.Errorf("invalid provisioning file encoding")
	}
	return b, nil
}

// DecodeInvitation requires canonical bounded bytes from protected storage and
// an exact target before a connection can send the bearer proof. It validates
// the pin, not the remote TLS peer: transport must authenticate that peer too.
func DecodeInvitation(raw []byte, endpoint, audience string, now time.Time) (Invitation, error) {
	var i Invitation
	if len(raw) > maxInvitationBytes {
		return i, fmt.Errorf("provisioning file too large")
	}
	if err := json.Unmarshal(raw, &i); err != nil {
		i.Clear()
		return Invitation{}, fmt.Errorf("malformed provisioning file")
	}
	again, err := json.Marshal(i)
	defer clear(again)
	if err != nil || !bytes.Equal(raw, again) {
		i.Clear()
		return Invitation{}, fmt.Errorf("noncanonical provisioning file")
	}
	if _, err := i.ValidateTarget(endpoint, audience, now); err != nil {
		i.Clear()
		return Invitation{}, err
	}
	return i, nil
}
