package enrollment

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/statehome"
)

// WriteInvitation never replaces another provisioning file. The owning caller
// selects a private directory; statehome enforces ownership, modes, ACLs, and
// no-link semantics. No file secret is returned by an error or printed here.
func WriteInvitation(dir statehome.Path, name string, i Invitation, now time.Time) error {
	b, err := EncodeInvitation(i, now)
	if err != nil {
		return err
	}
	defer clear(b)
	return dir.CompareAndSwap(name, nil, b)
}

func ReadInvitation(dir statehome.Path, name string, now time.Time) (Invitation, error) {
	b, err := dir.ReadFileLimit(name, maxInvitationBytes)
	if err != nil {
		return Invitation{}, err
	}
	defer clear(b)
	var i Invitation
	// Decode once to get the exact file-owned endpoint, then canonical and pin
	// validation still applies. Routing hints never supply a substitute target.
	if err := json.Unmarshal(b, &i); err != nil {
		i.Clear()
		return Invitation{}, err
	}
	defer i.Clear()
	return DecodeInvitation(b, i.Endpoint, i.Audience, now)
}

// DestroyInvitation removes only the exact consumed file. An unrelated file
// written at the same name is not removed on a stale cleanup attempt.
func DestroyInvitation(dir statehome.Path, name string, i Invitation) error {
	b, err := json.Marshal(i)
	if err != nil {
		return err
	}
	defer clear(b)
	if len(i.Secret) != 32 {
		return fmt.Errorf("original provisioning secret required for exact cleanup")
	}
	return dir.CompareAndSwap(name, b, nil)
}
