package trust

import (
	"bytes"
	"fmt"
	"time"

	"github.com/mrothroc/mixlab/trust/internal/certificates"
)

// VerifySnapshotReceiver verifies only the destination of a public signed
// snapshot. It deliberately permits stale/revoked trust, but never expired
// credentials. The returned identity is not application authorization.
func VerifySnapshotReceiver(a Anchor, chain [][]byte, now time.Time) (string, error) {
	if len(chain) != 3 || !bytes.Equal(chain[2], a.DER()) {
		return "", fmt.Errorf("invalid snapshot receiver chain")
	}
	for _, der := range chain {
		if len(der) == 0 || len(der) > 16<<10 {
			return "", fmt.Errorf("invalid snapshot receiver certificate size")
		}
	}
	_, id, err := certificates.Verify(a, chain[0], chain[1], certificates.Principal, now)
	if err != nil {
		return "", err
	}
	if id.Role != Node {
		return "", fmt.Errorf("node snapshot receiver required")
	}
	return id.Principal, nil
}
