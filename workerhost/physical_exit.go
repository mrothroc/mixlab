package workerhost

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"

	"github.com/mrothroc/mixlab/statehome"
	"github.com/mrothroc/mixlab/workerhost/contract"
)

const physicalExitFile = "physical-exit.json"

type physicalClaim struct {
	Approval string `json:"approval"`
	Boot     string `json:"boot"`
}

type physicalExit struct {
	Claim physicalClaim        `json:"claim"`
	PID   int                  `json:"pid"`
	Exit  contract.ProcessExit `json:"exit,omitzero"`
}

// Receipts are immutable and published only after owned-process cleanup. They
// can be read without taking the physical lock, even if its owner died before
// publishing the higher-level guardian result. Omitting a zero Exit preserves
// canonical JSON for receipts written before OS diagnostics were available.
func loadPhysicalExit(dir statehome.Path) (physicalExit, error) {
	b, err := dir.ReadFileLimit(physicalExitFile, 2048)
	if err != nil {
		return physicalExit{}, err
	}
	var exit physicalExit
	if err := json.Unmarshal(b, &exit); err != nil {
		return exit, err
	}
	canonical, _ := json.Marshal(exit)
	if !bytes.Equal(b, canonical) || !contract.Hash(exit.Claim.Approval) || !validBootIdentity(exit.Claim.Boot) || exit.PID < 0 {
		return exit, fmt.Errorf("physical exit evidence changed")
	}
	if err := exit.Exit.Validate(); err != nil {
		return exit, err
	}
	if (exit.Exit.Kind == contract.ProcessNotStarted && exit.PID != 0) || ((exit.Exit.Kind == contract.ProcessExited || exit.Exit.Kind == contract.ProcessSignaled) && exit.PID == 0) {
		return exit, fmt.Errorf("physical exit evidence conflicts with child PID")
	}
	return exit, nil
}

func readPhysicalExit(dir statehome.Path, claim physicalClaim) (physicalExit, error) {
	exit, err := loadPhysicalExit(dir)
	if err == nil && exit.Claim != claim {
		err = fmt.Errorf("physical exit evidence changed")
	}
	return exit, err
}

// Read the durable receipt directly, not through an optional Runner interface:
// production wrappers may embed only Runner, hiding additional methods.
func (s *AttemptStore) attachProcessExit(r Attempt, out *contract.Outcome) error {
	exit, err := loadPhysicalExit(s.path)
	if errors.Is(err, os.ErrNotExist) {
		// Legacy/reboot reconciliation and nonphysical runners may have no
		// receipt. NoChild does not imply a known OS wait status or no spawn.
		return nil
	}
	if err != nil {
		return err
	}
	if exit.Claim.Approval != r.Digest || (r.Started != nil && exit.PID != r.Started.PID) {
		return fmt.Errorf("physical exit differs from approved attempt or published start")
	}
	out.Exit = exit.Exit
	// Physical spawn can precede a failed Started publication. Preserve its PID
	// without manufacturing Started history or advancing the outcome version.
	if r.Started == nil {
		out.PID = exit.PID
	}
	return nil
}

func validBootIdentity(s string) bool {
	if len(s) != 36 {
		return false
	}
	for i, c := range s {
		if i == 8 || i == 13 || i == 18 || i == 23 {
			if c != '-' {
				return false
			}
		} else if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}
