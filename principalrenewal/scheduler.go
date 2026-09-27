// Package principalrenewal owns long-lived process renewal triggers, not
// certificate eligibility, issuance, key access, or transport. All identities
// and renewals come through the owning composition's trust-authority ports.
package principalrenewal

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/mrothroc/mixlab/statehome"
)

var ErrReenrollmentRequired = errors.New("principal expired; explicit approved reenrollment required")

const filename = "renewal-schedule.json"
const version = "mixlab_principal_renewal_schedule_v1"

type Identity struct {
	Principal, Certificate string
	Issued, Expires        time.Time
}
type Ports struct {
	Identity func(time.Time) (Identity, error)
	Renew    func(context.Context, time.Time) error
}
type Scheduler struct {
	path  statehome.Path
	ports Ports
}
type State struct {
	Version     string `json:"version"`
	Principal   string `json:"principal"`
	Certificate string `json:"certificate"`
	Expires     int64  `json:"expires"`
	Next        int64  `json:"next"`
	LastAttempt int64  `json:"last_attempt"`
	Outcome     string `json:"outcome"`
	Failures    uint32 `json:"failures"`
}

func New(path statehome.Path, ports Ports) (*Scheduler, error) {
	if path.Kind() != statehome.Principal || ports.Identity == nil || ports.Renew == nil {
		return nil, fmt.Errorf("principal context and renewal ports required")
	}
	if err := path.Validate(); err != nil {
		return nil, err
	}
	return &Scheduler{path, ports}, nil
}

func validID(s string, n int) bool {
	b, e := hex.DecodeString(s)
	return e == nil && len(b) == n && hex.EncodeToString(b) == s
}
func (i Identity) validate(now time.Time) error {
	if !validID(i.Principal, 16) || !validID(i.Certificate, 32) || !i.Expires.After(i.Issued) || now.Before(i.Issued) {
		return fmt.Errorf("invalid renewal identity")
	}
	return nil
}

// Due renews before the final third begins: subtract clock skew allowance and
// deterministic principal-derived jitter, so neither delays renewal past the
// policy threshold. No process RNG, wall-clock sampling, or network is used.
func Due(i Identity) time.Time {
	h := sha256.Sum256([]byte("mixlab-principal-renewal-v1:" + i.Principal))
	jitter := time.Duration(binary.BigEndian.Uint64(h[:8])%301) * time.Second
	due := i.Issued.Add(i.Expires.Sub(i.Issued)*2/3 - 5*time.Minute - jitter)
	if due.Before(i.Issued) {
		return i.Issued
	}
	return due
}

func fresh(i Identity) State {
	return State{Version: version, Principal: i.Principal, Certificate: i.Certificate, Expires: i.Expires.Unix(), Next: Due(i).Unix(), Outcome: "scheduled"}
}
func (s *Scheduler) load(i Identity) ([]byte, State, error) {
	b, err := s.path.ReadFileLimit(filename, 4096)
	if errors.Is(err, os.ErrNotExist) {
		return nil, fresh(i), nil
	}
	if err != nil {
		return nil, State{}, err
	}
	var r State
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, r, err
	}
	again, err := json.Marshal(r)
	if err != nil || !bytes.Equal(b, again) || r.Version != version || r.Principal != i.Principal || !validID(r.Certificate, 32) || r.Next <= 0 || r.Expires <= 0 || r.Next > r.Expires || r.Failures > 32 {
		return nil, r, fmt.Errorf("invalid renewal schedule")
	}
	switch r.Outcome {
	case "scheduled", "attempting", "failed", "renewed":
	default:
		return nil, r, fmt.Errorf("invalid renewal outcome")
	}
	if r.Certificate != i.Certificate {
		return b, fresh(i), nil
	}
	if r.Expires != i.Expires.Unix() {
		return nil, r, fmt.Errorf("renewal expiry changed without certificate change")
	}
	return b, r, nil
}
func (s *Scheduler) save(old []byte, r State) ([]byte, error) {
	b, err := json.Marshal(r)
	if err != nil {
		return nil, err
	}
	return b, s.path.CompareAndSwap(filename, old, b)
}

// Tick commits an attempt and its bounded retry time before invoking the
// authority. Callers log returned failures while respecting State.Next. An
// expired identity fails closed without an anonymous recovery attempt.
func (s *Scheduler) Tick(ctx context.Context, now time.Time) (out State, result error) {
	result = s.path.WithProcessLock(ctx, "renewal-schedule.lock", func() error {
		i, err := s.ports.Identity(now)
		if err != nil {
			return err
		}
		if err := i.validate(now); err != nil {
			return err
		}
		if !now.Before(i.Expires) {
			return ErrReenrollmentRequired
		}
		old, r, err := s.load(i)
		if err != nil {
			return err
		}
		out = r
		if now.Unix() < r.Next {
			_, err = s.save(old, r)
			return err
		}
		if r.Failures < 32 {
			r.Failures++
		}
		exponent := min(r.Failures-1, 6)
		delay := min(5*time.Second*time.Duration(uint64(1)<<exponent), 5*time.Minute)
		r.LastAttempt = now.Unix()
		r.Next = min(now.Add(delay).Unix(), i.Expires.Unix())
		r.Outcome = "attempting"
		old, err = s.save(old, r)
		if err != nil {
			return err
		}
		attempt, done := context.WithTimeout(ctx, 20*time.Second)
		defer done()
		renewErr := s.ports.Renew(attempt, now)
		if renewErr == nil {
			next, err := s.ports.Identity(now)
			switch {
			case err != nil:
				renewErr = err
			case next.validate(now) != nil || next.Principal != i.Principal || next.Certificate == i.Certificate || !next.Expires.After(i.Expires):
				renewErr = fmt.Errorf("authority did not install a renewed principal")
			default:
				r = fresh(next)
				r.LastAttempt = now.Unix()
				r.Outcome = "renewed"
			}
		}
		if renewErr != nil {
			r.Outcome = "failed"
		}
		out = r
		_, saveErr := s.save(old, r)
		return errors.Join(renewErr, saveErr)
	})
	return out, result
}
