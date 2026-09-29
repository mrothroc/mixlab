package clusterservice

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/mrothroc/mixlab/statehome"
)

type Runner func(context.Context, string, ...string) ([]byte, error)

type Manager struct {
	Platform, Home string
	UID            int
	Run            Runner
}

func (m Manager) Paths(role string) (statehome.Path, string, error) {
	if role != "agent" && role != "authority" {
		return statehome.Path{}, "", fmt.Errorf("invalid service role")
	}
	p, err := statehome.Resolve(statehome.Options{ExactDir: filepath.Join(m.Home, ".mixlab", "services", role)}, statehome.Context{Kind: statehome.Agent})
	unit := ""
	switch m.Platform {
	case "darwin":
		unit = filepath.Join(m.Home, "Library", "LaunchAgents", Label(role)+".plist")
	case "linux":
		unit = filepath.Join(m.Home, ".config", "systemd", "user", Label(role)+".service")
	default:
		return p, "", fmt.Errorf("unsupported service platform")
	}
	return p, unit, err
}

func (m Manager) command(ctx context.Context, action, role, unit string) ([]byte, error) {
	if m.Run == nil {
		return nil, fmt.Errorf("service runner required")
	}
	ctx, cancel := context.WithTimeout(ctx, 130*time.Second)
	defer cancel()
	if m.Platform == "darwin" {
		domain := "gui/" + strconv.Itoa(m.UID)
		switch action {
		case "start":
			if _, err := m.Run(ctx, "/bin/launchctl", "print", domain+"/"+Label(role)); err == nil {
				return nil, nil
			}
			return m.Run(ctx, "/bin/launchctl", "bootstrap", domain, unit)
		case "stop":
			b, err := m.Run(ctx, "/bin/launchctl", "bootout", domain+"/"+Label(role))
			if err != nil && (strings.Contains(string(b), "No such process") || strings.Contains(string(b), "Could not find service")) {
				return nil, nil
			}
			return b, err
		case "status":
			return m.Run(ctx, "/bin/launchctl", "print", domain+"/"+Label(role))
		}
	} else {
		name := Label(role) + ".service"
		switch action {
		case "start":
			if b, err := m.Run(ctx, "systemctl", "--user", "daemon-reload"); err != nil {
				return b, err
			}
			return m.Run(ctx, "systemctl", "--user", "enable", "--now", name)
		case "stop":
			// Stop independently of the unit file: a prior interrupted uninstall
			// may have removed it while the manager still has a loaded job.
			b, err := m.Run(ctx, "systemctl", "--user", "stop", name)
			if err != nil && strings.Contains(string(b), "Unit "+name+" not loaded") {
				return nil, nil
			}
			return b, err
		case "disable":
			return m.Run(ctx, "systemctl", "--user", "disable", name)
		case "reload":
			return m.Run(ctx, "systemctl", "--user", "daemon-reload")
		case "status":
			return m.Run(ctx, "systemctl", "--user", "status", "--no-pager", name)
		}
	}
	return nil, fmt.Errorf("unsupported service action")
}

// Install persists an exact specification before loading the service. Failed
// bootstrap leaves it inspectable/retryable; it never resets cluster state.
func (m Manager) Install(ctx context.Context, s Spec) error {
	if err := s.Validate(); err != nil {
		return err
	}
	p, unit, err := m.Paths(s.Role)
	if err != nil {
		return err
	}
	if err := p.Ensure(); err != nil {
		return err
	}
	return p.WithProcessLock(ctx, "install.lock", func() error {
		b, err := json.Marshal(s)
		if err != nil {
			return err
		}
		old, err := p.ReadFileLimit("service.json", 16384)
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		if err == nil && !bytes.Equal(old, b) {
			return fmt.Errorf("different service already installed; uninstall it before changing settings")
		}
		if err != nil {
			if err := p.CompareAndSwap("service.json", nil, b); err != nil {
				return err
			}
		}
		data, err := Render(m.Platform, p.Dir(), s)
		if err != nil {
			return err
		}
		if err := publishUnit(unit, data); err != nil {
			return err
		}
		if _, err := m.command(ctx, "status", s.Role, unit); err == nil {
			return nil
		}
		b, err = m.command(ctx, "start", s.Role, unit)
		if err != nil {
			return fmt.Errorf("service saved but could not start: %w: %s", err, b)
		}
		return nil
	})
}

func Read(p statehome.Path) (Spec, error) {
	var s Spec
	b, err := p.ReadFileLimit("service.json", 16384)
	if err != nil {
		return s, err
	}
	if err := json.Unmarshal(b, &s); err != nil {
		return s, err
	}
	return s, s.Validate()
}

func (m Manager) Control(ctx context.Context, role, action string) ([]byte, error) {
	p, unit, err := m.Paths(role)
	if err != nil {
		return nil, err
	}
	if action == "status" {
		s, err := Read(p)
		if err != nil {
			return nil, err
		}
		if s.Role != role {
			return nil, fmt.Errorf("service role changed")
		}
		return m.command(ctx, action, role, unit)
	}
	var out []byte
	err = p.WithProcessLock(ctx, "install.lock", func() error {
		// Read under the same lock as install/uninstall, never act on a
		// registration that another administrator replaced while we waited.
		s, err := Read(p)
		if err != nil {
			return err
		}
		if s.Role != role {
			return fmt.Errorf("service role changed")
		}
		data, err := Render(m.Platform, p.Dir(), s)
		if err != nil {
			return err
		}
		unitErr := checkUnit(unit, data)
		if unitErr != nil && (action != "uninstall" || !errors.Is(unitErr, os.ErrNotExist)) {
			return unitErr
		}
		if action != "uninstall" {
			out, err = m.command(ctx, action, role, unit)
			return err
		}
		// Never delete a loaded definition if stopping it failed. The caller
		// must resolve an unavailable service manager instead of guessing.
		out, err = m.command(ctx, "stop", role, unit)
		if err != nil {
			return err
		}
		if m.Platform == "linux" {
			// systemd needs the [Install] metadata to remove enablement links.
			// Restore only our exact missing definition, never overwrite another
			// unit or guess which symlinks belong to the user's service manager.
			if errors.Is(unitErr, os.ErrNotExist) {
				if err := publishUnit(unit, data); err != nil {
					return err
				}
			}
			out, err = m.command(ctx, "disable", role, unit)
			if err != nil {
				return err
			}
		}
		if err := os.Remove(unit); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		if m.Platform == "linux" {
			// disable reloads before deletion; reload again so the removed
			// definition cannot remain cached. Keep service.json on failure so
			// uninstall can restore metadata and retry the complete cleanup.
			out, err = m.command(ctx, "reload", role, unit)
			if err != nil {
				return err
			}
		}
		return os.Remove(filepath.Join(p.Dir(), "service.json"))
	})
	return out, err
}
