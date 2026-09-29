// Package clusterservice owns per-user OS service installation, not cluster trust.
package clusterservice

import (
	"bytes"
	"encoding/xml"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
)

type Spec struct {
	Role      string   `json:"role"`
	Binary    string   `json:"binary"`
	Arguments []string `json:"arguments"`
}

func (s Spec) Validate() error {
	if s.Role != "agent" && s.Role != "authority" {
		return fmt.Errorf("service role must be agent or authority")
	}
	if !filepath.IsAbs(s.Binary) || filepath.Clean(s.Binary) != s.Binary {
		return fmt.Errorf("canonical absolute service executable required")
	}
	for _, arg := range append([]string{s.Binary}, s.Arguments...) {
		if strings.ContainsAny(arg, "\x00\r\n") {
			return fmt.Errorf("control characters in service arguments")
		}
	}
	if len(s.Arguments) == 0 || s.Arguments[0] != s.Role {
		return fmt.Errorf("service arguments must start with its role")
	}
	args := s.Arguments[1:]
	state := "-agent-state-dir"
	allowed := map[string]bool{"-agent-state-dir": true, "-agent-listen": true, "-agent-advertise": true}
	if s.Role == "authority" {
		if len(args) == 0 || args[0] != "serve" {
			return fmt.Errorf("authority service must serve")
		}
		args = args[1:]
		state = "-cluster-state-dir"
		allowed = map[string]bool{"-cluster-state-dir": true, "-trust-listen": true, "-trust-advertise": true}
	}
	seen := map[string]bool{}
	for len(args) > 0 {
		if len(args) < 2 || !allowed[args[0]] || seen[args[0]] {
			return fmt.Errorf("invalid or duplicate service option")
		}
		seen[args[0]] = true
		if args[0] == state && !filepath.IsAbs(args[1]) {
			return fmt.Errorf("absolute service state required")
		}
		args = args[2:]
	}
	if !seen[state] {
		return fmt.Errorf("explicit service state required")
	}
	return nil
}

func Label(role string) string { return "com.mixlab.cluster-" + role }

// Render never invokes a shell. launchd XML and systemd ExecStart use distinct
// escaping rules, including systemd's specifier and environment expansion.
func Render(platform, dir string, s Spec) ([]byte, error) {
	if err := s.Validate(); err != nil {
		return nil, err
	}
	if !filepath.IsAbs(dir) || strings.ContainsAny(dir, "\x00\n\r") {
		return nil, fmt.Errorf("absolute service directory required")
	}
	args := []string{s.Binary, "service-run", "-service-dir", dir}
	switch platform {
	case "darwin":
		var b bytes.Buffer
		b.WriteString("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist version=\"1.0\"><dict><key>Label</key><string>")
		_ = xml.EscapeText(&b, []byte(Label(s.Role)))
		b.WriteString("</string><key>ProgramArguments</key><array>")
		for _, arg := range args {
			b.WriteString("<string>")
			_ = xml.EscapeText(&b, []byte(arg))
			b.WriteString("</string>")
		}
		b.WriteString("</array><key>RunAtLoad</key><true/><key>KeepAlive</key><true/><key>ThrottleInterval</key><integer>30</integer><key>ExitTimeOut</key><integer>120</integer><key>Umask</key><integer>63</integer><key>LimitLoadToSessionType</key><string>Aqua</string></dict></plist>\n")
		return b.Bytes(), nil
	case "linux":
		for i, arg := range args {
			args[i] = systemdQuote(arg)
		}
		return []byte("[Unit]\nDescription=Mixlab " + s.Role + " (user service)\nStartLimitIntervalSec=0\n\n[Service]\nExecStart=" + strings.Join(args, " ") + "\nRestart=always\nRestartSec=30\nTimeoutStopSec=120\nKillMode=mixed\nUMask=0077\n\n[Install]\nWantedBy=default.target\n"), nil
	default:
		return nil, fmt.Errorf("unsupported service platform %q", platform)
	}
}

func systemdQuote(s string) string {
	s = strings.ReplaceAll(s, "%", "%%")
	s = strings.ReplaceAll(s, "$", "$$")
	return strconv.Quote(s)
}
