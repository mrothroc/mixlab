//go:build darwin || linux

package workerhost

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// Sampling shares the reap/signal fence: an old PGID must never be sampled after
// its leader is reaped and that number could belong to an unrelated process.
func (p *ownedProcess) sampleUsage(ctx context.Context) ([]processUsage, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.reaped {
		return nil, nil
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	cmd := exec.CommandContext(ctx, "/bin/ps", "-axo", "pid=,pgid=,time=,rss=,lstart=")
	cmd.Env = []string{"PATH=/usr/bin:/bin", "LC_ALL=C", "TZ=UTC"}
	cmd.WaitDelay = 100 * time.Millisecond
	out := &sampleBuffer{}
	cmd.Stdout, cmd.Stderr = out, out
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("process resource sample failed: %w", err)
	}
	return parseProcessUsage(out.String(), p.cmd.Process.Pid)
}

type sampleBuffer struct{ buffer bytes.Buffer }

func (b *sampleBuffer) String() string { return b.buffer.String() }

func (b *sampleBuffer) Write(p []byte) (int, error) {
	if len(p) > (8<<20)-b.buffer.Len() {
		return 0, fmt.Errorf("process resource sample exceeds 8 MiB")
	}
	return b.buffer.Write(p)
}

func parseProcessUsage(output string, group int) ([]processUsage, error) {
	var found []processUsage
	lines := strings.Split(strings.TrimSpace(output), "\n")
	if len(lines) > resourceEntryLimit {
		return nil, fmt.Errorf("process resource sample has too many entries")
	}
	for _, line := range lines {
		f := strings.Fields(line)
		if len(f) != 9 {
			return nil, fmt.Errorf("malformed process resource sample")
		}
		pgid, err := strconv.Atoi(f[1])
		if err != nil {
			return nil, fmt.Errorf("invalid process group in resource sample")
		}
		if pgid != group {
			continue
		}
		pid, err := strconv.Atoi(f[0])
		if err != nil || pid < 1 {
			return nil, fmt.Errorf("invalid process ID in resource sample")
		}
		cpu, err := parseCPUTime(f[2])
		if err != nil {
			return nil, err
		}
		rss, err := strconv.ParseInt(f[3], 10, 64)
		if err != nil || rss < 0 || rss > (1<<63-1)/1024 {
			return nil, fmt.Errorf("invalid RSS in process sample")
		}
		start := strings.Join(f[4:], " ")
		if _, err := time.Parse("Mon Jan 2 15:04:05 2006", start); err != nil {
			return nil, fmt.Errorf("invalid process start time: %w", err)
		}
		found = append(found, processUsage{Identity: f[0] + "/" + start, CPU: cpu, RSS: rss * 1024})
	}
	return found, nil
}

// BSD ps uses minutes:seconds.fraction; procps also uses [days-]hours:minutes:seconds.
func parseCPUTime(s string) (time.Duration, error) {
	var total time.Duration
	if i := strings.IndexByte(s, '-'); i >= 0 {
		days, err := strconv.ParseUint(s[:i], 10, 32)
		if err != nil || days > 100000 {
			return 0, fmt.Errorf("invalid CPU days")
		}
		total = time.Duration(days) * 24 * time.Hour
		s = s[i+1:]
	}
	parts := strings.Split(s, ":")
	if len(parts) < 2 || len(parts) > 3 {
		return 0, fmt.Errorf("invalid CPU time")
	}
	for i, part := range parts {
		unit := time.Second
		if i < len(parts)-1 {
			unit = time.Minute
			if len(parts) == 3 && i == 0 {
				unit = time.Hour
			}
		}
		for _, c := range part {
			if (c < '0' || c > '9') && (c != '.' || i != len(parts)-1) {
				return 0, fmt.Errorf("invalid CPU time component")
			}
		}
		value, err := time.ParseDuration(part + "s")
		if err != nil || value < 0 || (i > 0 && value >= time.Minute) {
			return 0, fmt.Errorf("invalid CPU time component")
		}
		mult := unit / time.Second
		if value > (time.Duration(1<<63-1)-total)/mult {
			return 0, fmt.Errorf("CPU time overflow")
		}
		total += value * mult
	}
	return total, nil
}
