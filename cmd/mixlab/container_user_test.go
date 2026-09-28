package main

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// The CLI image runs as an unprivileged user (since 2026-09-28). The RunPod
// image builds on it and stays root, because jobs run their own setup
// commands and write to the network volume. A host mount is then only
// writable when the documented `docker run` passes --user.
func TestContainerImagesRunAsTheirIntendedUsers(t *testing.T) {
	root := filepath.Join("..", "..")
	read := func(relative string) string {
		t.Helper()
		data, err := os.ReadFile(filepath.Join(root, relative))
		if err != nil {
			t.Fatalf("read %s: %v", relative, err)
		}
		return string(data)
	}

	app := read(filepath.Join("docker", "app.Dockerfile"))
	final := app[strings.LastIndex(app, "\nFROM "):]
	if !strings.Contains(final, " AS final") {
		t.Fatal("docker/app.Dockerfile: the last stage is no longer the final image")
	}
	users := regexp.MustCompile(`(?m)^USER (\S+)`).FindAllStringSubmatch(final, -1)
	if len(users) == 0 || users[len(users)-1][1] != "10001:10001" {
		t.Error("docker/app.Dockerfile: the final image must end as USER 10001:10001")
	}
	if !strings.Contains(final, "chown 10001:10001 /data") {
		t.Error("docker/app.Dockerfile: /data must be writable by the image user")
	}

	runpod := read(filepath.Join("docker", "runpod.Dockerfile"))
	from := strings.Index(runpod, "\nFROM ")
	if first := regexp.MustCompile(`(?m)^USER (\S+)`).FindStringSubmatch(runpod[from:]); first == nil || first[1] != "root" {
		t.Error("docker/runpod.Dockerfile must switch to USER root before installing packages")
	}

	mount := regexp.MustCompile(`docker run [^\n]*-v [^\n]*:/data`)
	for _, relative := range []string{"README.md", filepath.Join("docker", "DOCKERHUB.md"), filepath.Join("docker", "README.md")} {
		for _, line := range mount.FindAllString(read(relative), -1) {
			if !strings.Contains(line, "--user") {
				t.Errorf("%s: %q mounts /data without --user; the image user cannot write a host directory", relative, line)
			}
		}
	}
}
