package main

import (
	"os"
	"strings"
	"testing"
)

// The pull-request image check runs docker/cloudbuild-pr.yaml, a copy of the
// main build's image steps without anything that publishes. A copy can drift,
// and a PR check that builds something other than what main publishes is a
// green check on the wrong build, so the build steps must stay identical.
func TestPullRequestImageBuildMatchesMainBuild(t *testing.T) {
	main := readBuildConfig(t, "cloudbuild-ci.yaml")
	pr := readBuildConfig(t, "cloudbuild-pr.yaml")

	mainSteps := section(t, "cloudbuild-ci.yaml", main, "\nsteps:\n", "\n  # Push to Docker Hub")
	prSteps := section(t, "cloudbuild-pr.yaml", pr, "\nsteps:\n", "\noptions:\n")
	if strings.TrimSpace(mainSteps) != strings.TrimSpace(prSteps) {
		t.Errorf("cloudbuild-pr.yaml steps differ from cloudbuild-ci.yaml's image build steps:\n--- ci\n%s\n--- pr\n%s", mainSteps, prSteps)
	}
	if !strings.Contains(prSteps, "docker/app.Dockerfile") || !strings.Contains(prSteps, "docker/runpod.Dockerfile") {
		t.Errorf("cloudbuild-pr.yaml must build both the CLI and RunPod images:\n%s", prSteps)
	}

	// A PR build runs code nobody has merged yet, so it must not publish an
	// image or read a secret.
	for _, forbidden := range []string{"\nimages:", "secretEnv", "availableSecrets", "secretManager", "docker push", "'push'", "_DOCKERHUB_USER"} {
		if strings.Contains(pr, forbidden) {
			t.Errorf("cloudbuild-pr.yaml must not publish or read secrets; found %q", forbidden)
		}
	}
	mainOptions := section(t, "cloudbuild-ci.yaml", main, "\noptions:\n", "\ntimeout:")
	prOptions := section(t, "cloudbuild-pr.yaml", pr, "\noptions:\n", "\ntimeout:")
	if mainOptions != prOptions {
		t.Errorf("cloudbuild-pr.yaml options differ from cloudbuild-ci.yaml:\n--- ci\n%s\n--- pr\n%s", mainOptions, prOptions)
	}
}

func readBuildConfig(t *testing.T, name string) string {
	t.Helper()
	data, err := os.ReadFile("../../docker/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func section(t *testing.T, name, text, start, end string) string {
	t.Helper()
	i := strings.Index(text, start)
	if i < 0 {
		t.Fatalf("%s has no %q", name, start)
	}
	rest := text[i+len(start):]
	j := strings.Index(rest, end)
	if j < 0 {
		t.Fatalf("%s has no %q after %q", name, end, start)
	}
	if strings.TrimSpace(rest[:j]) == "" {
		t.Fatalf("%s has an empty section after %q", name, start)
	}
	return rest[:j]
}
