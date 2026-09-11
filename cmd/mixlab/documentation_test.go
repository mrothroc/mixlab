package main

import (
	"flag"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

func TestPublicCLIFlagsAreDocumented(t *testing.T) {
	root := filepath.Join("..", "..")
	paths := []string{
		filepath.Join(root, "README.md"),
		filepath.Join(root, "examples", "README.md"),
	}
	docPaths, err := filepath.Glob(filepath.Join(root, "docs", "*.md"))
	if err != nil {
		t.Fatal(err)
	}
	paths = append(paths, docPaths...)

	var corpus strings.Builder
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v", path, err)
		}
		corpus.Write(data)
		corpus.WriteByte('\n')
	}
	allDocs := corpus.String()

	var missing []string
	flag.CommandLine.VisitAll(func(f *flag.Flag) {
		if strings.HasPrefix(f.Name, "test.") {
			return
		}
		if !strings.Contains(allDocs, "-"+f.Name) {
			missing = append(missing, f.Name)
		}
	})
	sort.Strings(missing)
	if len(missing) > 0 {
		t.Fatalf("public CLI flags missing from README/docs:\n  -%s",
			strings.Join(missing, "\n  -"))
	}
}

func TestPublishedDocumentationIndexesHaveValidLocalLinks(t *testing.T) {
	root := filepath.Join("..", "..")
	for _, relative := range []string{
		"README.md",
		"llms.txt",
		"docs/README.md",
		"docs/feature-matrix.md",
		"examples/README.md",
	} {
		path := filepath.Join(root, relative)
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v", path, err)
		}
		for _, match := range markdownLinkPattern.FindAllStringSubmatch(string(data), -1) {
			target := strings.TrimSpace(match[1])
			if target == "" || strings.HasPrefix(target, "#") ||
				strings.HasPrefix(target, "http://") || strings.HasPrefix(target, "https://") ||
				strings.HasPrefix(target, "mailto:") {
				continue
			}
			target = strings.Split(target, "#")[0]
			target = strings.Trim(target, "<>")
			resolved := filepath.Join(filepath.Dir(path), filepath.FromSlash(target))
			if _, err := os.Stat(resolved); err != nil {
				t.Errorf("%s links to missing local target %q: %v", relative, target, err)
			}
		}
	}
}

var markdownLinkPattern = regexp.MustCompile(`\[[^\]]+\]\(([^)]+)\)`)

// The formula is published from this repository, so the tap must be installed
// by URL. The shorter mrothroc/tap/mixlab form resolves to a separate
// homebrew-tap repository that is archived and therefore frozen at an old
// release; it stayed publicly tappable, so pointing users at it served a stale
// binary silently rather than failing.
func TestHomebrewInstructionsUseThePublishedTap(t *testing.T) {
	root := filepath.Join("..", "..")
	for _, relative := range []string{"README.md", filepath.Join("docs", "releasing.md")} {
		path := filepath.Join(root, relative)
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v", relative, err)
		}
		text := string(data)
		if strings.Contains(text, "mrothroc/tap/mixlab") {
			t.Errorf("%s installs from the archived homebrew-tap repository", relative)
		}
		if !strings.Contains(text, "brew tap mrothroc/mixlab https://github.com/mrothroc/mixlab") {
			t.Errorf("%s does not tap the repository that carries Formula/mixlab.rb", relative)
		}
	}
}
