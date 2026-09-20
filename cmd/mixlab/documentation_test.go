package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// flagRegistrars are the package-level flag constructors whose first string
// argument (second for the Var forms) is the flag name.
var flagRegistrars = map[string]bool{
	"Bool": true, "Duration": true, "Float64": true, "Int": true, "Int64": true,
	"String": true, "Uint": true, "Uint64": true, "Func": true, "TextVar": true,
	"BoolVar": true, "DurationVar": true, "Float64Var": true, "IntVar": true,
	"Int64Var": true, "StringVar": true, "UintVar": true, "Uint64Var": true, "Var": true,
}

// declaredCLIFlagNames parses the command's source for flag registrations.
//
// The flags are locals inside main(), so they are never installed on
// flag.CommandLine during a test binary's run. An earlier version of this guard
// walked flag.CommandLine and therefore inspected nothing at all, passing no
// matter what was undocumented. Reading the source keeps the check honest
// without reshaping main() purely to satisfy a test.
func declaredCLIFlagNames(t *testing.T, dir string) []string {
	t.Helper()
	sources, err := filepath.Glob(filepath.Join(dir, "*.go"))
	if err != nil {
		t.Fatalf("glob %s: %v", dir, err)
	}
	fset := token.NewFileSet()
	var names []string
	for _, path := range sources {
		if strings.HasSuffix(path, "_test.go") {
			continue
		}
		file, err := parser.ParseFile(fset, path, nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", path, err)
		}
		ast.Inspect(file, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			sel, ok := call.Fun.(*ast.SelectorExpr)
			if !ok || !flagRegistrars[sel.Sel.Name] {
				return true
			}
			pkgIdent, ok := sel.X.(*ast.Ident)
			if !ok || pkgIdent.Name != "flag" {
				return true
			}
			arg := 0
			if strings.HasSuffix(sel.Sel.Name, "Var") {
				arg = 1
			}
			if len(call.Args) <= arg {
				return true
			}
			lit, ok := call.Args[arg].(*ast.BasicLit)
			if !ok || lit.Kind != token.STRING {
				t.Errorf("flag.%s at %s uses a non-literal name; the documentation guard cannot check it",
					sel.Sel.Name, fset.Position(call.Pos()))
				return true
			}
			name, err := strconv.Unquote(lit.Value)
			if err != nil {
				t.Fatalf("unquote flag name %s: %v", lit.Value, err)
			}
			names = append(names, name)
			return true
		})
	}
	sort.Strings(names)
	// A parse that finds nothing would silently pass every flag, which is the
	// exact failure this guard exists to prevent.
	if len(names) == 0 {
		t.Fatalf("no flag registrations found in %s; the documentation guard is not inspecting anything", dir)
	}
	return names
}

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
	for _, name := range declaredCLIFlagNames(t, ".") {
		if !strings.Contains(allDocs, "-"+name) {
			missing = append(missing, name)
		}
	}
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

// Homebrew's user/tap/formula form requires a repository named homebrew-tap,
// so the formula is necessarily copied out of this repository. That copy once
// drifted for five months without failing: an archived repo stays publicly
// tappable, so the stale formula kept installing instead of erroring. These
// guard the two halves that made it invisible -- the documented install path,
// and a release step that refreshes the copy.
func TestHomebrewInstructionsUseThePublishedTap(t *testing.T) {
	root := filepath.Join("..", "..")
	readme, err := os.ReadFile(filepath.Join(root, "README.md"))
	if err != nil {
		t.Fatalf("read README.md: %v", err)
	}
	if !strings.Contains(string(readme), "brew install mrothroc/tap/mixlab") {
		t.Error("README.md no longer documents the tap users install from")
	}
	releasing, err := os.ReadFile(filepath.Join(root, "docs", "releasing.md"))
	if err != nil {
		t.Fatalf("read docs/releasing.md: %v", err)
	}
	for _, required := range []string{
		"repos/mrothroc/homebrew-tap/contents/Formula/mixlab.rb",
		"brew upgrade mrothroc/tap/mixlab",
	} {
		if !strings.Contains(string(releasing), required) {
			t.Errorf("docs/releasing.md no longer covers %q, so the tap copy can go stale unnoticed", required)
		}
	}
}
