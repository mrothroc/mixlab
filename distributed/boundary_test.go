package distributed

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

func TestDistributedBoundaryDependencies(t *testing.T) {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("locate distributed boundary test")
	}
	repoRoot := filepath.Dir(filepath.Dir(filename))
	forbiddenConsumers := []string{
		"discovery",
		"node-agent",
		"node_agent",
		"nodeagent",
		"sampler",
	}
	err := filepath.WalkDir(repoRoot, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			if entry.Name() == ".git" || entry.Name() == "vendor" {
				return filepath.SkipDir
			}
			return nil
		}
		if filepath.Ext(path) != ".go" {
			return nil
		}
		rel, err := filepath.Rel(repoRoot, path)
		if err != nil {
			return err
		}
		pkgPath := filepath.ToSlash(filepath.Dir(rel))
		if pkgPath != "distributed" && !containsPathComponent(pkgPath, forbiddenConsumers) {
			return nil
		}
		file, err := parser.ParseFile(token.NewFileSet(), path, nil, parser.ImportsOnly)
		if err != nil {
			return err
		}
		for _, spec := range file.Imports {
			importPath, err := strconv.Unquote(spec.Path.Value)
			if err != nil {
				return err
			}
			if pkgPath == "distributed" &&
				(importPath == "github.com/mrothroc/mixlab/gpu" ||
					importPath == "github.com/mrothroc/mixlab/train") {
				t.Errorf("%s crosses the identity-kernel boundary by importing %s", rel, importPath)
			}
			if containsPathComponent(pkgPath, forbiddenConsumers) &&
				(importPath == "github.com/mrothroc/mixlab/gpu" ||
					importPath == "github.com/mrothroc/mixlab/train") {
				t.Errorf("%s crosses the runtime/trainer boundary by importing %s", rel, importPath)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("scan distributed boundaries: %v", err)
	}
}

func containsPathComponent(path string, candidates []string) bool {
	for _, component := range strings.Split(path, "/") {
		for _, candidate := range candidates {
			if component == candidate {
				return true
			}
		}
	}
	return false
}
