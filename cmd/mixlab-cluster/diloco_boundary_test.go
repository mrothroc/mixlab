package main

import (
	"strings"
	"testing"
)

func TestDiLoCoPureContractBoundaries(t *testing.T) {
	allowed := map[string]bool{}
	for _, p := range []string{"diloco/run", "diloco/learner", "diloco/internal/wire", "distributed", "precision", "artifact", "internal/strictjson"} {
		allowed[modulePath+p] = true
	}
	for _, target := range []string{"./diloco/run", "./diloco/learner", "./precision"} {
		for _, p := range dependencies(t, "0", "", target) {
			if !p.Standard && !allowed[p.ImportPath] {
				t.Errorf("%s contract depends on implementation %s", target, p.ImportPath)
			}
		}
	}
	for _, target := range []string{"./cmd/mixlab", "./train", "./gpu"} {
		for _, p := range dependencies(t, "0", "", target) {
			if p.ImportPath == modulePath+"diloco/authenticated" || p.ImportPath == modulePath+"coordinator" || strings.HasPrefix(p.ImportPath, modulePath+"coordinator/") || p.ImportPath == modulePath+"learner" || strings.HasPrefix(p.ImportPath, modulePath+"learner/") {
				// Only cmd/mixlab may compose the future learner application; neither
				// train nor gpu may import their caller. Pure diloco/learner is separate.
				if target == "./cmd/mixlab" && (p.ImportPath == modulePath+"learner" || strings.HasPrefix(p.ImportPath, modulePath+"learner/")) {
					continue
				}
				t.Errorf("%s imports forbidden context %s", target, p.ImportPath)
			}
		}
	}
}

// Apply these edges to future application packages as they land, without
// requiring placeholder application services in the P0 contract release.
func TestDiLoCoApplicationEdges(t *testing.T) {
	rules := map[string][]string{
		"train":       {"learner", "coordinator", "admission", "trust", "transport", "nodeagent", "recruitment"},
		"gpu":         {"learner", "coordinator", "admission", "trust", "transport", "nodeagent", "recruitment"},
		"learner":     {"coordinator", "admission", "trust", "transport", "nodeagent", "recruitment", "securekeys"},
		"coordinator": {"admission", "train", "gpu", "learner", "securekeys", "trust/authority", "trust/workload"},
		"admission":   {"train", "gpu", "learner"},
	}
	for _, p := range dependencies(t, "0", "", "./...") {
		for owner, forbidden := range rules {
			if p.ImportPath != modulePath+owner && !strings.HasPrefix(p.ImportPath, modulePath+owner+"/") {
				continue
			}
			for _, imp := range p.Imports {
				for _, bad := range forbidden {
					if imp == modulePath+bad || strings.HasPrefix(imp, modulePath+bad+"/") {
						t.Errorf("%s imports forbidden context %s", p.ImportPath, imp)
					}
				}
			}
		}
	}
}
