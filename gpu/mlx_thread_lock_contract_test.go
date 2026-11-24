package gpu

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"testing"
)

// MLX streams are per OS thread. lockMLXThread pins only the calling
// goroutine, and every t.Run body runs on a new goroutine, so a subtest of a
// pinned test must pin itself or a migration between cgo calls loses its
// streams ("There is no Stream(gpu, N) in current thread").
func TestMLXSubtestsLockTheirOwnThread(t *testing.T) {
	files, err := filepath.Glob("*_test.go")
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	checked := 0
	for _, name := range files {
		file, err := parser.ParseFile(fset, name, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body == nil || !callsLockMLXThread(fn.Body) {
				continue
			}
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				body := subtestBody(n)
				if body == nil {
					return true
				}
				checked++
				if !callsLockMLXThread(body) {
					t.Errorf("%s: subtest of %s does not call lockMLXThread(t)", fset.Position(body.Pos()), fn.Name.Name)
				}
				return true
			})
		}
	}
	if checked == 0 {
		t.Fatal("found no subtests of thread-locked tests; the scan is broken")
	}
}

// subtestBody returns the function literal body passed to t.Run, if n is such a call.
func subtestBody(n ast.Node) *ast.BlockStmt {
	call, ok := n.(*ast.CallExpr)
	if !ok || len(call.Args) != 2 {
		return nil
	}
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok || sel.Sel.Name != "Run" {
		return nil
	}
	lit, ok := call.Args[1].(*ast.FuncLit)
	if !ok {
		return nil
	}
	return lit.Body
}

// callsLockMLXThread reports whether body calls lockMLXThread directly, not
// inside a nested function literal.
func callsLockMLXThread(body *ast.BlockStmt) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		if found {
			return false
		}
		if _, ok := n.(*ast.FuncLit); ok {
			return false
		}
		if call, ok := n.(*ast.CallExpr); ok {
			if id, ok := call.Fun.(*ast.Ident); ok && id.Name == "lockMLXThread" {
				found = true
			}
		}
		return true
	})
	return found
}
