package workercontrol

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

func TestPureFoundationDependencies(t *testing.T) {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("locate source")
	}
	files, err := os.ReadDir(filepath.Dir(filename))
	if err != nil {
		t.Fatal(err)
	}
	allowed := map[string]bool{
		"bytes": true, "encoding/binary": true, "encoding/json": true,
		"errors": true, "fmt": true, "io": true, "unicode/utf8": true,
		"crypto/rand": true, "crypto/subtle": true, "crypto/sha256": true,
		"hash": true, "sync": true,
		"github.com/mrothroc/mixlab/internal/strictjson": true,
	}
	for _, file := range files {
		if !strings.HasSuffix(file.Name(), ".go") || strings.HasSuffix(file.Name(), "_test.go") {
			continue
		}
		parsed, err := parser.ParseFile(token.NewFileSet(), filepath.Join(filepath.Dir(filename), file.Name()), nil, parser.ImportsOnly)
		if err != nil {
			t.Fatal(err)
		}
		for _, spec := range parsed.Imports {
			name, err := strconv.Unquote(spec.Path.Value)
			if err != nil {
				t.Fatal(err)
			}
			peerOSImport := strings.HasPrefix(file.Name(), "peer") &&
				(name == "syscall" || (file.Name() == "peer_darwin.go" && name == "unsafe"))
			if !allowed[name] && !peerOSImport {
				t.Errorf("%s imports %s outside the pure foundation", file.Name(), name)
			}
		}
	}
}
