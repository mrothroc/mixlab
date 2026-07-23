package arch

import (
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

func TestPublicConfigFieldsAreDocumented(t *testing.T) {
	referencePath := filepath.Join("..", "docs", "config-reference.md")
	data, err := os.ReadFile(referencePath)
	if err != nil {
		t.Fatalf("read %s: %v", referencePath, err)
	}
	reference := string(data)

	types := []reflect.Type{
		reflect.TypeOf(ArchConfig{}),
		reflect.TypeOf(BlockSpec{}),
		reflect.TypeOf(DataSpec{}),
		reflect.TypeOf(TrainingSpec{}),
		reflect.TypeOf(EvalSpec{}),
	}
	var missing []string
	for _, typ := range types {
		for i := 0; i < typ.NumField(); i++ {
			tag := typ.Field(i).Tag.Get("json")
			name := strings.Split(tag, ",")[0]
			if name == "" || name == "-" {
				continue
			}
			if !strings.Contains(reference, "`"+name+"`") {
				missing = append(missing, typ.Name()+"."+name)
			}
		}
	}
	sort.Strings(missing)
	if len(missing) > 0 {
		t.Fatalf("public JSON fields missing from docs/config-reference.md:\n  %s",
			strings.Join(missing, "\n  "))
	}
}
