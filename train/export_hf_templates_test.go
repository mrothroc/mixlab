package train

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestHFTemplateNamesMatchConfiguredBlockTypes(t *testing.T) {
	tests := []struct {
		name   string
		blocks []BlockSpec
		want   []string
	}{
		{
			name:   "plain transformer",
			blocks: []BlockSpec{{Type: "plain"}, {Type: "swiglu"}},
			want:   []string{"configuration_mixlab.py", "modeling_mixlab.py", "pooling_mixlab.py"},
		},
		{
			name:   "s4d only",
			blocks: []BlockSpec{{Type: "s4d"}},
			want:   []string{"configuration_mixlab.py", "modeling_mixlab.py", "pooling_mixlab.py", "s4d_mixlab.py"},
		},
		{
			name:   "mixed optional modules are deduplicated",
			blocks: []BlockSpec{{Type: "mamba3-canonical"}, {Type: "ttt_mlp"}, {Type: "MAMBA3-CANONICAL"}},
			want:   []string{"configuration_mixlab.py", "modeling_mixlab.py", "pooling_mixlab.py", "ttt_mlp_mixlab.py", "mamba3_mixlab.py"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := hfTemplateNamesForConfig(&ArchConfig{Blocks: tt.blocks})
			if !slices.Equal(got, tt.want) {
				t.Fatalf("hfTemplateNamesForConfig()=%v, want %v", got, tt.want)
			}
		})
	}
}

func TestWriteHFTemplatesRemovesStaleOptionalModules(t *testing.T) {
	dir := t.TempDir()
	for _, name := range hfTemplateNamesForConfig(&ArchConfig{Blocks: []BlockSpec{{Type: "s4d"}, {Type: "ttt_mlp"}}}) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("stale"), 0o644); err != nil {
			t.Fatalf("seed stale template %s: %v", name, err)
		}
	}
	if err := writeHFTemplates(dir, &ArchConfig{Blocks: []BlockSpec{{Type: "plain"}}}); err != nil {
		t.Fatalf("writeHFTemplates: %v", err)
	}
	for _, name := range []string{"ttt_mlp_mixlab.py", "mamba3_mixlab.py", "s4d_mixlab.py"} {
		if _, err := os.Stat(filepath.Join(dir, name)); !os.IsNotExist(err) {
			t.Fatalf("stale optional template %s remains (stat err=%v)", name, err)
		}
	}
}

func TestWriteHFTemplatesRendersOnlySelectedOptionalImports(t *testing.T) {
	tests := []struct {
		name       string
		blocks     []BlockSpec
		wantImport string
	}{
		{name: "plain", blocks: []BlockSpec{{Type: "plain"}}},
		{name: "ttt", blocks: []BlockSpec{{Type: "ttt_mlp"}}, wantImport: "from .ttt_mlp_mixlab import"},
		{name: "mamba3", blocks: []BlockSpec{{Type: "mamba3-canonical"}}, wantImport: "from .mamba3_mixlab import"},
		{name: "s4d", blocks: []BlockSpec{{Type: "s4d"}}, wantImport: "from .s4d_mixlab import"},
	}
	allImports := []string{
		"from .ttt_mlp_mixlab import",
		"from .mamba3_mixlab import",
		"from .s4d_mixlab import",
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := writeHFTemplates(dir, &ArchConfig{Blocks: tt.blocks}); err != nil {
				t.Fatalf("writeHFTemplates: %v", err)
			}
			data, err := os.ReadFile(filepath.Join(dir, "modeling_mixlab.py"))
			if err != nil {
				t.Fatalf("read rendered modeling template: %v", err)
			}
			source := string(data)
			if strings.Contains(source, "MIXLAB_OPTIONAL_BLOCK_IMPORTS") {
				t.Fatal("rendered modeling template retained optional-import marker")
			}
			for _, candidate := range allImports {
				want := candidate == tt.wantImport
				if got := strings.Contains(source, candidate); got != want {
					t.Fatalf("import %q present=%v, want %v", candidate, got, want)
				}
			}
		})
	}
}
