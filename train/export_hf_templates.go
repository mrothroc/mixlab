package train

import (
	"embed"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

//go:embed hf_templates/configuration_mixlab.py hf_templates/modeling_mixlab.py hf_templates/pooling_mixlab.py hf_templates/ttt_mlp_mixlab.py hf_templates/mamba3_mixlab.py hf_templates/s4d_mixlab.py
var hfTemplateFS embed.FS

type hfOptionalBlockTemplate struct {
	blockType  string
	fileName   string
	importLine string
}

var hfOptionalBlockTemplates = []hfOptionalBlockTemplate{
	{
		blockType:  "ttt_mlp",
		fileName:   "ttt_mlp_mixlab.py",
		importLine: "from .ttt_mlp_mixlab import MixlabTTTMLPBlock, require_right_padded_ttt_batch",
	},
	{
		blockType:  "mamba3-canonical",
		fileName:   "mamba3_mixlab.py",
		importLine: "from .mamba3_mixlab import MixlabMamba3CanonicalBlock",
	},
	{
		blockType:  "s4d",
		fileName:   "s4d_mixlab.py",
		importLine: "from .s4d_mixlab import MixlabS4DBlock",
	},
}

func hfTemplateNamesForConfig(cfg *ArchConfig) []string {
	names := []string{"configuration_mixlab.py", "modeling_mixlab.py", "pooling_mixlab.py"}
	required := hfRequiredBlockTypes(cfg)
	for _, optional := range hfOptionalBlockTemplates {
		if required[optional.blockType] {
			names = append(names, optional.fileName)
		}
	}
	return names
}

func hfRequiredBlockTypes(cfg *ArchConfig) map[string]bool {
	required := make(map[string]bool)
	if cfg != nil {
		for _, block := range cfg.Blocks {
			required[strings.ToLower(strings.TrimSpace(block.Type))] = true
		}
	}
	return required
}

func writeHFTemplates(outputDir string, cfg *ArchConfig) error {
	names := hfTemplateNamesForConfig(cfg)
	selected := make(map[string]bool, len(names))
	for _, name := range names {
		selected[name] = true
	}
	for _, optional := range hfOptionalBlockTemplates {
		if selected[optional.fileName] {
			continue
		}
		if err := os.Remove(filepath.Join(outputDir, optional.fileName)); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("remove unused HF template %s: %w", optional.fileName, err)
		}
	}
	for _, name := range names {
		data, err := hfTemplateFS.ReadFile(filepath.Join("hf_templates", name))
		if err != nil {
			return fmt.Errorf("read HF template %s: %w", name, err)
		}
		if name == "modeling_mixlab.py" {
			data, err = renderHFModelingTemplate(data, cfg)
			if err != nil {
				return err
			}
		}
		if err := os.WriteFile(filepath.Join(outputDir, name), data, 0o644); err != nil {
			return fmt.Errorf("write HF template %s: %w", name, err)
		}
	}
	return nil
}

func renderHFModelingTemplate(data []byte, cfg *ArchConfig) ([]byte, error) {
	const marker = "# MIXLAB_OPTIONAL_BLOCK_IMPORTS"
	source := string(data)
	if strings.Count(source, marker) != 1 {
		return nil, fmt.Errorf("HF modeling template must contain exactly one optional-import marker")
	}
	required := hfRequiredBlockTypes(cfg)
	imports := make([]string, 0, len(required))
	for _, optional := range hfOptionalBlockTemplates {
		if required[optional.blockType] {
			imports = append(imports, optional.importLine)
		}
	}
	return []byte(strings.Replace(source, marker, strings.Join(imports, "\n"), 1)), nil
}
