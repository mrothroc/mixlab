package train

import (
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

//go:embed torch_state_inference.py
var torchStateInference []byte

type ExportTorchStateOptions struct{ ConfigPath, SafetensorsLoad, MapPath, OutputDir string }

// RunExportTorchState exports data only. It never imports or executes Python.
func RunExportTorchState(opts ExportTorchStateOptions) error {
	if opts.ConfigPath == "" || opts.SafetensorsLoad == "" || opts.MapPath == "" || opts.OutputDir == "" {
		return fmt.Errorf("export-torch-state requires -config, -safetensors-load, -export-map and -export-dir")
	}
	cfg, err := LoadArchConfig(opts.ConfigPath)
	if err != nil {
		return err
	}
	if !cfg.GridEnabled() {
		return fmt.Errorf("export-torch-state v1 supports dense grid models only")
	}
	mapping, err := loadTorchStateMap(opts.MapPath)
	if err != nil {
		return err
	}
	shapes, err := computeWeightShapes(cfg)
	if err != nil {
		return err
	}
	weights, _, err := loadGridWeights(opts.SafetensorsLoad, cfg, shapes, nil)
	if err != nil {
		return err
	}
	tensors, err := mapTorchState(mapping, shapes, weights)
	if err != nil {
		return err
	}
	if err = os.MkdirAll(filepath.Dir(opts.OutputDir), 0755); err != nil {
		return err
	}
	if _, err = os.Lstat(opts.OutputDir); !os.IsNotExist(err) {
		return fmt.Errorf("export directory must not already exist: %s", opts.OutputDir)
	}
	tmp, err := os.MkdirTemp(filepath.Dir(opts.OutputDir), ".torch-state-")
	if err != nil {
		return err
	}
	defer func() { _ = os.RemoveAll(tmp) }()
	if err = writeFloatSafetensorsAtomic(filepath.Join(tmp, "model.safetensors"), tensors, map[string]string{"format": "pt"}, false); err != nil {
		return err
	}
	hashes := map[string]string{}
	for name, path := range map[string]string{"checkpoint": opts.SafetensorsLoad, "config": opts.ConfigPath, "export_map": opts.MapPath} {
		hashes[name], err = torchStateFileHash(path)
		if err != nil {
			return err
		}
	}
	manifest := map[string]any{
		"format": "mixlab.torch_state_export.v1", "weights": "model.safetensors", "dtype": "float32",
		"selected_output": cfg.DenseRegression.Output, "input_layout": "NCHW", "output_layout": "NCHW",
		"height": cfg.InputAdapter.Height, "width": cfg.InputAdapter.Width, "input_channels": cfg.InputAdapter.Channels,
		"target_channels": cfg.DenseRegression.TargetChannels, "units": "model", "source_sha256": hashes,
		"mappings": mapping.Mappings, "excluded": mapping.Excluded, "provenance": mapping.Provenance,
	}
	// One-time warm-start paths are not part of an inference package.
	cfg.Training.InitFrom, cfg.Training.InitAllowMissing = "", nil
	for name, value := range map[string]any{"export.json": manifest, "config.json": cfg, "export-map.json": mapping} {
		if err = atomicWriteJSON(filepath.Join(tmp, name), value); err != nil {
			return err
		}
	}
	if err = os.WriteFile(filepath.Join(tmp, "infer.py"), torchStateInference, 0644); err != nil {
		return err
	}
	return publishGridDirectory(tmp, opts.OutputDir)
}

func torchStateFileHash(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer func() { _ = f.Close() }()
	h := sha256.New()
	if _, err = io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
