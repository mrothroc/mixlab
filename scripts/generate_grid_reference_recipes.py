"""Publish small reproducible graph/map/shape artifacts; never copy tensor data."""
import argparse
import json
from pathlib import Path
import shutil


def generate(fixtures, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for channels, label in ((2, "two_channel"), (3, "three_channel")):
        for stage, phase in ((1, "firstU"), (2, "secondU")):
            directory = Path(fixtures) / str(channels) / phase
            cfg = json.loads((directory / f"model{stage}.json").read_text())
            cfg["name"] = f"grid_unet_{label}_stage{stage}"
            cfg["training"].update(steps=1000 if stage == 1 else 500,
                min_lr_fraction=0.01, compute_dtype="float32",
                embed_weight_decay=0.0, matrix_weight_decay=0.0,
                scalar_weight_decay=0.0, head_weight_decay=0.0,
                grid_augmentation={"dihedral": True})
            (output / f"{label}_stage{stage}.json").write_text(json.dumps(cfg, indent=2) + "\n")
            shutil.copyfile(directory / "intermediate-shapes.json", output / f"{label}_stage{stage}_shapes.json")
        for source, target in (("export-map.json", f"{label}_export_map.json"), ("provenance.json", f"{label}_provenance.json")):
            shutil.copyfile(directory / source, output / target)
    licenses = Path(__file__).resolve().parent.parent / "arch/testdata/grid_reference"
    for name in ("LICENSE", "RadioUNet-LICENSE"):
        shutil.copyfile(licenses / name, output / name)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixtures", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    generate(a.fixtures, a.output)
