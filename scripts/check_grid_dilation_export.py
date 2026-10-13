"""Check a tiny exported dilated Conv2d against native post-training predictions."""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def main():
    package, reference = map(Path, sys.argv[1:])
    config = json.loads((package / "config.json").read_text())
    params = config["blocks"][0]["ops"][0]["params"]
    native = json.loads(reference.read_text())
    model = torch.nn.Conv2d(2, 1, params["kernel"], stride=params.get("stride", 1),
                            padding=params["padding"], dilation=params["dilation"])
    model.load_state_dict(load_file(package / "model.safetensors"), strict=True)
    x = torch.tensor(native["inputs"], dtype=torch.float32).reshape(2, 8, 8, 2)
    with torch.no_grad():
        result = model(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).flatten()
    expected = torch.tensor(native["predictions"], dtype=torch.float32)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)
    print(f"dilation={params['dilation']} max_abs={(result-expected).abs().max().item():.9g}")


if __name__ == "__main__":
    main()
