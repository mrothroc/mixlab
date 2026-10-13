"""Compare the tiny convolution/GroupNorm example's exported state with native."""
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import load_file


def main():
    package, reference = map(Path, sys.argv[1:])
    config = json.loads((package / "config.json").read_text())
    params = config["blocks"][0]["ops"][1]["params"]
    native = json.loads(reference.read_text())
    model = torch.nn.Sequential(OrderedDict([
        ("conv", torch.nn.Conv2d(2, 4, 3, padding=1)),
        ("norm", torch.nn.GroupNorm(params["groups"], 4, eps=params.get("eps", 1e-5))),
        ("relu", torch.nn.ReLU()),
        ("out", torch.nn.Conv2d(4, 1, 1)),
    ])).eval()
    model.load_state_dict(load_file(package / "model.safetensors"), strict=True)
    x = torch.tensor(native["inputs"], dtype=torch.float32).reshape(1, 8, 8, 2)
    with torch.no_grad():
        result = model(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).flatten()
    expected = torch.tensor(native["predictions"], dtype=torch.float32)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)
    print(f"GroupNorm groups={params['groups']} max_abs={(result-expected).abs().max().item():.9g}")


if __name__ == "__main__":
    main()
