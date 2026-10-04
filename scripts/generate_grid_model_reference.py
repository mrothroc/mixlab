"""Generate spatial graph/count/forward fixtures from the pinned public model.

Run with --source modules.py --output-dir /tmp/reference. Large tensors stay out
of the repository. Only the unused torchvision import is removed; forward and
layer definitions execute unchanged. No benchmark-specific runtime code is used.
"""
import argparse
import ast
import hashlib
import json
import operator
from pathlib import Path
import struct

import numpy as np
import torch
from torch import nn
from torch.fx import symbolic_trace

COMMIT = "36ab70663443af615c37051ce97da14d04925c7a"
URL = f"https://raw.githubusercontent.com/RonLevie/RadioUNet/{COMMIT}/lib/modules.py"
SOURCE_SHA256 = "60ee896f812710f6710cc0f3fc2f573844c0ebaabb4c331b452ed259bfd24235"


def generate(source, output, channels, phase="firstU", samples=1):
    raw = Path(source).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("reference source does not match the pinned SHA256")
    tree = ast.parse(raw)
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == "torchvision")]
    ns = {}
    exec(compile(tree, str(source), "exec"), ns)
    torch.manual_seed(42)
    model = ns["RadioWNet"](inputs=channels, phase=phase).eval()
    graph = symbolic_trace(model)
    modules = dict(model.named_modules())
    params = dict(model.named_parameters())
    weights, arrays = [], []
    for name, p in params.items():
        parent, leaf = name.rsplit(".", 1)
        a = p.detach().numpy()
        if leaf == "weight":
            a = a.transpose(0, 2, 3, 1)
        init = {"kind": "pytorch_conv_uniform"}
        if leaf == "bias": init["weight"] = parent + ".weight"
        weights.append({"name": name, "shape": list(map(str, a.shape)), "init": init})
        arrays.append(np.ascontiguousarray(a))
    names, ops = {}, []
    outputs = None
    def emit(kind, ins, name, **params):
        op = dict(op=kind, inputs=ins, output=name)
        if params: op["params"] = params
        ops.append(op)
    for node in graph.graph.nodes:
        names[node] = node.name
        if node.op == "placeholder": names[node] = "x"
        elif node.op == "call_module":
            m = modules[node.target]; ins = [names[node.args[0]]]
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                emit("conv_transpose2d" if isinstance(m, nn.ConvTranspose2d) else "conv2d",
                     ins + [node.target + ".weight", node.target + ".bias"], node.name,
                     kernel=m.kernel_size[0], stride=m.stride[0], padding=m.padding[0])
            elif isinstance(m, nn.ReLU): emit("relu", ins, node.name)
            elif isinstance(m, nn.MaxPool2d): emit("max_pool2d", ins, node.name, kernel=m.kernel_size, stride=m.stride, padding=0)
            else: raise ValueError(type(m))
        elif node.op == "call_function" and node.target == operator.getitem:
            # This reference slices exactly the supplied input channels.
            assert node.args[1] == (slice(None), slice(0, channels), slice(None), slice(None))
            names[node] = names[node.args[0]]
        elif node.op == "call_function" and node.target == torch.cat:
            assert node.kwargs["dim"] == 1 and len(node.args[0]) == 2
            emit("concat", [names[n] for n in node.args[0]], node.name, axis=3)
        elif node.op == "call_method" and node.target == "detach": emit("stop_gradient", [names[node.args[0]]], node.name)
        elif node.op == "output": outputs = [names[n] for n in node.args[0]]
        else: raise ValueError(str(node))
    config = dict(name="grid_reference", input_adapter=dict(kind="grid", channels=channels, height=256, width=256),
                  dense_regression=dict(output="network." + outputs[0], target_channels=1),
                  blocks=[dict(type="custom", name="network", weights=weights, ops=ops)],
                  training=dict(objective="dense_regression", batch_size=1, steps=2, optimizer="adamw", lr=1e-4,
                                beta1=.9, beta2=.999, epsilon=1e-8, weight_decay=0., warmup_steps=0, hold_steps=0, seed=42))
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for j, selected in enumerate(outputs):
        config["dense_regression"]["output"] = "network." + selected
        if phase == "secondU" and j == 1:
            config["training"]["freeze"] = ["network." + name for name in params if not name.startswith("W")]
        else:
            config["training"].pop("freeze", None)
        (output / f"model{j+1}.json").write_text(json.dumps(config, indent=2) + "\n")
    mapping = {"format": "mixlab.torch_state_map.v1", "provenance": {
        "source_url": URL, "source_sha256": SOURCE_SHA256, "commit": COMMIT,
        "license": "MIT; see bundled reference license notices"}, "mappings": []}
    for name, p in params.items():
        entry = dict(source="network." + name, target=name, shape=list(p.shape), transform="identity")
        if p.ndim == 4:
            entry.update(transform="transpose", axes=[0, 3, 1, 2])
        mapping["mappings"].append(entry)
    (output / "export-map.json").write_text(json.dumps(mapping, indent=2) + "\n")
    header, offset = {}, 0
    for j, (w, a) in enumerate(zip(weights, arrays)):
        size = a.nbytes
        header[f"w{j}_network.{w['name']}"] = dict(dtype="F32", shape=list(a.shape), data_offsets=[offset, offset+size])
        offset += size
    meta = json.dumps(header).encode(); meta += b" " * ((-len(meta)) % 8)
    with (output / "weights.safetensors").open("wb") as f:
        f.write(struct.pack("<Q", len(meta))); f.write(meta)
        for a in arrays: f.write(a.astype("<f4").tobytes())
    x = torch.rand(samples, channels, 256, 256)
    with torch.no_grad():
        ys = [torch.cat(parts) for parts in zip(*(model(row.unsqueeze(0)) for row in x))]
        from torch.fx.passes.shape_prop import ShapeProp
        ShapeProp(graph).propagate(x[:1])
    intermediate = {}
    for node in graph.graph.nodes:
        if node.name in {op["output"] for op in ops}:
            shape = node.meta["tensor_meta"].shape
            intermediate[node.name] = [shape[0], shape[2], shape[3], shape[1]]
    (output / "intermediate-shapes.json").write_text(json.dumps(intermediate, indent=2) + "\n")
    np.save(output / "input.npy", x.numpy())
    for j, y in enumerate(ys): np.save(output / f"output{j+1}.npy", y.numpy())
    np.save(output / "mask.npy", np.ones((samples, 1, 256, 256), dtype=np.float32))
    (output / "grid-source.json").write_text(json.dumps({"splits": {"train": {
        "inputs": "input.npy", "targets": "output1.npy", "masks": "mask.npy"}}}) + "\n")
    counts = dict(total=sum(p.numel() for p in params.values()), first=sum(p.numel() for n,p in params.items() if not n.startswith("W")))
    counts["refinement"] = counts["total"] - counts["first"]
    expected = 13274031 if channels == 2 else 13275315
    assert counts["total"] == expected, counts
    (output / "provenance.json").write_text(json.dumps(dict(url=URL, commit=COMMIT, source_sha256=hashlib.sha256(raw).hexdigest(),
              modification="removed unused torchvision import only", torch=torch.__version__, numpy=np.__version__, channels=channels, phase=phase, counts=counts), indent=2)+"\n")
    print(json.dumps(counts))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--source", required=True); p.add_argument("--output-dir", required=True); p.add_argument("--channels", type=int, choices=(2,3), default=2)
    p.add_argument("--phase", choices=("firstU", "secondU"), default="firstU")
    p.add_argument("--samples", type=int, choices=(1, 4), default=1)
    args = p.parse_args(); generate(args.source, args.output_dir, args.channels, args.phase, args.samples)
