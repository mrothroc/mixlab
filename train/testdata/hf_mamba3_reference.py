#!/usr/bin/env python3
"""Check the shipped HF Mamba-3 block against the independent scalar fixture."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch


WEIGHT_MAP = {
    "pre_norm_scale": "pre_norm.weight",
    "w_x": "w_x.weight",
    "conv_w": "conv_weight",
    "w_dt_low": "w_dt_low.weight",
    "w_dt_high": "w_dt_high.weight",
    "w_lambda_low": "w_lambda_low.weight",
    "w_lambda_high": "w_lambda_high.weight",
    "w_theta_low": "w_theta_low.weight",
    "w_theta_high": "w_theta_high.weight",
    "w_B": "w_B.weight",
    "w_C": "w_C.weight",
    "B_norm_scale": "B_norm.weight",
    "C_norm_scale": "C_norm.weight",
    "B_bias": "B_bias",
    "C_bias": "C_bias",
    "A_log": "A_log",
    "dt_bias": "dt_bias",
    "post_norm_scale": "post_norm.weight",
    "w_gate": "w_gate.weight",
    "w_out": "w_out.weight",
}


def load_module(path):
    spec = importlib.util.spec_from_file_location("mamba3_mixlab", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--fixture", required=True)
    args = parser.parse_args()

    module = load_module(args.template)
    with open(args.fixture) as handle:
        fixture = json.load(handle)["fixtures"][0]

    config_doc = fixture["config"]
    block_config = config_doc["blocks"][0]
    config = type("Config", (), {"model_dim": int(config_doc["model_dim"])})()
    block = module.MixlabMamba3CanonicalBlock(config, block_config)

    source_weights = {weight["name"]: weight for weight in fixture["weights"]}
    state_dict = block.state_dict()
    for source_name, target_name in WEIGHT_MAP.items():
        source = source_weights[source_name]
        value = torch.tensor(source["values"], dtype=torch.float32).reshape(
            source["shape"]
        )
        state_dict[target_name] = value
    block.load_state_dict(state_dict, strict=True)
    block.eval()

    embedding = source_weights["embed"]
    embedding = torch.tensor(
        embedding["values"], dtype=torch.float32
    ).reshape(embedding["shape"])
    tokens = torch.tensor(fixture["tokens"], dtype=torch.long).reshape(
        fixture["batch"], fixture["seq_len"]
    )
    x = embedding[tokens]
    expected = torch.tensor(
        fixture["expected_hidden"], dtype=torch.float32
    ).reshape(fixture["batch"], fixture["seq_len"], fixture["model_dim"])
    with torch.no_grad():
        actual = block(x)
    max_diff = (actual - expected).abs().max().item()
    print(f"mamba3_reference_parity: max_hidden_diff={max_diff:.3e}")
    if max_diff >= 2e-5:
        raise SystemExit(
            f"canonical Mamba-3 HF reference diff {max_diff:.3e} >= 2.000e-05"
        )


if __name__ == "__main__":
    main()
