#!/usr/bin/env python3
"""Generate traceable S4D fixtures from state-spaces/s4 v3.0.0.

The script imports the pinned upstream kernel classes from a caller-provided
checkout. It does not vendor or reimplement the reference formulas.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import pathlib
import platform
import re
import subprocess
import sys
import types
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as torch_functional


UPSTREAM_REPOSITORY = "https://github.com/state-spaces/s4"
UPSTREAM_TAG = "v3.0.0"
UPSTREAM_COMMIT = "ab287c63f4938a76d06a6b6868ee4a7163b50b05"
FULL_SOURCE = pathlib.Path("src/models/s4/s4.py")
MINIMAL_SOURCE = pathlib.Path("src/models/s4/s4d.py")
DEFAULT_CHECK_TARGET = pathlib.Path("gpu/s4d_reference_fixtures_generated_test.go")


@dataclass(frozen=True)
class ReferenceFixtures:
    minimal_kernel: np.ndarray
    advanced_kernel: np.ndarray
    advanced_output: np.ndarray
    advanced_gradients: tuple[np.ndarray, ...]


def run_git(checkout: pathlib.Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def normalize_repository_url(value: str) -> str:
    value = value.strip().removesuffix(".git").removesuffix("/")
    if value.startswith("git@github.com:"):
        value = "https://github.com/" + value.removeprefix("git@github.com:")
    return value


def verify_checkout(checkout: pathlib.Path) -> str:
    if not checkout.is_dir():
        raise ValueError(f"S4 checkout does not exist: {checkout}")
    commit = run_git(checkout, "rev-parse", "HEAD")
    if commit != UPSTREAM_COMMIT:
        raise ValueError(
            f"S4 checkout must be {UPSTREAM_TAG} ({UPSTREAM_COMMIT}), got {commit}"
        )
    tags = run_git(checkout, "tag", "--points-at", "HEAD").splitlines()
    if UPSTREAM_TAG not in tags:
        raise ValueError(f"S4 checkout HEAD is not tagged {UPSTREAM_TAG}")
    repository = normalize_repository_url(run_git(checkout, "remote", "get-url", "origin"))
    if repository != UPSTREAM_REPOSITORY:
        raise ValueError(
            f"S4 checkout origin must be {UPSTREAM_REPOSITORY}, got {repository}"
        )
    dirty = run_git(
        checkout,
        "status",
        "--porcelain",
        "--untracked-files=no",
        "--",
        str(FULL_SOURCE),
        str(MINIMAL_SOURCE),
    )
    if dirty:
        raise ValueError("S4 reference source files have uncommitted modifications")
    for relative in (FULL_SOURCE, MINIMAL_SOURCE):
        if not (checkout / relative).is_file():
            raise ValueError(f"S4 checkout is missing reference source {relative}")
    return repository


def load_module(name: str, source: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reference module {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_full_reference(checkout: pathlib.Path):
    # The standalone file only uses this decorator for logging. Avoid forcing
    # fixture regeneration to install the full Lightning training stack.
    lightning = types.ModuleType("pytorch_lightning")
    utilities = types.ModuleType("pytorch_lightning.utilities")
    utilities.rank_zero_only = lambda function: function
    lightning.utilities = utilities
    sys.modules["pytorch_lightning"] = lightning
    sys.modules["pytorch_lightning.utilities"] = utilities
    return load_module("mixlab_upstream_s4_v3", checkout / FULL_SOURCE)


def load_minimal_reference(checkout: pathlib.Path):
    # S4DKernel itself does not use DropoutNd, but the containing module imports
    # it for the higher-level S4D block.
    src = types.ModuleType("src")
    models = types.ModuleType("src.models")
    nn_module = types.ModuleType("src.models.nn")
    nn_module.DropoutNd = torch.nn.Dropout
    src.models = models
    models.nn = nn_module
    sys.modules["src"] = src
    sys.modules["src.models"] = models
    sys.modules["src.models.nn"] = nn_module
    return load_module("mixlab_upstream_s4d_minimal_v3", checkout / MINIMAL_SOURCE)


def as_float32(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().to(torch.float32).numpy().reshape(-1).copy()


def generate_minimal_fixture(reference) -> np.ndarray:
    d_model, state_size, length = 2, 4, 6
    pairs = state_size // 2
    kernel = reference.S4DKernel(d_model, N=state_size)
    c_value = torch.tensor(
        [
            [complex(1.0, 0.5), complex(-0.25, 0.75)],
            [complex(0.3, -0.2), complex(0.8, 0.1)],
        ],
        dtype=torch.complex64,
    )
    with torch.no_grad():
        kernel.log_dt.copy_(
            torch.tensor([math.log(0.01), math.log(0.05)], dtype=torch.float32)
        )
        kernel.log_A_real.copy_(
            torch.full((d_model, pairs), math.log(0.5), dtype=torch.float32)
        )
        kernel.A_imag.copy_(
            torch.tensor([[0.0, math.pi], [0.0, math.pi]], dtype=torch.float32)
        )
        kernel.C.copy_(torch.view_as_real(c_value))
    return as_float32(kernel(length))


def advanced_inputs_and_weights():
    batch, length, d_model, state_size, n_ssm = 1, 5, 4, 4, 2
    pairs = state_size // 2
    x = torch.tensor(
        [0.2 * math.sin((index + 1) * 0.31) for index in range(batch * length * d_model)],
        dtype=torch.float32,
    ).reshape(batch, length, d_model)
    log_dt = torch.tensor(
        [math.log(0.004 + 0.002 * channel) for channel in range(d_model)],
        dtype=torch.float32,
    )
    log_a_real = torch.tensor(
        [
            math.log(0.5 + 0.25 * group)
            for group in range(n_ssm)
            for _ in range(pairs)
        ],
        dtype=torch.float32,
    ).reshape(n_ssm, pairs)
    a_imag = torch.tensor(
        [math.pi * (pair + group) for group in range(n_ssm) for pair in range(pairs)],
        dtype=torch.float32,
    ).reshape(n_ssm, pairs)
    b_real = torch.tensor(
        [1.0 + 0.05 * index for index in range(n_ssm * pairs)], dtype=torch.float32
    ).reshape(n_ssm, pairs)
    b_imag = torch.tensor(
        [-0.03 * (index + 1) for index in range(n_ssm * pairs)], dtype=torch.float32
    ).reshape(n_ssm, pairs)
    c_forward_real = torch.tensor(
        [0.15 * math.sin(index + 1) for index in range(d_model * pairs)],
        dtype=torch.float32,
    ).reshape(d_model, pairs)
    c_forward_imag = torch.tensor(
        [0.12 * math.cos(index + 1) for index in range(d_model * pairs)],
        dtype=torch.float32,
    ).reshape(d_model, pairs)
    c_backward_real = torch.tensor(
        [0.11 * math.cos(index + 2) for index in range(d_model * pairs)],
        dtype=torch.float32,
    ).reshape(d_model, pairs)
    c_backward_imag = torch.tensor(
        [0.09 * math.sin(index + 2) for index in range(d_model * pairs)],
        dtype=torch.float32,
    ).reshape(d_model, pairs)
    direct = torch.tensor(
        [0.03 * (channel + 1) for channel in range(d_model)],
        dtype=torch.float32,
    )
    return (
        (batch, length, d_model, state_size, n_ssm),
        x,
        log_dt,
        log_a_real,
        a_imag,
        b_real,
        b_imag,
        c_forward_real,
        c_forward_imag,
        c_backward_real,
        c_backward_imag,
        direct,
    )


def generate_advanced_fixture(reference):
    (
        dimensions,
        x,
        log_dt,
        log_a_real,
        a_imag,
        b_real,
        b_imag,
        c_forward_real,
        c_forward_imag,
        c_backward_real,
        c_backward_imag,
        direct,
    ) = advanced_inputs_and_weights()
    batch, length, d_model, state_size, _ = dimensions
    reference_block = reference.S4(
        d_model,
        d_state=state_size,
        l_max=length,
        channels=1,
        bidirectional=True,
        activation=None,
        postact=None,
        dropout=0.0,
        transposed=True,
        mode="diag",
        measure="diag-lin",
        n_ssm=dimensions[4],
        disc="bilinear",
        real_type="exp",
        lr=None,
    )
    reference_block.activation = torch.nn.Identity()
    reference_block.dropout = torch.nn.Identity()
    reference_block.output_linear = torch.nn.Identity()
    kernel_module = reference_block.kernel.kernel

    complex_a = -torch.exp(log_a_real) + 1j * a_imag
    complex_b = b_real + 1j * b_imag
    complex_c = torch.stack(
        [
            c_forward_real + 1j * c_forward_imag,
            c_backward_real + 1j * c_backward_imag,
        ],
        dim=0,
    )
    # Set the upstream module's stored real-view parameters directly. This
    # avoids initialization conventions while retaining upstream forward and
    # autograd semantics for every trainable Mixlab tensor.
    with torch.no_grad():
        kernel_module.log_dt.copy_(log_dt)
        kernel_module.inv_A_real.copy_(log_a_real)
        kernel_module.A_imag.copy_(a_imag)
        kernel_module.B.copy_(torch.view_as_real(complex_b))
        kernel_module.C.copy_(torch.view_as_real(complex_c))
        reference_block.D.copy_(direct)

    directional_kernel, _ = kernel_module(L=length)
    forward_kernel, backward_kernel = directional_kernel.reshape(2, 1, d_model, length)[:, 0]
    combined_kernel = torch_functional.pad(forward_kernel, (0, length)) + torch_functional.pad(
        backward_kernel.flip(-1), (length, 0)
    )
    output, _ = reference_block(x.transpose(1, 2))
    output = output.transpose(1, 2)
    output.square().mean().backward()

    gradients = (
        as_float32(kernel_module.log_dt.grad),
        as_float32(kernel_module.inv_A_real.grad),
        as_float32(kernel_module.A_imag.grad),
        as_float32(kernel_module.B.grad[..., 0]),
        as_float32(kernel_module.B.grad[..., 1]),
        as_float32(kernel_module.C.grad[0, ..., 0]),
        as_float32(kernel_module.C.grad[0, ..., 1]),
        as_float32(kernel_module.C.grad[1, ..., 0]),
        as_float32(kernel_module.C.grad[1, ..., 1]),
        as_float32(reference_block.D.grad),
    )
    return as_float32(combined_kernel), as_float32(output), gradients


def generate_fixtures(checkout: pathlib.Path) -> ReferenceFixtures:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    minimal = generate_minimal_fixture(load_minimal_reference(checkout))
    advanced_kernel, advanced_output, gradients = generate_advanced_fixture(
        load_full_reference(checkout)
    )
    return ReferenceFixtures(minimal, advanced_kernel, advanced_output, gradients)


def go_float(value: np.float32) -> str:
    return format(float(np.float32(value)), ".10g")


def go_slice(name: str, values: Iterable[np.float32], indent: str = "") -> str:
    values = list(values)
    lines = [f"{indent}var {name} = []float32{{"]
    for start in range(0, len(values), 5):
        chunk = ", ".join(go_float(value) for value in values[start : start + 5])
        lines.append(f"{indent}\t{chunk},")
    lines.append(f"{indent}}}")
    return "\n".join(lines)


def render_go(fixtures: ReferenceFixtures, repository: str) -> str:
    gradients = []
    for values in fixtures.advanced_gradients:
        rows = []
        for start in range(0, len(values), 5):
            rows.append("\t\t" + ", ".join(go_float(value) for value in values[start : start + 5]) + ",")
        gradients.append("\t{\n" + "\n".join(rows) + "\n\t}")
    gradient_text = ",\n".join(gradients)
    return f'''// Code generated by scripts/gen_s4d_reference_fixture.py; DO NOT EDIT.
//
// Reference repository: {repository}
// Reference tag: {UPSTREAM_TAG}
// Reference commit: {UPSTREAM_COMMIT}
// Reference entry points:
//   - {MINIMAL_SOURCE}:S4DKernel.forward
//   - {FULL_SOURCE}:SSKernelDiag.forward through S4.forward
// Reference scope: core SSM output; activation, dropout, and output projection are identity.
// Generator environment: torch {torch.__version__}; numpy {np.__version__}; {platform.platform()}; float32

//go:build mlx && cgo && (darwin || linux)

package gpu

// Produced by {MINIMAL_SOURCE}:S4DKernel.forward.
{go_slice("s4dOfficialMinimalKernelFixture", fixtures.minimal_kernel)}

// Produced by {FULL_SOURCE}:SSKernelDiag.forward through S4.forward.
{go_slice("s4dOfficialAdvancedKernelFixture", fixtures.advanced_kernel)}

// Produced by {FULL_SOURCE}:S4.forward before activation/output projection.
{go_slice("s4dOfficialAdvancedOutputFixture", fixtures.advanced_output)}

// Produced by torch autograd over mean(square(output)) from S4.forward.
var s4dOfficialAdvancedGradientFixtures = [][]float32{{
{gradient_text},
}}
'''


def parse_go_slice(source: str, name: str) -> np.ndarray:
    match = re.search(
        rf"var\s+{re.escape(name)}\s*=\s*\[\]float32\s*\{{(.*?)\n\}}",
        source,
        re.DOTALL,
    )
    if match is None:
        raise ValueError(f"generated fixture is missing {name}")
    values = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", match.group(1))
    return np.asarray([np.float32(value) for value in values], dtype=np.float32)


def parse_go_gradients(source: str) -> tuple[np.ndarray, ...]:
    match = re.search(
        r"var\s+s4dOfficialAdvancedGradientFixtures\s*=\s*\[\]\[\]float32\s*\{(.*?)\n\}",
        source,
        re.DOTALL,
    )
    if match is None:
        raise ValueError("generated fixture is missing advanced gradients")
    arrays = []
    for body in re.findall(r"\{(.*?)\}", match.group(1), re.DOTALL):
        values = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", body)
        arrays.append(np.asarray([np.float32(value) for value in values], dtype=np.float32))
    return tuple(arrays)


def assert_close(name: str, got: np.ndarray, want: np.ndarray, tolerance: float) -> None:
    if got.shape != want.shape:
        raise ValueError(f"{name} shape mismatch: generated {got.shape}, checked-in {want.shape}")
    difference = float(np.max(np.abs(got - want), initial=0.0))
    if difference > tolerance:
        raise ValueError(f"{name} differs by {difference:g}, tolerance {tolerance:g}")
    print(f"{name}: match (L_inf={difference:g}, tolerance={tolerance:g})")


def check_generated(path: pathlib.Path, fixtures: ReferenceFixtures) -> None:
    source = path.read_text(encoding="utf-8")
    for required in (UPSTREAM_REPOSITORY, UPSTREAM_TAG, UPSTREAM_COMMIT, str(FULL_SOURCE), str(MINIMAL_SOURCE)):
        if required not in source:
            raise ValueError(f"generated fixture provenance is missing {required}")
    assert_close(
        "minimal kernel",
        fixtures.minimal_kernel,
        parse_go_slice(source, "s4dOfficialMinimalKernelFixture"),
        2e-6,
    )
    assert_close(
        "advanced kernel",
        fixtures.advanced_kernel,
        parse_go_slice(source, "s4dOfficialAdvancedKernelFixture"),
        3e-5,
    )
    assert_close(
        "advanced output",
        fixtures.advanced_output,
        parse_go_slice(source, "s4dOfficialAdvancedOutputFixture"),
        4e-5,
    )
    checked_gradients = parse_go_gradients(source)
    if len(checked_gradients) != len(fixtures.advanced_gradients):
        raise ValueError(
            f"advanced gradient count mismatch: generated {len(fixtures.advanced_gradients)}, "
            f"checked-in {len(checked_gradients)}"
        )
    for index, (got, want) in enumerate(zip(fixtures.advanced_gradients, checked_gradients)):
        assert_close(f"advanced gradient {index}", got, want, 2e-6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s4-checkout", required=True, type=pathlib.Path)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output", type=pathlib.Path, help="write a generated Go fixture file")
    output.add_argument(
        "--check",
        nargs="?",
        const=DEFAULT_CHECK_TARGET,
        type=pathlib.Path,
        help="compare upstream values with a checked-in generated fixture",
    )
    args = parser.parse_args()
    try:
        checkout = args.s4_checkout.resolve()
        repository = verify_checkout(checkout)
        fixtures = generate_fixtures(checkout)
        rendered = render_go(fixtures, repository)
        if args.check is not None:
            check_generated(args.check, fixtures)
        elif args.output is not None:
            args.output.write_text(rendered, encoding="utf-8")
            print(f"wrote {args.output}")
        else:
            sys.stdout.write(rendered)
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
