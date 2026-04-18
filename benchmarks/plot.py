#!/usr/bin/env python3
"""Plot training loss curves from a mixlab benchmark CSV."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

BASELINE_VAL_LOSS = 1.4697


def read_loss_csv(path: Path) -> tuple[list[int], list[float], list[float | None]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"no rows found in {path}")

    steps = [int(row["step"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [
        float(row["val_loss"]) if row.get("val_loss") else None
        for row in rows
    ]
    return steps, train_loss, val_loss


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results" / "shakespeare_loss.csv"
    output_path = csv_path.parent / "shakespeare_loss.png"

    steps, train_loss, val_loss = read_loss_csv(csv_path)
    val_points = [(step, loss) for step, loss in zip(steps, val_loss) if loss is not None]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)

    ax.plot(steps, train_loss, color="#1f77b4", linewidth=2.0, label="mixlab train loss")
    if val_points:
        val_steps, val_values = zip(*val_points)
        ax.plot(val_steps, val_values, color="#d62728", linewidth=2.0,
                marker="o", markersize=3.5, label="mixlab val loss")

    ax.axhline(
        BASELINE_VAL_LOSS,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.8,
        label=f"nanoGPT published val loss ({BASELINE_VAL_LOSS:.4f})",
    )

    ax.set_title("Shakespeare Character-Level Loss", pad=12)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss (nats)")
    ax.set_xlim(left=0)
    ax.legend(frameon=True, loc="best")
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
