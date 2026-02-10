#!/usr/bin/env python3
"""
Plot raw training loss timeseries from `train_steps.csv`.

This file is written automatically into each run directory:
  out/<proj_name>-<suffix...>/train_steps.csv

Usage:
  python scripts/plot_train_steps.py --input out/.../train_steps.csv --output out/.../train_steps.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to train_steps.csv")
    ap.add_argument("--output", required=True, help="Path to output PNG")
    ap.add_argument("--title", default="Training loss", help="Plot title")
    ap.add_argument("--max_points", type=int, default=0, help="Optional downsample cap (0 = no cap)")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required. Install it with: pip install matplotlib"
        ) from e

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokens: list[int] = []
    loss: list[float] = []

    with in_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            try:
                tokens.append(int(float(row["tokens"])))
                loss.append(float(row["loss"]))
            except Exception:
                # Skip malformed rows
                continue

    if not tokens:
        raise SystemExit(f"No data found in {in_path}")

    if args.max_points and len(tokens) > args.max_points:
        # Uniform downsample
        step = max(1, len(tokens) // args.max_points)
        tokens = tokens[::step]
        loss = loss[::step]

    plt.figure(figsize=(10, 6))
    plt.plot(tokens, loss, linewidth=1.0)
    plt.xlabel("Tokens")
    plt.ylabel("Loss")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()

