#!/usr/bin/env python3
"""
Update Stage 1/2/3 training curves and eval comparison chart from W&B runs.

Outputs (relative to repo root):
  ../wandb_plots/stage1_attention_loss.png
  ../wandb_plots/stage2_kl_loss.png
  ../wandb_plots/stage2_token_accuracy.png
  ../wandb_plots/stage3_ce_loss.png
  ../wandb_plots/eval_comparison.png
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from datetime import datetime

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None


def _load_env() -> None:
    if load_dotenv is None:
        return
    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _norm_name(value: str) -> str:
    return " ".join(value.split()).strip()


def _find_run(api: wandb.Api, entity: str, project: str, run_name: str):
    target = _norm_name(run_name)
    runs = api.runs(f"{entity}/{project}")
    for run in runs:
        name = _norm_name(run.name)
        if name == target:
            return run
    for run in runs:
        name = _norm_name(run.name)
        if target in name:
            return run
    return None


def _run_timestamp(run) -> float:
    created = getattr(run, "created_at", None)
    if isinstance(created, str) and created:
        try:
            return datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
        except Exception:
            pass
    summary_ts = None
    try:
        summary_ts = run.summary.get("_timestamp")
    except Exception:
        summary_ts = None
    if isinstance(summary_ts, (int, float)):
        return float(summary_ts)
    return 0.0


def _history(run, keys):
    hist = run.history(keys=keys, pandas=True)
    if isinstance(hist, pd.DataFrame) and not hist.empty:
        return hist
    # Fallback to full history if keyed history is empty
    hist = run.history(pandas=True)
    if isinstance(hist, pd.DataFrame):
        return hist
    return pd.DataFrame(hist)


def _scan_history(run, keys, max_rows: int = 20000) -> pd.DataFrame:
    rows = []
    try:
        for row in run.scan_history(keys=keys):
            rows.append(row)
            if len(rows) >= max_rows:
                break
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _pick_key(hist: pd.DataFrame, candidates: list[str]) -> str | None:
    for cand in candidates:
        if cand in hist and hist[cand].notna().any():
            return cand
    return None


def _combine_histories(runs, key_candidates: list[str]) -> tuple[pd.DataFrame, bool]:
    frames = []
    use_tokens = False
    token_cols = ["train/tokens", "tokens"]
    requested_keys = token_cols + key_candidates + ["_step"]

    for run in runs:
        hist = _history(run, requested_keys)
        if hist.empty:
            hist = _scan_history(run, requested_keys)
        if hist.empty:
            continue
        frames.append(hist)

    if not frames:
        return pd.DataFrame(), False

    # Build a monotonic x-axis across multiple runs
    x_frames = []
    token_offset = 0.0
    step_offset = 0.0
    for hist in frames:
        hist = hist.copy()
        token_col = next((c for c in token_cols if c in hist and hist[c].notna().any()), None)
        if token_col:
            use_tokens = True
            tokens = hist[token_col].fillna(0.0)
            hist["__x"] = tokens + token_offset
            token_offset += float(tokens.max()) if len(tokens) else 0.0
        else:
            steps = hist["_step"] if "_step" in hist else pd.Series(range(len(hist)))
            hist["__x"] = steps.fillna(0.0) + step_offset
            step_offset += float(steps.max()) if len(steps) else 0.0
        x_frames.append(hist)

    combined = pd.concat(x_frames, ignore_index=True)
    combined = combined.sort_values("__x")
    return combined, use_tokens


def _plot_lines(
    runs,
    key_candidates: list[str],
    title: str,
    ylabel: str,
    output_path: Path,
    *,
    y_max: float | None = None,
    token_multipliers: dict[str, float] | None = None,
):
    """Plot training curves.

    Args:
        y_max: Optional upper limit for y-axis.
        token_multipliers: Per-label multiplier for train/tokens (e.g. to
            correct for acc_grad not being included in logged token counts).
    """
    plt.figure(figsize=(10, 6))
    xlabel = "Training steps"
    for label, run_list in runs:
        hist, use_tokens = _combine_histories(run_list, key_candidates)
        if hist.empty:
            print(f"[warn] {label}: empty history")
            continue
        key = _pick_key(hist, key_candidates)
        if key is None:
            print(f"[warn] {label}: missing keys {key_candidates}")
            continue
        hist = hist.dropna(subset=[key])
        x = hist["__x"]
        if use_tokens:
            mult = 1.0
            if token_multipliers and label in token_multipliers:
                mult = token_multipliers[label]
            x = x * mult / 1e9
            xlabel = "Training tokens (billions)"
        else:
            xlabel = "Training steps"
        plt.plot(x, hist[key], label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if y_max is not None:
        plt.ylim(bottom=0, top=y_max)
    plt.legend()
    plt.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[ok] wrote {output_path}")


def _plot_eval_comparison(output_path: Path):
    """Generate a grouped bar chart comparing eval benchmarks."""
    benchmarks = ["lmbda", "mmlu", "arc_c", "arc_e", "hella", "piqa", "winog"]
    labels = ["lambada", "mmlu", "arc_c", "arc_e", "hellaswag", "piqa", "winogrande"]

    rwkv_05b = [0.5973, 0.3748, 0.3089, 0.6397, 0.3879, 0.6991, 0.5620]
    poly_s2 = [0.4207, 0.2649, 0.2713, 0.6178, 0.3718, 0.6882, 0.5375]
    poly_s3 = [0.4244, 0.2621, 0.2739, 0.6124, 0.3725, 0.6877, 0.5343]

    x = np.arange(len(benchmarks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, rwkv_05b, width, label="RWKV 0.5B (Stage 2)", color="#4C72B0", alpha=0.9)
    bars2 = ax.bar(x, poly_s2, width, label="Polysketch Stage 2", color="#DD8452", alpha=0.9)
    bars3 = ax.bar(x + width, poly_s3, width, label="Polysketch Stage 3", color="#55A868", alpha=0.9)

    ax.set_ylabel("Score")
    ax.set_title("Atlas-LMM 0.5B (Polysketch) vs RWKV 0.5B Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 0.85)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[ok] wrote {output_path}")


def _infer_entity(api: wandb.Api) -> str | None:
    try:
        viewer = getattr(api, "viewer", None)
        if callable(viewer):
            viewer = viewer()
    except Exception:
        viewer = None
    if isinstance(viewer, dict):
        return viewer.get("entity") or viewer.get("username")
    if viewer is not None:
        return getattr(viewer, "entity", None) or getattr(viewer, "username", None)
    return None


def main() -> None:
    _load_env()

    project = os.environ.get("WANDB_PROJECT", "radlads-atlas-ablation")
    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY") or os.environ.get("WANDB_USERNAME") or _infer_entity(api)
    if not entity:
        raise SystemExit("WANDB_ENTITY (or WANDB_USERNAME) is required")

    # --- Atlas-LMM 0.5B (Polysketch) run names ---
    atlas_stage1 = (
        "2026-01-30-03-49-58 | ablation-detach-poly2-polymodepolysketch-qknorm0-"
        "qkvconvnone-rope0-gn0-stage1/20260130-034819Z | stage1-align | "
        "Qwen2.5-0.5B-Instruct L24 D896 | ctx512 | atlas=lmm ow=16 poly=2 qknorm0 rope0 gn0"
    )
    atlas_stage2 = (
        "2026-02-02-13-59-43 | polysketch-stage2-gradcp0-20260202-135934Z | stage2-kl | "
        "hf-model | ctx512 | atlas=lmm ow=16 poly=2 qknorm0 rope0 gn0"
    )
    atlas_stage3 = (
        "2026-02-03-06-56-37 | atlas-stage3-ctx16384 | hf-model | ctx16384 | "
        "atlas=lmm ow=16 poly=2 qknorm0 rope0 gn0"
    )

    # --- RWKV run names ---
    rwkv05_stage1 = "rwkv0.5b stage 1 | qwerky7_qwen2 L24 D896 ctx512 2026-01-20-09-47-20"
    rwkv05_stage2 = "rwkv0.5b stage 2 | qwerky7_qwen2 L24 D896 ctx512 2026-01-20-10-15-20"
    rwkv05_stage3 = "rwkv0.5b stage 3 | qwerky7_qwen2 L24 D896 ctx16384 2026-01-22-04-02-54"
    rwkv7_stage1 = "rwkv7b stage 1 | qwerky7_qwen2 L28 D3584 ctx512 2026-01-19-04-22-37"
    rwkv7_stage2 = "rwkv7b stage 2 | qwerky7_qwen2 L28 D3584 ctx512 2026-01-19-14-17-47"
    rwkv7_stage3 = "rwkv7b stage 3 | qwerky7_qwen2 L28 D3584 ctx16384 2026-01-22-05-19-11"

    # Stage 1 runs
    stage1_runs = [
        ("Atlas-LMM 0.5B Polysketch", [_find_run(api, entity, project, atlas_stage1)]),
        ("RWKV 0.5B", [_find_run(api, entity, project, rwkv05_stage1)]),
        ("RWKV 7B", [_find_run(api, entity, project, rwkv7_stage1)]),
    ]

    # Stage 2 runs
    stage2_runs = [
        ("Atlas-LMM 0.5B Polysketch", [_find_run(api, entity, project, atlas_stage2)]),
        ("RWKV 0.5B", [_find_run(api, entity, project, rwkv05_stage2)]),
        ("RWKV 7B", [_find_run(api, entity, project, rwkv7_stage2)]),
    ]

    # Stage 3 runs
    stage3_runs = [
        ("Atlas-LMM 0.5B Polysketch", [_find_run(api, entity, project, atlas_stage3)]),
        ("RWKV 0.5B", [_find_run(api, entity, project, rwkv05_stage3)]),
        ("RWKV 7B", [_find_run(api, entity, project, rwkv7_stage3)]),
    ]

    for label, runs in stage1_runs + stage2_runs + stage3_runs:
        for run in runs:
            if run is None:
                raise SystemExit(f"Run not found: {label}")

    root = Path(__file__).resolve().parents[1].parent
    out_dir = root / "wandb_plots"

    _plot_lines(
        stage1_runs,
        key_candidates=["train/loss", "loss"],
        title="Stage 1 Attention Alignment Loss",
        ylabel="Attention alignment loss",
        output_path=out_dir / "stage1_attention_loss.png",
        y_max=2,
    )
    _plot_lines(
        stage2_runs,
        key_candidates=["train/loss", "loss"],
        title="Stage 2 KL Divergence Loss",
        ylabel="KL divergence loss",
        output_path=out_dir / "stage2_kl_loss.png",
    )
    _plot_lines(
        stage2_runs,
        key_candidates=["train/acc", "acc"],
        title="Stage 2 Token Prediction Accuracy",
        ylabel="Token accuracy",
        output_path=out_dir / "stage2_token_accuracy.png",
    )
    # Atlas Stage 3 logs train/tokens per micro-step (without acc_grad=6),
    # so multiply by 6 to get actual tokens processed.
    _plot_lines(
        stage3_runs,
        key_candidates=["train/loss", "loss"],
        title="Stage 3: Long Context Training Loss",
        ylabel="Cross-entropy loss",
        output_path=out_dir / "stage3_ce_loss.png",
        token_multipliers={"Atlas-LMM 0.5B Polysketch": 6.0},
    )

    # Eval comparison chart (static data, no W&B needed)
    _plot_eval_comparison(out_dir / "eval_comparison.png")


if __name__ == "__main__":
    main()
