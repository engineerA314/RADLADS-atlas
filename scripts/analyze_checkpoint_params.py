#!/usr/bin/env python3
"""
Checkpoint Parameter Analysis Script

Compares parameters between two checkpoints to identify potential sources of
gradient explosion. Focuses on:
1. Polynomial feature map parameters (polysketch: poly_coeff, proj)
2. Hyperparameter layers (to_lr, to_decay, to_momentum, to_omega_gate)
3. Memory state S value distributions
4. Layer-wise parameter statistics

Usage:
    python scripts/analyze_checkpoint_params.py \
        --ckpt1 path/to/stage1_elementwise/rwkv-final.pth \
        --ckpt2 path/to/stage1_polysketch/rwkv-final.pth \
        --output analysis_results.json

Or download from GCS first:
    gsutil cp gs://bucket/prefix/path/rwkv-final.pth ./ckpt1.pth
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import numpy as np


def tensor_stats(t: torch.Tensor) -> dict:
    """Compute comprehensive statistics for a tensor."""
    t_flat = t.float().flatten()
    t_abs = t_flat.abs()

    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "min": float(t_flat.min()),
        "max": float(t_flat.max()),
        "mean": float(t_flat.mean()),
        "std": float(t_flat.std()),
        "abs_max": float(t_abs.max()),
        "abs_mean": float(t_abs.mean()),
        "l2_norm": float(t_flat.norm(2)),
        "num_nan": int(torch.isnan(t_flat).sum()),
        "num_inf": int(torch.isinf(t_flat).sum()),
        "percentile_1": float(torch.quantile(t_flat, 0.01)),
        "percentile_99": float(torch.quantile(t_flat, 0.99)),
        "num_gt_1": int((t_abs > 1.0).sum()),
        "num_gt_10": int((t_abs > 10.0).sum()),
        "num_gt_100": int((t_abs > 100.0).sum()),
    }


def analyze_memory_layer_params(state_dict: dict, layer_idx: int, prefix: str = "model.layers") -> dict:
    """Analyze parameters of a specific memory layer."""
    layer_prefix = f"{prefix}.{layer_idx}.memory"

    result = {
        "layer_idx": layer_idx,
        "params": {}
    }

    # Key parameters to analyze
    target_params = [
        # Polynomial feature map
        "cell.phi.poly_coeff",      # polysketch only
        "cell.phi.proj.weight",      # polysketch only
        "cell.phi.proj.bias",        # polysketch only

        # Hyperparameter layers
        "cell.to_lr.0.weight",
        "cell.to_lr.0.bias",
        "cell.to_decay.0.weight",
        "cell.to_decay.0.bias",
        "cell.to_momentum.0.weight",
        "cell.to_momentum.0.bias",
        "cell.to_omega_gate.0.weight",
        "cell.to_omega_gate.0.bias",

        # Q/K/V projections
        "cell.to_q.weight",
        "cell.to_k.weight",
        "cell.to_v.weight",
        "cell.to_out.weight",

        # Norms
        "cell.q_norm.gamma",
        "cell.k_norm.gamma",
        "cell.pre_norm.weight",
    ]

    for param_suffix in target_params:
        full_key = f"{layer_prefix}.{param_suffix}"
        if full_key in state_dict:
            result["params"][param_suffix] = tensor_stats(state_dict[full_key])

    return result


def analyze_hyperparams_effective_values(state_dict: dict, layer_idx: int, prefix: str = "model.layers") -> dict:
    """Compute effective hyperparameter values after sigmoid."""
    layer_prefix = f"{prefix}.{layer_idx}.memory"

    result = {}

    # Check what the initialized bias values would produce through sigmoid
    hyperparam_configs = [
        ("to_lr", -4.0, "learning_rate"),      # sigmoid(-4) ≈ 0.018
        ("to_decay", 4.0, "decay"),             # sigmoid(4) ≈ 0.982
        ("to_momentum", 2.0, "momentum"),       # sigmoid(2) ≈ 0.88
        ("to_omega_gate", 0.0, "omega_gate"),   # sigmoid(0) = 0.5
    ]

    for param_name, expected_init, label in hyperparam_configs:
        bias_key = f"{layer_prefix}.cell.{param_name}.0.bias"
        weight_key = f"{layer_prefix}.cell.{param_name}.0.weight"

        if bias_key in state_dict:
            bias = state_dict[bias_key].float()
            # Effective value when input is zero (or very small)
            effective_at_zero = torch.sigmoid(bias)

            result[label] = {
                "bias_stats": tensor_stats(state_dict[bias_key]),
                "expected_init_bias": expected_init,
                "effective_at_zero_input": {
                    "min": float(effective_at_zero.min()),
                    "max": float(effective_at_zero.max()),
                    "mean": float(effective_at_zero.mean()),
                },
            }

            if weight_key in state_dict:
                weight = state_dict[weight_key].float()
                # Check if weight has grown significantly from zero init
                result[label]["weight_l2_norm"] = float(weight.norm(2))
                result[label]["weight_abs_max"] = float(weight.abs().max())

    return result


def analyze_polysketch_specific(state_dict: dict, prefix: str = "model.layers") -> dict:
    """Analyze polysketch-specific parameters across all layers."""
    result = {
        "has_polysketch": False,
        "layers": {}
    }

    for key in state_dict.keys():
        if "phi.poly_coeff" in key:
            result["has_polysketch"] = True
            break

    if not result["has_polysketch"]:
        return result

    # Find all layers
    layer_indices = set()
    for key in state_dict.keys():
        if ".memory.cell.phi.poly_coeff" in key:
            # Extract layer index
            parts = key.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_indices.add(int(parts[i + 1]))
                    except ValueError:
                        pass

    for layer_idx in sorted(layer_indices):
        layer_prefix = f"{prefix}.{layer_idx}.memory"
        layer_result = {}

        # poly_coeff (Taylor coefficients)
        coeff_key = f"{layer_prefix}.cell.phi.poly_coeff"
        if coeff_key in state_dict:
            coeff = state_dict[coeff_key].float()
            layer_result["poly_coeff"] = {
                "values": coeff.tolist(),
                "expected_init": [1.0, 1.0, 0.5],  # exp(x) Taylor coefficients
                "deviation_from_init": [
                    float(coeff[0] - 1.0),
                    float(coeff[1] - 1.0),
                    float(coeff[2] - 0.5),
                ],
            }

        # proj weight
        proj_key = f"{layer_prefix}.cell.phi.proj.weight"
        if proj_key in state_dict:
            layer_result["proj_weight"] = tensor_stats(state_dict[proj_key])

        result["layers"][layer_idx] = layer_result

    return result


def compare_checkpoints(ckpt1_path: str, ckpt2_path: str, output_path: str = None) -> dict:
    """Compare two checkpoints and identify significant differences."""

    print(f"Loading checkpoint 1: {ckpt1_path}")
    ckpt1 = torch.load(ckpt1_path, map_location="cpu")

    print(f"Loading checkpoint 2: {ckpt2_path}")
    ckpt2 = torch.load(ckpt2_path, map_location="cpu")

    # Handle different checkpoint formats
    if "state_dict" in ckpt1:
        state1 = ckpt1["state_dict"]
    else:
        state1 = ckpt1

    if "state_dict" in ckpt2:
        state2 = ckpt2["state_dict"]
    else:
        state2 = ckpt2

    result = {
        "ckpt1_path": str(ckpt1_path),
        "ckpt2_path": str(ckpt2_path),
        "ckpt1_keys": len(state1),
        "ckpt2_keys": len(state2),
        "analysis": {}
    }

    # Find all memory layers
    layer_indices = set()
    for key in list(state1.keys()) + list(state2.keys()):
        if ".memory." in key:
            parts = key.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_indices.add(int(parts[i + 1]))
                    except ValueError:
                        pass

    print(f"Found {len(layer_indices)} memory layers: {sorted(layer_indices)}")

    # Analyze each layer
    for layer_idx in sorted(layer_indices):
        print(f"Analyzing layer {layer_idx}...")

        result["analysis"][f"layer_{layer_idx}"] = {
            "ckpt1": analyze_memory_layer_params(state1, layer_idx),
            "ckpt2": analyze_memory_layer_params(state2, layer_idx),
            "ckpt1_hyperparams": analyze_hyperparams_effective_values(state1, layer_idx),
            "ckpt2_hyperparams": analyze_hyperparams_effective_values(state2, layer_idx),
        }

    # Polysketch-specific analysis
    result["polysketch_ckpt1"] = analyze_polysketch_specific(state1)
    result["polysketch_ckpt2"] = analyze_polysketch_specific(state2)

    # Summary of key differences
    result["summary"] = generate_summary(result)

    if output_path:
        print(f"Saving results to {output_path}")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    return result


def generate_summary(analysis: dict) -> dict:
    """Generate a summary of key differences between checkpoints."""
    summary = {
        "warnings": [],
        "key_differences": [],
    }

    # Check polysketch differences
    ps1 = analysis.get("polysketch_ckpt1", {})
    ps2 = analysis.get("polysketch_ckpt2", {})

    if ps1.get("has_polysketch") != ps2.get("has_polysketch"):
        summary["key_differences"].append({
            "type": "polysketch_mode",
            "ckpt1": ps1.get("has_polysketch"),
            "ckpt2": ps2.get("has_polysketch"),
        })

    # Check poly_coeff deviations
    if ps2.get("has_polysketch"):
        for layer_idx, layer_data in ps2.get("layers", {}).items():
            if "poly_coeff" in layer_data:
                deviations = layer_data["poly_coeff"].get("deviation_from_init", [])
                max_dev = max(abs(d) for d in deviations) if deviations else 0
                if max_dev > 0.5:
                    summary["warnings"].append({
                        "type": "poly_coeff_deviation",
                        "layer": layer_idx,
                        "max_deviation": max_dev,
                        "message": f"Layer {layer_idx} poly_coeff deviated significantly from init",
                    })

    # Check for potential gradient explosion indicators
    for layer_key, layer_data in analysis.get("analysis", {}).items():
        for ckpt_key in ["ckpt1", "ckpt2"]:
            params = layer_data.get(ckpt_key, {}).get("params", {})

            # Check for large values in key parameters
            for param_name, stats in params.items():
                if stats.get("num_nan", 0) > 0:
                    summary["warnings"].append({
                        "type": "nan_detected",
                        "checkpoint": ckpt_key,
                        "layer": layer_key,
                        "param": param_name,
                        "count": stats["num_nan"],
                    })

                if stats.get("num_inf", 0) > 0:
                    summary["warnings"].append({
                        "type": "inf_detected",
                        "checkpoint": ckpt_key,
                        "layer": layer_key,
                        "param": param_name,
                        "count": stats["num_inf"],
                    })

                if stats.get("abs_max", 0) > 1000:
                    summary["warnings"].append({
                        "type": "large_values",
                        "checkpoint": ckpt_key,
                        "layer": layer_key,
                        "param": param_name,
                        "abs_max": stats["abs_max"],
                    })

    return summary


def analyze_single_checkpoint(ckpt_path: str, output_path: str = None) -> dict:
    """Analyze a single checkpoint."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    result = {
        "ckpt_path": str(ckpt_path),
        "num_keys": len(state),
        "layers": {},
        "polysketch": analyze_polysketch_specific(state),
    }

    # Find all memory layers
    layer_indices = set()
    for key in state.keys():
        if ".memory." in key:
            parts = key.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_indices.add(int(parts[i + 1]))
                    except ValueError:
                        pass

    print(f"Found {len(layer_indices)} memory layers")

    for layer_idx in sorted(layer_indices):
        result["layers"][layer_idx] = {
            "params": analyze_memory_layer_params(state, layer_idx),
            "hyperparams": analyze_hyperparams_effective_values(state, layer_idx),
        }

    # Generate summary
    result["summary"] = {
        "has_polysketch": result["polysketch"].get("has_polysketch", False),
        "num_layers": len(layer_indices),
        "warnings": [],
    }

    # Check for issues
    for layer_idx, layer_data in result["layers"].items():
        params = layer_data.get("params", {}).get("params", {})
        for param_name, stats in params.items():
            if stats.get("num_nan", 0) > 0 or stats.get("num_inf", 0) > 0:
                result["summary"]["warnings"].append({
                    "layer": layer_idx,
                    "param": param_name,
                    "nan": stats.get("num_nan", 0),
                    "inf": stats.get("num_inf", 0),
                })

    if output_path:
        print(f"Saving results to {output_path}")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze checkpoint parameters")
    parser.add_argument("--ckpt1", type=str, help="Path to first checkpoint")
    parser.add_argument("--ckpt2", type=str, help="Path to second checkpoint (optional)")
    parser.add_argument("--output", "-o", type=str, default="checkpoint_analysis.json",
                        help="Output JSON file path")

    args = parser.parse_args()

    if not args.ckpt1:
        parser.print_help()
        sys.exit(1)

    if args.ckpt2:
        result = compare_checkpoints(args.ckpt1, args.ckpt2, args.output)
    else:
        result = analyze_single_checkpoint(args.ckpt1, args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    if "summary" in result:
        warnings = result["summary"].get("warnings", [])
        if warnings:
            print(f"\n⚠️  Found {len(warnings)} warnings:")
            for w in warnings[:10]:  # Limit output
                print(f"   - {w}")
        else:
            print("\n✅ No critical issues found")

    print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    main()
