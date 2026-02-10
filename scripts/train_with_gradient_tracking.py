#!/usr/bin/env python3
"""
Training Script with Gradient Tracking

Runs a short training session while tracking:
1. Per-layer gradient norms (forward and backward)
2. S state value ranges (min, max, mean, std)
3. phi_k, phi_q value ranges
4. Loss components over time

This helps identify exactly when and where gradient explosion occurs.

Usage:
    python scripts/train_with_gradient_tracking.py \
        --checkpoint path/to/stage1/rwkv-final.pth \
        --stage 2 \
        --max_steps 50 \
        --output gradient_tracking_results.json

Environment variables:
    GRADIENT_TRACK_LAYERS: comma-separated layer indices to track (default: all)
    GRADIENT_TRACK_EVERY_N: track every N steps (default: 1)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any, Optional
import warnings

import torch
import torch.nn as nn
from torch import Tensor

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.atlas_memory import OmegaRNNMemoryCell, PolynomialFeatureMap


class GradientTracker:
    """Tracks gradients and intermediate values during training."""

    def __init__(
        self,
        track_layers: Optional[list[int]] = None,
        track_every_n: int = 1,
    ):
        self.track_layers = track_layers
        self.track_every_n = track_every_n
        self.step = 0

        # Storage
        self.gradient_norms = defaultdict(list)  # layer_idx -> [(step, grad_norm)]
        self.param_stats = defaultdict(list)  # param_name -> [(step, stats)]
        self.state_S_stats = defaultdict(list)  # layer_idx -> [(step, S_stats)]
        self.phi_stats = defaultdict(list)  # layer_idx -> [(step, phi_k_stats, phi_q_stats)]
        self.loss_history = []
        self.hooks = []

    def should_track(self, layer_idx: int) -> bool:
        """Check if this layer should be tracked."""
        if self.track_layers is None:
            return True
        return layer_idx in self.track_layers

    def tensor_stats(self, t: Tensor, name: str = "") -> dict:
        """Compute statistics for a tensor."""
        with torch.no_grad():
            t_flat = t.float().flatten()
            t_abs = t_flat.abs()

            stats = {
                "min": float(t_flat.min()),
                "max": float(t_flat.max()),
                "mean": float(t_flat.mean()),
                "std": float(t_flat.std()) if t_flat.numel() > 1 else 0.0,
                "abs_max": float(t_abs.max()),
                "l2_norm": float(t_flat.norm(2)),
                "num_nan": int(torch.isnan(t_flat).sum()),
                "num_inf": int(torch.isinf(t_flat).sum()),
            }

            # Check for explosion
            if stats["abs_max"] > 1e6 or stats["num_nan"] > 0 or stats["num_inf"] > 0:
                stats["_warning"] = True

            return stats

    def register_hooks(self, model: nn.Module):
        """Register forward and backward hooks on the model."""
        for name, module in model.named_modules():
            # Track OmegaRNNMemoryCell layers
            if isinstance(module, OmegaRNNMemoryCell):
                # Extract layer index from name
                layer_idx = self._extract_layer_idx(name)
                if layer_idx is not None and self.should_track(layer_idx):
                    self._register_memory_hooks(module, layer_idx, name)

            # Track PolynomialFeatureMap
            if isinstance(module, PolynomialFeatureMap):
                layer_idx = self._extract_layer_idx(name)
                if layer_idx is not None and self.should_track(layer_idx):
                    self._register_phi_hooks(module, layer_idx, name)

    def _extract_layer_idx(self, name: str) -> Optional[int]:
        """Extract layer index from module name."""
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return None

    def _register_memory_hooks(self, module: OmegaRNNMemoryCell, layer_idx: int, name: str):
        """Register hooks for a memory cell."""

        def forward_hook(module, input, output):
            if self.step % self.track_every_n != 0:
                return

            # output is (retrieved, next_state)
            retrieved, next_state = output

            # Track S state
            if next_state.S is not None:
                S_stats = self.tensor_stats(next_state.S, f"layer_{layer_idx}_S")
                self.state_S_stats[layer_idx].append((self.step, S_stats))

                if S_stats.get("_warning"):
                    print(f"[WARN] Step {self.step}, Layer {layer_idx}: S state explosion detected!")
                    print(f"       S stats: abs_max={S_stats['abs_max']:.2e}, nan={S_stats['num_nan']}, inf={S_stats['num_inf']}")

        def backward_hook(module, grad_input, grad_output):
            if self.step % self.track_every_n != 0:
                return

            # Track gradient on output
            if grad_output and grad_output[0] is not None:
                grad_stats = self.tensor_stats(grad_output[0], f"layer_{layer_idx}_grad")
                self.gradient_norms[layer_idx].append((self.step, grad_stats["l2_norm"], grad_stats))

                if grad_stats.get("_warning"):
                    print(f"[WARN] Step {self.step}, Layer {layer_idx}: Gradient explosion detected!")
                    print(f"       Grad stats: abs_max={grad_stats['abs_max']:.2e}, nan={grad_stats['num_nan']}, inf={grad_stats['num_inf']}")

        self.hooks.append(module.register_forward_hook(forward_hook))
        self.hooks.append(module.register_full_backward_hook(backward_hook))

    def _register_phi_hooks(self, module: PolynomialFeatureMap, layer_idx: int, name: str):
        """Register hooks for polynomial feature map."""

        def forward_hook(module, input, output):
            if self.step % self.track_every_n != 0:
                return

            # Track input and output of phi
            if input and input[0] is not None:
                in_stats = self.tensor_stats(input[0], f"layer_{layer_idx}_phi_input")
            else:
                in_stats = {}

            out_stats = self.tensor_stats(output, f"layer_{layer_idx}_phi_output")
            self.phi_stats[layer_idx].append((self.step, in_stats, out_stats))

            if out_stats.get("_warning"):
                print(f"[WARN] Step {self.step}, Layer {layer_idx}: Phi output explosion detected!")
                print(f"       Phi output stats: abs_max={out_stats['abs_max']:.2e}")

        self.hooks.append(module.register_forward_hook(forward_hook))

    def track_loss(self, loss: Tensor):
        """Track loss value."""
        loss_val = float(loss.detach())
        self.loss_history.append((self.step, loss_val))

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[ERROR] Step {self.step}: Loss is NaN/Inf!")

    def track_param_gradients(self, model: nn.Module):
        """Track gradients on model parameters after backward."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = float(param.grad.norm(2))
                self.param_stats[name].append((self.step, {
                    "grad_norm": grad_norm,
                    "grad_abs_max": float(param.grad.abs().max()),
                    "param_abs_max": float(param.abs().max()),
                }))

    def increment_step(self):
        """Increment the step counter."""
        self.step += 1

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_results(self) -> dict:
        """Get all tracking results."""
        return {
            "gradient_norms": {k: v for k, v in self.gradient_norms.items()},
            "state_S_stats": {k: v for k, v in self.state_S_stats.items()},
            "phi_stats": {k: v for k, v in self.phi_stats.items()},
            "param_stats": {k: v for k, v in self.param_stats.items()},
            "loss_history": self.loss_history,
            "total_steps": self.step,
        }

    def print_summary(self):
        """Print a summary of tracking results."""
        print("\n" + "=" * 60)
        print("GRADIENT TRACKING SUMMARY")
        print("=" * 60)

        print(f"\nTotal steps: {self.step}")
        print(f"Loss history: start={self.loss_history[0][1]:.4f}, end={self.loss_history[-1][1]:.4f}")

        # Find layers with largest gradient norms
        print("\nTop 5 layers by max gradient norm:")
        layer_max_grads = []
        for layer_idx, grad_data in self.gradient_norms.items():
            if grad_data:
                max_grad = max(g[1] for g in grad_data)
                layer_max_grads.append((layer_idx, max_grad))

        for layer_idx, max_grad in sorted(layer_max_grads, key=lambda x: -x[1])[:5]:
            print(f"  Layer {layer_idx}: max_grad_norm={max_grad:.2e}")

        # Find layers with S state issues
        print("\nLayers with S state issues (abs_max > 100):")
        for layer_idx, S_data in self.state_S_stats.items():
            issues = [(step, stats) for step, stats in S_data if stats.get("abs_max", 0) > 100]
            if issues:
                print(f"  Layer {layer_idx}: {len(issues)} steps with abs_max > 100")
                if issues:
                    print(f"    First issue at step {issues[0][0]}: abs_max={issues[0][1]['abs_max']:.2e}")


def run_gradient_tracking(
    checkpoint_path: str,
    stage: int = 2,
    max_steps: int = 50,
    output_path: str = "gradient_tracking_results.json",
    config_path: str = None,
):
    """Run training with gradient tracking."""

    # Import training infrastructure
    try:
        from train import Trainer, load_config
        from models.atlasqwen2 import AtlasQwen2LMHeadModel
    except ImportError as e:
        print(f"Error importing training infrastructure: {e}")
        print("Make sure you're running from the RADLADS-atlas directory")
        sys.exit(1)

    # Load config
    if config_path is None:
        config_path = "configs/atlasqwen0b5.yaml"

    print(f"Loading config from {config_path}")
    config = load_config(config_path)

    # Override for gradient tracking
    config.train.my_exit_tokens = max_steps * config.train.micro_bsz * config.model.ctx_len
    config.train.devices = 1
    config.train.num_nodes = 1
    config.train.accumulate_grad_batches = 1

    print(f"Loading checkpoint from {checkpoint_path}")

    # Load model
    model = AtlasQwen2LMHeadModel.from_pretrained(
        checkpoint_path,
        config=config.model,
    )

    # Set up gradient tracker
    track_layers = None
    if os.environ.get("GRADIENT_TRACK_LAYERS"):
        track_layers = [int(x) for x in os.environ["GRADIENT_TRACK_LAYERS"].split(",")]

    track_every_n = int(os.environ.get("GRADIENT_TRACK_EVERY_N", "1"))

    tracker = GradientTracker(
        track_layers=track_layers,
        track_every_n=track_every_n,
    )

    tracker.register_hooks(model)

    print(f"Starting gradient tracking for {max_steps} steps...")
    print(f"Tracking layers: {track_layers or 'all'}")
    print(f"Tracking every {track_every_n} steps")

    # Simple training loop for testing
    model.train()
    model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Create dummy data for testing
    batch_size = config.train.micro_bsz
    ctx_len = config.model.ctx_len
    vocab_size = config.model.vocab_size

    for step in range(max_steps):
        # Generate random input
        input_ids = torch.randint(0, vocab_size, (batch_size, ctx_len), device="cuda")
        labels = input_ids.clone()

        # Forward pass
        try:
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            tracker.track_loss(loss)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Track gradients
            tracker.track_param_gradients(model)

            # Gradient clipping (to prevent explosion during tracking)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        except Exception as e:
            print(f"[ERROR] Step {step}: {e}")
            break

        tracker.increment_step()

        if step % 10 == 0:
            print(f"Step {step}/{max_steps}, Loss: {loss.item():.4f}")

    # Clean up
    tracker.remove_hooks()

    # Get and save results
    results = tracker.get_results()
    results["config"] = {
        "checkpoint_path": checkpoint_path,
        "stage": stage,
        "max_steps": max_steps,
        "track_layers": track_layers,
        "track_every_n": track_every_n,
    }

    print(f"\nSaving results to {output_path}")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    tracker.print_summary()

    return results


def main():
    parser = argparse.ArgumentParser(description="Train with gradient tracking")
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                        help="Path to checkpoint")
    parser.add_argument("--stage", type=int, default=2,
                        help="Training stage (2 or 3)")
    parser.add_argument("--max_steps", type=int, default=50,
                        help="Maximum training steps")
    parser.add_argument("--output", "-o", type=str, default="gradient_tracking_results.json",
                        help="Output JSON file")
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML file")

    args = parser.parse_args()

    run_gradient_tracking(
        checkpoint_path=args.checkpoint,
        stage=args.stage,
        max_steps=args.max_steps,
        output_path=args.output,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
