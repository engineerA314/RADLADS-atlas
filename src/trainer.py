import os, math, time, datetime, subprocess, json
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_only

from configs import TrainerCLI_Config

from src.logger import print0 as print
import deepspeed.utils

class train_callback(pl.Callback):
    def __init__(self, config:TrainerCLI_Config):
        super().__init__()
        self.config = config
        # For full (optimizer+step) checkpointing throttles
        self._last_full_ckpt_step = -1
        # Prevent repeated rwkv-final.pth saves when global_step doesn't advance (e.g., grad accumulation)
        self._last_magic_prime_save_step = -1
        # Early stopping state (step-based)
        self._early_last_checked_step = -1
        self._early_ema = None
        self._early_best = None
        self._early_bad_count = 0
        self._early_target_good = 0

    def _gcs_default_enable(self) -> str:
        gcs_bucket_env = os.environ.get("GCS_BUCKET", "").strip()
        return "1" if gcs_bucket_env.startswith("gs://") else "0"

    def _stage_label_for_gcs(self) -> str:
        """
        Use proj_suffix when available for uniqueness.
        Examples:
          - atlas-stage1        -> stage1
          - atlas-stage2        -> stage2
          - atlas-stage3-ctx512 -> stage3-ctx512
        """
        suf = (str(getattr(self.config.train, "proj_suffix", "") or "")).strip().lower()
        if suf:
            return suf[len("atlas-"):] if suf.startswith("atlas-") else suf
        s = getattr(self.config.train, "attention_distillation_stage", None)
        if s == 1:
            return "stage1"
        if s == 2:
            return "stage2"
        return "train"

    def _maybe_save_full_resume_checkpoint(self, trainer, pl_module) -> None:
        """
        - full resume: full-resume.ckpt + meta (optimizer+step)
        """
        default_enable = self._gcs_default_enable()
        stage_label = self._stage_label_for_gcs()

        # Defaults (when GCS is configured):
        # - full checkpoint every 500 optimizer steps (heavy but "proper resume")
        try:
            default_full_steps = int(os.environ.get("FULL_RESUME_EVERY_N_STEPS", "500" if default_enable == "1" else "0"))
        except Exception:
            default_full_steps = 0
        default_full_steps = max(0, default_full_steps)

        enable_full = str(os.environ.get("ENABLE_FULL_RESUME_CKPT", default_enable)).strip() == "1"

        # --- full resume ---
        if not (enable_full and default_full_steps > 0):
            return

        # Lightning's progress bar often uses "global_step + 1" semantics.
        # Use that for trigger alignment, but store the underlying global_step in meta.
        base_step = int(getattr(trainer, "global_step", 0) or 0)
        trigger_step = base_step + 1
        if trigger_step <= 0:
            return
        if (trigger_step % default_full_steps) != 0:
            return
        if trigger_step == self._last_full_ckpt_step:
            return
        self._last_full_ckpt_step = trigger_step

        full_ckpt_path = os.path.join(self.config.runtime.proj_path, "full-resume.ckpt")
        full_meta_path = os.path.join(self.config.runtime.proj_path, "full-resume.meta.json")

        try:
            # IMPORTANT:
            # - For DeepSpeed checkpoints, all ranks may need to participate.
            # - For non-DeepSpeed strategies, call from rank0 only to avoid file contention.
            strategy_name = str(getattr(getattr(self.config, "train", None), "strategy", "") or "").lower()
            is_deepspeed = "deepspeed" in strategy_name
            if (not is_deepspeed) and (not trainer.is_global_zero):
                return

            try:
                trainer.save_checkpoint(full_ckpt_path, weights_only=False)
            except TypeError:
                trainer.save_checkpoint(full_ckpt_path)

            # Best-effort barrier before rank0 uploads.
            try:
                if is_deepspeed and torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.barrier()
            except Exception:
                pass

            if trainer.is_global_zero:
                ckpt_is_dir = bool(os.path.isdir(full_ckpt_path))
                meta = {
                    "exp_id": os.environ.get("EXP_ID", "").strip(),
                    "run_id": os.environ.get("RUN_ID", "").strip(),
                    "proj_suffix": str(getattr(self.config.train, "proj_suffix", "") or ""),
                    "attention_distillation_stage": int(getattr(self.config.train, "attention_distillation_stage", -999) or -999),
                    "lightning_global_step": base_step,
                    "lightning_current_epoch": int(getattr(trainer, "current_epoch", 0) or 0),
                    "ctx_len": int(getattr(self.config.model, "ctx_len", 0) or 0),
                    "global_step_bsz": int(getattr(self.config.runtime, "global_step_bsz", 0) or 0),
                    "epoch_global_steps": int(getattr(self.config.runtime, "epoch_global_steps", 0) or 0),
                    "ckpt_is_dir": ckpt_is_dir,
                    "saved_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
                }
                with open(full_meta_path, "w") as f:
                    json.dump(meta, f, ensure_ascii=False)
                    f.write("\n")

                gcs_bucket = os.environ.get("GCS_BUCKET", "").strip()
                gcs_prefix = os.environ.get("GCS_PREFIX", "").strip().strip("/")
                exp_id = meta["exp_id"]
                if gcs_bucket.startswith("gs://") and gcs_prefix and exp_id:
                    print(f"[full-resume] saving checkpoint at step={trigger_step} (global_step={base_step}) → uploading to GCS")
                    gcs_dir = f"{gcs_bucket.rstrip('/')}/{gcs_prefix}/{exp_id}/_latest/full_resume/{stage_label}"
                    if ckpt_is_dir:
                        upload_cmd = (
                            f"gsutil -m rsync -d -r '{full_ckpt_path}' '{gcs_dir}/full-resume.ckpt' && "
                            f"gsutil cp -q '{full_meta_path}' '{gcs_dir}/full-resume.meta.json'"
                        )
                    else:
                        upload_cmd = (
                            f"gsutil cp -q '{full_ckpt_path}' '{gcs_dir}/full-resume.ckpt' && "
                            f"gsutil cp -q '{full_meta_path}' '{gcs_dir}/full-resume.meta.json'"
                        )
                    subprocess.Popen(["bash", "-lc", upload_cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            if trainer.is_global_zero:
                print("Error while saving/uploading full-resume checkpoint:", e)

    def _force_save_full_resume_checkpoint(self, trainer, pl_module, reason: str) -> None:
        """
        Save a full Lightning checkpoint immediately (ignores step modulo).
        Intended for early-stop / finalization paths.
        """
        default_enable = self._gcs_default_enable()
        stage_label = self._stage_label_for_gcs()

        enable_full = str(os.environ.get("ENABLE_FULL_RESUME_CKPT", default_enable)).strip() == "1"
        if not enable_full:
            return

        base_step = int(getattr(trainer, "global_step", 0) or 0)
        full_ckpt_path = os.path.join(self.config.runtime.proj_path, "full-resume.ckpt")
        full_meta_path = os.path.join(self.config.runtime.proj_path, "full-resume.meta.json")

        try:
            strategy_name = str(getattr(getattr(self.config, "train", None), "strategy", "") or "").lower()
            is_deepspeed = "deepspeed" in strategy_name
            if (not is_deepspeed) and (not trainer.is_global_zero):
                return

            try:
                trainer.save_checkpoint(full_ckpt_path, weights_only=False)
            except TypeError:
                trainer.save_checkpoint(full_ckpt_path)

            # Best-effort barrier before rank0 uploads.
            try:
                if is_deepspeed and torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.barrier()
            except Exception:
                pass

            if trainer.is_global_zero:
                ckpt_is_dir = bool(os.path.isdir(full_ckpt_path))
                meta = {
                    "exp_id": os.environ.get("EXP_ID", "").strip(),
                    "run_id": os.environ.get("RUN_ID", "").strip(),
                    "proj_suffix": str(getattr(self.config.train, "proj_suffix", "") or ""),
                    "attention_distillation_stage": int(getattr(self.config.train, "attention_distillation_stage", -999) or -999),
                    "lightning_global_step": base_step,
                    "lightning_current_epoch": int(getattr(trainer, "current_epoch", 0) or 0),
                    "ctx_len": int(getattr(self.config.model, "ctx_len", 0) or 0),
                    "global_step_bsz": int(getattr(self.config.runtime, "global_step_bsz", 0) or 0),
                    "epoch_global_steps": int(getattr(self.config.runtime, "epoch_global_steps", 0) or 0),
                    "ckpt_is_dir": ckpt_is_dir,
                    "saved_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
                    "reason": str(reason),
                }
                with open(full_meta_path, "w") as f:
                    json.dump(meta, f, ensure_ascii=False)
                    f.write("\n")

                # NOTE:
                # We intentionally do NOT upload to GCS here in the early-stop path.
                # Uploading a DeepSpeed directory checkpoint can take minutes; if we do it here,
                # other ranks may have already proceeded to a barrier and will deadlock.
                # The early-stop handler will upload synchronously on rank0 *after* destroying
                # the process group (safe, no further collectives).
        except Exception as e:
            if trainer.is_global_zero:
                print("Error while force-saving/uploading full-resume checkpoint:", e)

    def _upload_full_resume_to_gcs_sync(self) -> None:
        """
        Synchronous upload of the most recent full-resume checkpoint + meta from proj_path.
        Intended to run on rank0 AFTER distributed teardown (no collectives).
        """
        try:
            gcs_bucket = os.environ.get("GCS_BUCKET", "").strip()
            gcs_prefix = os.environ.get("GCS_PREFIX", "").strip().strip("/")
            exp_id = os.environ.get("EXP_ID", "").strip()
            if not (gcs_bucket.startswith("gs://") and gcs_prefix and exp_id):
                return

            stage_label = self._stage_label_for_gcs()
            full_ckpt_path = os.path.join(self.config.runtime.proj_path, "full-resume.ckpt")
            full_meta_path = os.path.join(self.config.runtime.proj_path, "full-resume.meta.json")
            if not os.path.exists(full_meta_path):
                print("[full-resume] meta missing locally; skipping upload:", full_meta_path)
                return

            ckpt_is_dir = bool(os.path.isdir(full_ckpt_path))
            gcs_dir = f"{gcs_bucket.rstrip('/')}/{gcs_prefix}/{exp_id}/_latest/full_resume/{stage_label}"
            gcs_meta = f"{gcs_dir}/full-resume.meta.json"
            gcs_ckpt = f"{gcs_dir}/full-resume.ckpt"

            # Upload meta FIRST so downstream resume logic can correctly detect dir/file.
            # (Also makes verification deterministic.)
            print(f"[full-resume] uploading meta → {gcs_meta}", flush=True)
            # Avoid shell quoting issues by invoking gsutil directly.
            # NOTE: some gsutil builds are picky about option parsing; if -q fails, retry without it.
            r = subprocess.run(["gsutil", "cp", "-q", full_meta_path, gcs_meta], check=False)
            if r.returncode != 0:
                print(f"[full-resume] meta upload failed with -q (rc={r.returncode}); retrying without -q", flush=True)
                subprocess.run(["gsutil", "cp", full_meta_path, gcs_meta], check=False)

            if ckpt_is_dir:
                print(f"[full-resume] uploading dir checkpoint → {gcs_ckpt}/", flush=True)
                subprocess.run(["gsutil", "-m", "rsync", "-d", "-r", full_ckpt_path, gcs_ckpt], check=False)
            else:
                print(f"[full-resume] uploading file checkpoint → {gcs_ckpt}", flush=True)
                r2 = subprocess.run(["gsutil", "cp", "-q", full_ckpt_path, gcs_ckpt], check=False)
                if r2.returncode != 0:
                    print(f"[full-resume] ckpt upload failed with -q (rc={r2.returncode}); retrying without -q", flush=True)
                    subprocess.run(["gsutil", "cp", full_ckpt_path, gcs_ckpt], check=False)
        except Exception as e:
            print("[full-resume] ERROR during sync upload:", e)

    def _maybe_early_stop(self, trainer, pl_module, loss_value: float) -> tuple[bool, str]:
        """
        Step-based early stopping driven by an EMA of the (optionally clipped) train loss.
        Returns (should_stop, reason_string).
        """
        cfg = getattr(self.config, "train", None)
        if cfg is None:
            return (False, "")
        if not bool(getattr(cfg, "early_stop", False)):
            return (False, "")

        step = int(getattr(trainer, "global_step", 0) or 0) + 1
        if step <= 0:
            return (False, "")
        if step == self._early_last_checked_step:
            return (False, "")
        self._early_last_checked_step = step

        min_steps = max(0, int(getattr(cfg, "early_stop_min_steps", 0) or 0))
        if step < min_steps:
            return (False, "")

        # Parameters
        try:
            alpha = float(getattr(cfg, "early_stop_ema_alpha", 0.05))
        except Exception:
            alpha = 0.05
        alpha = max(0.0001, min(1.0, alpha))

        try:
            min_delta = float(getattr(cfg, "early_stop_min_delta", 0.0))
        except Exception:
            min_delta = 0.0
        min_delta = max(0.0, min_delta)

        patience_steps = max(0, int(getattr(cfg, "early_stop_patience_steps", 0) or 0))
        target_pat = max(0, int(getattr(cfg, "early_stop_target_patience_steps", 0) or 0))

        target_loss = getattr(cfg, "early_stop_target_loss", None)
        try:
            target_loss = None if target_loss is None else float(target_loss)
        except Exception:
            target_loss = None

        clip = getattr(cfg, "early_stop_loss_clip", None)
        try:
            clip = None if clip is None else float(clip)
        except Exception:
            clip = None

        x = float(loss_value)
        if clip is not None:
            # Robustness to rare extreme spikes (keeps EMA stable so plateau detection works).
            x = min(x, clip)

        if self._early_ema is None:
            self._early_ema = x
        else:
            self._early_ema = (1.0 - alpha) * float(self._early_ema) + alpha * x

        ema = float(self._early_ema)

        if self._early_best is None:
            self._early_best = ema
            self._early_bad_count = 0
        else:
            if ema < (float(self._early_best) - min_delta):
                self._early_best = ema
                self._early_bad_count = 0
            else:
                self._early_bad_count += 1

        # "Target reached" stopping (optional): EMA <= target_loss for N consecutive steps
        if target_loss is not None and target_pat > 0:
            if ema <= target_loss:
                self._early_target_good += 1
            else:
                self._early_target_good = 0
            if self._early_target_good >= target_pat:
                return (True, f"target_loss_reached: ema={ema:.6g} <= {target_loss} for {target_pat} steps")

        # "Plateau" stopping (optional): no improvement for N steps
        if patience_steps > 0 and self._early_bad_count >= patience_steps:
            return (True, f"plateau: ema={ema:.6g} best={float(self._early_best):.6g} bad_steps={self._early_bad_count} patience={patience_steps}")

        return (False, "")

    def on_after_backward(self, trainer, pl_module) -> None:
        # Stage 2 NaN debug: print per-layer parameter gradient absmax (gradient explosion check).
        # Usage: SKIP_NAN_CHECK=1 STAGE2_DEBUG_GRAD_ABSMAX=1 bash gcp_test/debug_nan_stage2.sh
        if os.environ.get("STAGE2_DEBUG_GRAD_ABSMAX", "").strip().lower() not in ("1", "true", "yes"):
            return
        if not trainer.is_global_zero:
            return
        try:
            model = getattr(pl_module, "model", pl_module)
            layer_absmax = {}
            for name, p in model.named_parameters():
                if p.grad is None:
                    try:
                        grad = deepspeed.utils.safe_get_full_grad(p)
                    except Exception:
                        grad = None
                else:
                    grad = p.grad
                if grad is None:
                    continue
                g = grad.detach().float()
                if not torch.isfinite(g).any():
                    absmax = float("nan")
                else:
                    absmax = float(g.abs().max().item())
                # Group by layer: model.layers.0.* -> 0, model.layers.23.* -> 23
                parts = name.split(".")
                layer_idx = None
                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                        layer_idx = int(parts[i + 1])
                        break
                if layer_idx is not None:
                    if layer_idx not in layer_absmax:
                        layer_absmax[layer_idx] = absmax
                    else:
                        try:
                            cur = layer_absmax[layer_idx]
                            if math.isfinite(cur) and math.isfinite(absmax):
                                layer_absmax[layer_idx] = max(cur, absmax)
                            elif math.isfinite(absmax):
                                layer_absmax[layer_idx] = absmax
                        except Exception:
                            layer_absmax[layer_idx] = absmax
            if layer_absmax:
                print("[STAGE2_DEBUG_GRAD_ABSMAX] per-layer max param grad absmax (backward order ~ layer 23 -> 0):")
                for idx in sorted(layer_absmax.keys(), reverse=True):
                    print(f"  layer{idx} ~ {layer_absmax[idx]:.6e}")
        except Exception as e:
            print(f"[STAGE2_DEBUG_GRAD_ABSMAX] error: {e}")

    def on_predict_start(self, trainer, pl_module) -> None:
        pl_module.load_weights()

    def on_train_start(self, trainer, pl_module) -> None:
        # If we are not resuming from a full Lightning checkpoint, keep epoch_begin behavior.
        if trainer.global_step == 0 and self.config.train.epoch_begin > 0:
            trainer.fit_loop.epoch_progress.current.ready = self.config.train.epoch_begin
            trainer.fit_loop.epoch_progress.current.completed = self.config.train.epoch_begin
        # Reset early-stop state
        self._early_last_checked_step = -1
        self._early_ema = None
        self._early_best = None
        self._early_bad_count = 0
        self._early_target_good = 0
        self._ensure_logging_state(trainer)
        self._init_wandb(trainer, pl_module)

    def _ensure_logging_state(self, trainer) -> None:
        if not trainer.is_global_zero:
            return
        config = self.config
        if not hasattr(trainer, "my_loss_sum"):
            trainer.my_loss_sum = 0
        if not hasattr(trainer, "my_loss_count"):
            trainer.my_loss_count = 0
        if not hasattr(trainer, "my_time_ns"):
            trainer.my_time_ns = time.time_ns()
        try:
            if (not hasattr(trainer, "my_log")) or trainer.my_log is None or trainer.my_log.closed:
                trainer.my_log = open(config.runtime.proj_path + "/train_log.txt", "a")
                run_tag = "RESUME RUN" if trainer.global_step > 0 else "NEW RUN"
                trainer.my_log.write(f"{run_tag} {config.runtime.my_timestamp}\n")
                trainer.my_log.flush()
        except Exception:
            pass

    def _init_wandb(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return
        config = self.config
        if len(config.train.wandb) == 0:
            return
        if hasattr(trainer, "my_wandb") and trainer.my_wandb is not None:
            return
        try:
            print("Login to wandb...")
            import wandb
            exp_id = os.environ.get("EXP_ID", "").strip()
            run_id = os.environ.get("RUN_ID", "").strip()

            def _short_bool(v) -> str:
                return "1" if str(v).lower() in ("1", "true", "yes", "y", "t") else "0"

            def _get_hf_meta(m) -> tuple[str, str] | None:
                """
                Return (pretty_name, ld_summary) if this looks like an HF model.
                """
                cfg = getattr(m, "config", None)
                if cfg is None:
                    return None
                # HF config usually exposes these
                n_layer = getattr(cfg, "num_hidden_layers", None)
                n_embd = getattr(cfg, "hidden_size", None)
                # Prefer the configured hf_path for readability
                hf_path = getattr(config.model, "hf_path", "") or ""
                base = hf_path.split("/")[-1] if hf_path else (getattr(cfg, "_name_or_path", "") or getattr(cfg, "name_or_path", "") or "")
                base = str(base).strip() or "hf-model"
                ld = []
                if n_layer is not None:
                    ld.append(f"L{int(n_layer)}")
                if n_embd is not None:
                    ld.append(f"D{int(n_embd)}")
                return (base, " ".join(ld).strip())

            def _stage_label() -> str:
                # Highest-signal: explicit distillation stage if present.
                s = getattr(config.train, "attention_distillation_stage", None)
                if s == 1:
                    return "stage1-align"
                if s == 2:
                    return "stage2-kl"
                # Fall back to suffix if provided in yaml.
                suf = (getattr(config.train, "proj_suffix", "") or "").strip()
                return suf or "train"

            # Build a meaning-first run name from *actual* model + knobs.
            # This avoids legacy defaults like tmix=x060, n_layer=6, n_embd=512 leaking into W&B.
            ts = str(config.runtime.my_timestamp)
            stage = _stage_label()

            # Model meta (HF)
            hf_meta = _get_hf_meta(pl_module.model)
            model_part = ""
            if hf_meta is not None:
                base, ld = hf_meta
                model_part = base + (f" {ld}" if ld else "")

            ctx = getattr(config.model, "ctx_len", None)
            ctx_part = f"ctx{int(ctx)}" if ctx is not None else ""

            # Atlas knobs (even in stage1 HF patching, these live in config.model)
            knobs = []
            for k, label in [
                ("atlas_variant", "atlas"),
                ("omega_window", "ow"),
                ("poly_degree", "poly"),
                ("qk_norm", "qknorm"),
                ("qkv_conv_kernel", "qkvconv"),
                ("use_rope", "rope"),
                ("use_groupnorm", "gn"),
            ]:
                v = getattr(config.model, k, None)
                if v is None:
                    continue
                if k in ("qk_norm", "use_rope", "use_groupnorm"):
                    knobs.append(f"{label}{_short_bool(v)}")
                elif k == "atlas_variant":
                    knobs.append(f"{label}={v}")
                else:
                    knobs.append(f"{label}={v}")

            # Example:
            # 2025-12-26-02-05-22 | setup-smoke-stage1/20251226-015909 | stage1-align | Qwen2.5-0.5B-Instruct L24 D896 ctx512 | atlas=lmm ow=16 poly=2 qknorm0 qkvconv=None rope0 gn0
            parts = [ts]
            if exp_id or run_id:
                parts.append(f"{exp_id}/{run_id}".strip("/"))
            parts.append(stage)
            if model_part:
                parts.append(model_part)
            if ctx_part:
                parts.append(ctx_part)
            if knobs:
                parts.append(" ".join(knobs))
            wandb_name = " | ".join([p for p in parts if p])

            def _short_tag(tag: str, max_len: int = 64) -> str:
                if len(tag) <= max_len:
                    return tag
                import hashlib
                suffix = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:6]
                return f"{tag[:max_len - 7]}~{suffix}"

            tags = []
            if exp_id:
                tags.append(_short_tag(f"exp:{exp_id}"))
            if run_id:
                tags.append(_short_tag(f"run:{run_id}"))
            tags.append(_short_tag(f"stage:{stage}"))
            if ctx_part:
                tags.append(_short_tag(ctx_part))
            if str(os.environ.get("SMOKE_TEST", "")).strip() == "1":
                tags.append("smoke")

            env_mode = os.environ.get("WANDB_MODE", "").strip()
            env_disabled = os.environ.get("WANDB_DISABLED", "").strip()
            has_key = "set" if os.environ.get("WANDB_API_KEY") else "empty"
            print(
                f"[wandb] project={config.train.wandb} mode={env_mode} "
                f"disabled={env_disabled} api_key={has_key} "
                f"exp_id={exp_id} run_id={run_id}"
            )
            init_kwargs = dict(
                project=config.train.wandb,
                name=wandb_name,
                config=config,
                save_code=False,
                tags=tags,
            )
            if exp_id:
                init_kwargs["group"] = exp_id
            wandb.init(**init_kwargs)
            trainer.my_wandb = wandb
        except Exception as e:
            try:
                print(f"[wandb] init failed: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
            except Exception:
                pass
        try:
            step_csv_path = os.path.join(config.runtime.proj_path, "train_steps.csv")
            if (not hasattr(trainer, "my_step_log")) or trainer.my_step_log is None or trainer.my_step_log.closed:
                is_new = not os.path.exists(step_csv_path) or os.path.getsize(step_csv_path) == 0
                trainer.my_step_log = open(step_csv_path, "a")
                if is_new:
                    trainer.my_step_log.write(
                        "timestamp,real_global_step,tokens,loss,lr,lr2,weight_decay,kt_per_s,ctx_len,global_step_bsz\n"
                    )
                trainer.my_step_log.flush()
        except Exception:
            pass

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        config = self.config
        # if config.cuda_cleanup > 0:
        #     torch.cuda.empty_cache()

        # "Real" global step is used for schedule/logging continuity across restarts.
        # This is NOT the same as Lightning's trainer.global_step (which resets if we don't
        # resume from a full framework checkpoint).
        real_global_step = None
        try:
            real_global_step = int(pl_module.get_real_global_step())
        except Exception:
            real_global_step = int(getattr(trainer, "global_step", 0))

        # LR schedule
        if config.runtime.epoch_count == 0 or config.train.my_exit_tokens == 0:
            lr = config.train.lr_init
            lr2 = config.train.lr2_init
            lr_progress = 0.0
        else:
            lr_progress = pl_module.get_lr_progress()

            def calc_lr(lr_decay_type,lr_progress, lr_init, lr_final):
                match lr_decay_type:
                    case 'linear':
                        init_amt = 1.0 - lr_progress
                        return lr_final + (lr_init - lr_final) * init_amt
                    case 'exp':
                        return lr_init * math.exp(math.log(lr_final / lr_init) * lr_progress)
                    case 'cos':
                        init_amt = math.cos(math.pi / 2 * lr_progress)
                        return lr_final + (lr_init - lr_final) * init_amt
                    case 'oneminussqrt':
                        init_amt = 1.0 - math.sqrt(lr_progress)
                        return lr_final + (lr_init - lr_final) * init_amt
                    case 'none':
                        # No decay: keep lr constant at lr_init
                        return lr_init
                    case _:
                        print("bad lr_decay_type specified")
                        exit()
            
            lr = calc_lr(config.train.lr_decay_type, lr_progress, config.train.lr_init, config.train.lr_final)
            lr2 = calc_lr(config.train.lr_decay_type, lr_progress, config.train.lr_init if config.train.lr2_init<0 else config.train.lr2_init, config.train.lr_final if config.train.lr2_final<0 else config.train.lr2_final)

            real_progress = pl_module.get_real_progress()
            if real_progress >= 1:
                # IMPORTANT: this repo uses `exit(0)` to end training immediately.
                # Ensure we flush/close logs first, otherwise short runs may lose buffered rows
                # (e.g., train_steps.csv header is written but data rows never flush).
                if trainer.is_global_zero:
                    try:
                        if hasattr(trainer, "my_step_log") and trainer.my_step_log is not None:
                            trainer.my_step_log.flush()
                            trainer.my_step_log.close()
                    except Exception:
                        pass
                    try:
                        if hasattr(trainer, "my_log") and trainer.my_log is not None:
                            trainer.my_log.flush()
                            trainer.my_log.close()
                    except Exception:
                        pass
                pl_module.save_weights(f"{config.runtime.proj_path}/rwkv-final.pth")
                print("!!!TRAINING COMPLETE!!!")
                exit(0)

        if trainer.global_step < config.train.warmup_steps:
            #lr = lr * (0.2 + 0.8 * trainer.global_step / config.train.warmup_steps)
            # IMPORTANT: use the "real" step so warmup doesn't restart incorrectly on restarts.
            # (With full checkpoint resume, Lightning restores global_step anyway, but this keeps
            # behaviour consistent across both resume mechanisms.)
            warm_step = real_global_step
            lr = lr * (0.01 + .99 * warm_step / config.train.warmup_steps)
            lr2 = lr2 * (0.01 + .99 * warm_step / config.train.warmup_steps)

        if config.train.weight_decay_final > 0:
            wd_now = config.train.weight_decay * math.exp(math.log(config.train.weight_decay_final / config.train.weight_decay) * lr_progress)
        else:
            wd_now = config.train.weight_decay

        for param_group in trainer.optimizers[0].param_groups:
            if param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_now
            if config.train.layerwise_lr > 0:
                if param_group["name"] == "lr_2":
                    param_group["lr"] = lr2 * param_group["my_lr_scale"]
                else:
                    param_group["lr"] = lr * param_group["my_lr_scale"]
                # print(param_group["lr"], param_group["my_lr_scale"])
            else:
                param_group["lr"] = lr

        trainer.my_lr = lr
        trainer.my_lr2 = lr2
        trainer.my_wd = wd_now
        # rank_zero_info(f"{real_global_step} {lr}")

        if trainer.global_step == 0 and batch_idx == 0:
            if trainer.is_global_zero:  # logging
                trainer.my_loss_sum = 0
                trainer.my_loss_count = 0
                trainer.my_log = open(config.runtime.proj_path + "/train_log.txt", "a")
                trainer.my_log.write(f"NEW RUN {config.runtime.my_timestamp}\n{vars(self.config)}\n")
                # Step-level metrics log (raw timeseries).
                # This is intentionally plain CSV so it can be plotted / uploaded to GCS easily.
                step_csv_path = os.path.join(config.runtime.proj_path, "train_steps.csv")
                is_new = not os.path.exists(step_csv_path) or os.path.getsize(step_csv_path) == 0
                trainer.my_step_log = open(step_csv_path, "a")
                if is_new:
                    trainer.my_step_log.write(
                        "timestamp,real_global_step,tokens,loss,lr,lr2,weight_decay,kt_per_s,ctx_len,global_step_bsz\n"
                    )
                try:
                    print(f"\n{trainer.strategy.config}\n")
                    trainer.my_log.write(f"{trainer.strategy.config}\n")
                except:
                    pass
                trainer.my_log.flush()
                trainer.my_step_log.flush()
                self._init_wandb(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        config = self.config
        tokens_per_micro_step = config.model.ctx_len * config.runtime.global_step_bsz / config.train.accumulate_grad_batches
        real_global_step = pl_module.get_real_global_step()
        # Keep a loss scalar available on ALL ranks (needed for synchronized early-stop decisions).
        # NOTE: Lightning may pass `outputs` as a dict like {"loss": tensor} even if training_step returns a tensor.
        micro_step_loss = None
        if trainer.is_global_zero:  # logging
            self._ensure_logging_state(trainer)
            t_now = time.time_ns()
            kt_s = 0
            try:
                t_cost = (t_now - trainer.my_time_ns) / 1e9
                kt_s = tokens_per_micro_step / t_cost / 1000
                self.log("REAL it/s", 1.0 / t_cost, prog_bar=True, on_step=True)
                self.log("Kt/s", kt_s, prog_bar=True, on_step=True)
            except:
                pass
            trainer.my_time_ns = t_now
            if pl.__version__[0]=='2':
                try:
                    micro_step_loss = outputs["loss"] * trainer.accumulate_grad_batches
                except Exception:
                    micro_step_loss = outputs * trainer.accumulate_grad_batches
            else:
                micro_step_loss = trainer.my_loss_all.float().mean().item() * trainer.accumulate_grad_batches
            trainer.my_loss_sum += micro_step_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
            self.log("lr2", trainer.my_lr2, prog_bar=True, on_step=True)
            # self.log("s", real_global_step, prog_bar=True, on_step=True)

            # Raw timeseries CSV (one row per optimizer step).
            # Note: this is per *micro step* as seen by Lightning.
            # In practice this is good enough for ablation comparisons and plotting.
            if hasattr(trainer, "my_step_log") and trainer.my_step_log is not None:
                ts = time.time()
                tokens = real_global_step * tokens_per_micro_step
                ctx_len = config.model.ctx_len
                gbsz = config.runtime.global_step_bsz
                trainer.my_step_log.write(
                    f"{ts},{int(real_global_step)},{int(tokens)},{float(micro_step_loss)},{float(trainer.my_lr)},{float(trainer.my_lr2)},{float(trainer.my_wd)},{float(kt_s)},{int(ctx_len)},{int(gbsz)}\n"
                )
                # flush occasionally to avoid data loss on preemption
                if (trainer.global_step + 1) % max(1, config.train.log_every_n_steps) == 0:
                    trainer.my_step_log.flush()

            # if len(config.train.wandb) > 0:
            if hasattr(trainer, "my_wandb") and trainer.my_wandb is not None:
                # Log only on the main process.
                # Also throttle to reduce overhead.
                # - Default: log every step (WANDB_LOG_EVERY_N_STEPS defaults to 1).
                try:
                    wandb_every = int(os.environ.get("WANDB_LOG_EVERY_N_STEPS", "1"))
                except Exception:
                    wandb_every = 1
                wandb_every = max(1, wandb_every)
                if (trainer.global_step + 1) % wandb_every == 0:
                    metrics = {
                        # Keep both keys for compatibility with existing W&B dashboards.
                        "loss": float(micro_step_loss),
                        "train/loss": float(micro_step_loss),
                        "train/lr": float(trainer.my_lr),
                        "train/lr2": float(trainer.my_lr2),
                        "train/weight_decay": float(trainer.my_wd),
                        "train/tokens": float(real_global_step * tokens_per_micro_step),
                        "train/ctx_len": int(config.model.ctx_len),
                        "train/global_step_bsz": int(config.runtime.global_step_bsz),
                    }
                    if kt_s > 0:
                        metrics["train/kt_per_s"] = float(kt_s)
                    if hasattr(trainer, 'my_wandb'):
                        trainer.my_wandb.log(metrics, step=int(real_global_step))
        # Early stopping (optional). Must be synchronized across ranks.
        #
        # CRITICAL:
        # - DeepSpeed full checkpointing (trainer.save_checkpoint) may require *all ranks* to participate.
        # - Therefore, any early-stop finalization that calls full checkpointing must be executed on all ranks,
        #   otherwise rank0 may hang and the rest of the ranks will deadlock in later collectives (NCCL timeout).
        #
        # We compute the stop decision on rank0 (for determinism), broadcast it, then run the finalize path on all ranks.
        stop_now, stop_reason = (False, "")
        if trainer.is_global_zero:
            try:
                # Ensure we have a scalar loss value even if logging is disabled or outputs type differs.
                if micro_step_loss is None:
                    if isinstance(outputs, dict) and "loss" in outputs:
                        micro_step_loss = outputs["loss"] * trainer.accumulate_grad_batches
                    else:
                        micro_step_loss = outputs * trainer.accumulate_grad_batches
                loss_value = float(micro_step_loss.detach().float().item()) if hasattr(micro_step_loss, "detach") else float(micro_step_loss)
                stop_now, stop_reason = self._maybe_early_stop(trainer, pl_module, loss_value)
            except Exception:
                stop_now, stop_reason = (False, "")

        # Broadcast the decision (do NOT swallow errors; better to fail fast than desync ranks).
        stop_flag = torch.tensor(1 if stop_now else 0, device=pl_module.device, dtype=torch.int32)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(stop_flag, src=0)

        if int(stop_flag.item()) == 1:
            # Rank0 writes the artifact; then we synchronize and exit together to avoid desync.
            if trainer.is_global_zero:
                # Flush/close logs for durability
                try:
                    if hasattr(trainer, "my_step_log") and trainer.my_step_log is not None:
                        trainer.my_step_log.flush()
                        trainer.my_step_log.close()
                except Exception:
                    pass
                try:
                    if hasattr(trainer, "my_log") and trainer.my_log is not None:
                        trainer.my_log.flush()
                        trainer.my_log.close()
                except Exception:
                    pass
                
                # Save checkpoint locally
                local_ckpt_path = f"{config.runtime.proj_path}/rwkv-final.pth"
                try:
                    pl_module.save_weights(local_ckpt_path)
                    print(f"[early-stop] saved checkpoint: {local_ckpt_path}", flush=True)
                except Exception as e:
                    print(f"[early-stop] ERROR saving checkpoint: {e}", flush=True)
                
                # Upload to GCS (if configured)
                try:
                    gcs_bucket = getattr(config.train, "gcs_bucket", None) or os.environ.get("GCS_BUCKET", "")
                    gcs_prefix = getattr(config.train, "gcs_prefix", None) or os.environ.get("GCS_PREFIX", "")
                    exp_id = getattr(config.train, "exp_id", None) or os.environ.get("EXP_ID", "")
                    run_id = getattr(config.train, "run_id", None) or os.environ.get("RUN_ID", "")
                    
                    if gcs_bucket and exp_id:
                        stage_label = self._stage_label_for_gcs()
                        if run_id:
                            gcs_dir = f"{gcs_bucket.rstrip('/')}/{gcs_prefix}/{exp_id}/{run_id}/checkpoints/{stage_label}"
                        else:
                            gcs_dir = f"{gcs_bucket.rstrip('/')}/{gcs_prefix}/{exp_id}/_latest/checkpoints/{stage_label}"
                        gcs_ckpt_path = f"{gcs_dir}/rwkv-final.pth"
                        
                        print(f"[early-stop] uploading checkpoint to GCS → {gcs_ckpt_path}", flush=True)
                        import subprocess
                        r = subprocess.run(["gsutil", "cp", "-q", local_ckpt_path, gcs_ckpt_path], check=False)
                        if r.returncode != 0:
                            print(f"[early-stop] GCS upload failed with -q (rc={r.returncode}); retrying without -q", flush=True)
                            subprocess.run(["gsutil", "cp", local_ckpt_path, gcs_ckpt_path], check=False)
                        print(f"[early-stop] ✅ checkpoint uploaded to GCS", flush=True)
                    else:
                        print(f"[early-stop] GCS not configured (bucket={bool(gcs_bucket)}, exp_id={bool(exp_id)}), skipping upload", flush=True)
                except Exception as e:
                    print(f"[early-stop] ERROR uploading to GCS: {e}", flush=True)
                
                print(f"!!!EARLY STOP!!! {stop_reason}")

            # Ensure all ranks reach the exit together (prevents noisy comms warnings and avoids
            # leaving straggler ranks running).
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                try:
                    torch.distributed.barrier()
                    torch.distributed.destroy_process_group()
                except Exception:
                    pass

            exit(0)

        # Step-based periodic full checkpoints (proper resume).
        # IMPORTANT: call from every rank (DeepSpeed checkpoints may require all ranks).
        try:
            self._maybe_save_full_resume_checkpoint(trainer, pl_module)
        except Exception:
            pass
        if config.train.magic_prime > 0:
            expand_factor = 1
            trigger_step = int(config.train.magic_prime * expand_factor // self.config.runtime.global_step_bsz) - 1
            if int(real_global_step) == int(trigger_step):
                # Save at most once per real_global_step (important when accumulate_grad_batches > 1)
                if int(real_global_step) != int(self._last_magic_prime_save_step):
                    self._last_magic_prime_save_step = int(real_global_step)
                    pl_module.save_weights(f"{config.runtime.proj_path}/rwkv-final.pth")
                

    def on_train_epoch_start(self, trainer, pl_module):
        config = self.config
        if pl.__version__[0]=='2':
            dataset = trainer.train_dataloader.dataset
        else:
            dataset = trainer.train_dataloader.dataset.datasets
        assert "MyDataset" in str(dataset)
        # print(f'########## world_size {trainer.world_size} global_rank {trainer.global_rank} real_epoch {trainer.real_epoch} ##########')

    def on_train_epoch_end(self, trainer, pl_module):
        config = self.config
        to_save_dict = {}
        real_current_epoch = trainer.current_epoch
        if (config.train.epoch_save > 0 and (real_current_epoch+1) % config.train.epoch_save == 0) or (real_current_epoch == config.runtime.epoch_count - 1):
            try:
                pl_module.save_weights(f"{config.runtime.proj_path}/rwkv-{trainer.current_epoch}.pth")
            except Exception as e:
                print('Error\n\n', e, '\n\n')

        # Note: full-resume checkpointing is step-based only (see on_train_batch_end).

        if trainer.is_global_zero:  # logging
            # NOTE: `math.exp(loss)` is only meaningful for CE loss (perplexity).
            # For other losses (e.g., Stage 1 alignment L2, Stage 2 KL) this can overflow and crash training.
            try:
                ppl = math.exp(float(trainer.my_epoch_loss))
            except OverflowError:
                ppl = float("inf")
            except Exception:
                ppl = float("nan")
            trainer.my_log.write(
                f"{real_current_epoch} {trainer.my_epoch_loss:.6f} {ppl:.4f} {trainer.my_lr:.8f} {datetime.datetime.now()} {real_current_epoch - config.train.epoch_begin}\n"
            )
            trainer.my_log.flush()

            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0

