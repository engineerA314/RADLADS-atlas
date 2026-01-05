import json
import os
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass
class _MockTrain:
    epoch_begin: int = 0
    epoch_save: int = 0
    wandb: str = ""  # disable
    proj_suffix: str = "atlas-stage1"
    attention_distillation_stage: int = 1


@dataclass
class _MockModel:
    ctx_len: int = 512


@dataclass
class _MockRuntime:
    proj_path: str = ""
    epoch_count: int = 999
    my_timestamp: str = "20990101-000000"
    global_step_bsz: int = 8
    epoch_global_steps: int = 10


@dataclass
class _MockConfig:
    train: _MockTrain
    model: _MockModel
    runtime: _MockRuntime


class _DummyPLModule:
    def __init__(self):
        self.saved = []

    def save_weights(self, path: str):
        self.saved.append(path)
        Path(path).write_bytes(b"dummy-weights")

    def get_real_global_step(self) -> int:
        return 123

    def get_real_tokens(self) -> int:
        return 456789


class _DummyTrainer:
    def __init__(self, *, proj_path: Path):
        self.current_epoch = 0
        self.global_step = 123
        self.is_global_zero = True
        self.my_epoch_loss = 0.0
        self.my_lr = 1e-4
        self.my_loss_sum = 0
        self.my_loss_count = 0
        self.my_log = open(proj_path / "train_log.txt", "a", encoding="utf-8")
        self._proj_path = proj_path

    # For full resume checkpoint tests
    def save_checkpoint(self, path: str, *args, **kwargs):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"dummy-full-ckpt")


def _install_import_stubs_for_trainer(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The local dev environment for orchestration may not have the full training deps installed
    (e.g., lightning / deepspeed). For unit-testing our resume logic, we stub the minimal
    modules required to import `src.trainer`.
    """
    def _rank_zero_info(*args, **kwargs):
        return None

    def _rank_zero_only(fn):
        return fn

    # ---- lightning.pytorch stub ----
    pl_mod = types.ModuleType("lightning.pytorch")
    pl_mod.__version__ = "2.0.0"
    pl_mod.Callback = type("Callback", (), {})

    # lightning.pytorch.utilities.rank_zero is used by src/logger.py
    pl_util_mod = types.ModuleType("lightning.pytorch.utilities")
    pl_rank_zero_mod = types.ModuleType("lightning.pytorch.utilities.rank_zero")
    pl_rank_zero_mod.rank_zero_info = _rank_zero_info
    pl_rank_zero_mod.rank_zero_only = _rank_zero_only
    pl_util_mod.rank_zero = pl_rank_zero_mod
    pl_mod.utilities = pl_util_mod

    lightning_mod = types.ModuleType("lightning")
    lightning_mod.pytorch = pl_mod

    monkeypatch.setitem(sys.modules, "lightning", lightning_mod)
    monkeypatch.setitem(sys.modules, "lightning.pytorch", pl_mod)
    monkeypatch.setitem(sys.modules, "lightning.pytorch.utilities", pl_util_mod)
    monkeypatch.setitem(sys.modules, "lightning.pytorch.utilities.rank_zero", pl_rank_zero_mod)

    # ---- lightning_utilities.core.rank_zero stub ----
    rank_zero_mod = types.ModuleType("lightning_utilities.core.rank_zero")

    rank_zero_mod.rank_zero_info = _rank_zero_info
    rank_zero_mod.rank_zero_only = _rank_zero_only

    lu_core_mod = types.ModuleType("lightning_utilities.core")
    lu_core_mod.rank_zero = rank_zero_mod
    lu_mod = types.ModuleType("lightning_utilities")
    lu_mod.core = lu_core_mod

    monkeypatch.setitem(sys.modules, "lightning_utilities", lu_mod)
    monkeypatch.setitem(sys.modules, "lightning_utilities.core", lu_core_mod)
    monkeypatch.setitem(sys.modules, "lightning_utilities.core.rank_zero", rank_zero_mod)

    # ---- deepspeed.utils stub ----
    ds_utils_mod = types.ModuleType("deepspeed.utils")
    ds_utils_mod.safe_get_full_grad = lambda *args, **kwargs: None
    ds_mod = types.ModuleType("deepspeed")
    ds_mod.utils = ds_utils_mod

    monkeypatch.setitem(sys.modules, "deepspeed", ds_mod)
    monkeypatch.setitem(sys.modules, "deepspeed.utils", ds_utils_mod)


def test_resume_checkpoint_saves_meta_and_uploads(monkeypatch, tmp_path: Path):
    # Arrange: config + callback
    # NOTE: weights-only resume was removed; this test now asserts that no weights-only
    # resume artifacts are produced when full resume is disabled.
    try:
        import lightning.pytorch  # noqa: F401
        import deepspeed  # noqa: F401
    except Exception:
        _install_import_stubs_for_trainer(monkeypatch)

    from src.trainer import train_callback

    proj_path = tmp_path / "out" / "run"
    proj_path.mkdir(parents=True, exist_ok=True)

    cfg = _MockConfig(
        train=_MockTrain(epoch_begin=0, epoch_save=0, wandb="", proj_suffix="atlas-stage1", attention_distillation_stage=1),
        model=_MockModel(ctx_len=512),
        runtime=_MockRuntime(
            proj_path=str(proj_path),
            epoch_count=999,
            my_timestamp="20990101-000000",
            global_step_bsz=8,
            epoch_global_steps=10,
        ),
    )
    cb = train_callback(cfg)

    trainer = _DummyTrainer(proj_path=proj_path)
    pl_module = _DummyPLModule()

    # Disable full checkpointing (and no weights-only exists anymore)
    monkeypatch.setenv("GCS_BUCKET", "gs://dummy-bucket")
    monkeypatch.setenv("GCS_PREFIX", "radlads-atlas/ablation")
    monkeypatch.setenv("EXP_ID", "exp-test")
    monkeypatch.setenv("RUN_ID", "run-test")
    monkeypatch.setenv("ENABLE_FULL_RESUME_CKPT", "0")
    monkeypatch.setenv("FULL_RESUME_EVERY_N_STEPS", "0")

    popen_calls = []

    class _PopenStub:
        def __init__(self, args, stdout=None, stderr=None):
            popen_calls.append({"args": args, "stdout": stdout, "stderr": stderr})

    monkeypatch.setattr("src.trainer.subprocess.Popen", _PopenStub)

    # Act
    cb.on_train_epoch_end(trainer, pl_module)

    # Assert: no weights-only artifacts created and no uploads attempted
    assert not (proj_path / "rwkv-resume.pth").exists()
    assert not (proj_path / "rwkv-resume.meta.json").exists()
    assert len(popen_calls) == 0

    trainer.my_log.close()


def test_full_resume_checkpoint_saves_meta_and_uploads(monkeypatch, tmp_path: Path):
    # Arrange: import stubs if needed
    try:
        import lightning.pytorch  # noqa: F401
        import deepspeed  # noqa: F401
    except Exception:
        _install_import_stubs_for_trainer(monkeypatch)

    from src.trainer import train_callback

    proj_path = tmp_path / "out" / "run"
    proj_path.mkdir(parents=True, exist_ok=True)

    cfg = _MockConfig(
        train=_MockTrain(epoch_begin=0, epoch_save=0, wandb="", proj_suffix="atlas-stage3-ctx512", attention_distillation_stage=-1),
        model=_MockModel(ctx_len=512),
        runtime=_MockRuntime(
            proj_path=str(proj_path),
            epoch_count=999,
            my_timestamp="20990101-000000",
            global_step_bsz=8,
            epoch_global_steps=10,
        ),
    )
    cb = train_callback(cfg)

    trainer = _DummyTrainer(proj_path=proj_path)
    pl_module = _DummyPLModule()

    monkeypatch.setenv("GCS_BUCKET", "gs://dummy-bucket")
    monkeypatch.setenv("GCS_PREFIX", "radlads-atlas/ablation")
    monkeypatch.setenv("EXP_ID", "exp-test")
    monkeypatch.setenv("RUN_ID", "run-test")
    monkeypatch.setenv("ENABLE_FULL_RESUME_CKPT", "1")
    monkeypatch.setenv("FULL_RESUME_EVERY_N_STEPS", "1")

    popen_calls = []

    class _PopenStub:
        def __init__(self, args, stdout=None, stderr=None):
            popen_calls.append({"args": args, "stdout": stdout, "stderr": stderr})

    monkeypatch.setattr("src.trainer.subprocess.Popen", _PopenStub)

    # Act (step-based only)
    cb._maybe_save_full_resume_checkpoint(trainer, pl_module)

    # Assert: local files created
    full_ckpt = proj_path / "full-resume.ckpt"
    full_meta = proj_path / "full-resume.meta.json"
    assert full_ckpt.exists()
    assert full_meta.exists()

    # Assert: upload command targets stage label derived from proj_suffix (stage3-ctx512)
    assert len(popen_calls) == 1
    cmd = popen_calls[0]["args"][2]
    assert "gsutil" in cmd
    assert "full-resume.ckpt" in cmd
    assert "full-resume.meta.json" in cmd
    assert "gs://dummy-bucket/radlads-atlas/ablation/exp-test/_latest/full_resume/stage3-ctx512/full-resume.ckpt" in cmd

    trainer.my_log.close()


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def test_run_stage1_script_force_fresh_skips_full_auto_resume(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "run_stage1.sh"
    assert script.exists()

    fake_gcs = tmp_path / "fake_gcs"
    fake_gcs.mkdir(parents=True, exist_ok=True)
    (fake_gcs / "full-resume.ckpt").write_bytes(b"dummy-full")
    (fake_gcs / "full-resume.meta.json").write_text(
        json.dumps({"ctx_len": 512, "ckpt_is_dir": False}) + "\n",
        encoding="utf-8",
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    _write_executable(
        bin_dir / "gsutil",
        """#!/usr/bin/env bash
set -euo pipefail
echo "gsutil should not be called when FORCE_FRESH=1" >&2
exit 99
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": str(bin_dir) + os.pathsep + env.get("PATH", ""),
            "FAKE_GCS_DIR": str(fake_gcs),
            "EXP_ID": "exp-test",
            "GCS_BUCKET": "gs://dummy-bucket",
            "GCS_PREFIX": "radlads-atlas/ablation",
            "FULL_AUTO_RESUME": "1",
            "FORCE_FRESH": "1",
            "DRY_RUN": "1",
        }
    )

    p = subprocess.run(
        ["bash", str(script), "100", "1", "512"],
        cwd=str(repo_root),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Found FULL resume checkpoint in GCS" not in p.stdout
    assert "FULL auto-resume enabled" not in p.stdout


def test_run_stage1_script_full_auto_resume_sets_ckpt_path(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "run_stage1.sh"
    assert script.exists()

    fake_gcs = tmp_path / "fake_gcs"
    fake_gcs.mkdir(parents=True, exist_ok=True)

    # Provide full checkpoint objects
    (fake_gcs / "full-resume.ckpt").write_bytes(b"dummy-full")
    (fake_gcs / "full-resume.meta.json").write_text(
        json.dumps({"ctx_len": 512, "ckpt_is_dir": False}) + "\n",
        encoding="utf-8",
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    _write_executable(
        bin_dir / "gsutil",
        """#!/usr/bin/env bash
set -euo pipefail
cmd="${1:-}"; shift || true
case "$cmd" in
  ls)
    uri="${1:-}"
    base="$(basename "$uri")"
    test -f "${FAKE_GCS_DIR}/${base}"
    ;;
  cp)
    if [ "${1:-}" = "-q" ]; then shift; fi
    src="${1:-}"; dst="${2:-}"
    base="$(basename "$src")"
    cp -f "${FAKE_GCS_DIR}/${base}" "${dst}"
    ;;
  -m)
    # support: gsutil -m rsync -r SRC DST
    sub="${1:-}"; shift
    if [ "$sub" != "rsync" ]; then
      echo "unsupported gsutil -m subcmd: $sub" >&2
      exit 2
    fi
    # ignore flags
    while [[ "${1:-}" == -* ]]; do shift; done
    src="${1:-}"; dst="${2:-}"
    base="$(basename "$src")"
    mkdir -p "${dst}"
    # In our mock we only support file-like full-resume.ckpt, so just copy it into dst.
    cp -f "${FAKE_GCS_DIR}/${base}" "${dst}"
    ;;
  *)
    echo "unsupported gsutil cmd: $cmd" >&2
    exit 2
    ;;
esac
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": str(bin_dir) + os.pathsep + env.get("PATH", ""),
            "FAKE_GCS_DIR": str(fake_gcs),
            "EXP_ID": "exp-test",
            "GCS_BUCKET": "gs://dummy-bucket",
            "GCS_PREFIX": "radlads-atlas/ablation",
            "FULL_AUTO_RESUME": "1",
            "FORCE_FRESH": "0",
            "DRY_RUN": "1",
        }
    )

    p = subprocess.run(
        ["bash", str(script), "100", "1", "512"],
        cwd=str(repo_root),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "🔁 Found FULL resume checkpoint in GCS" in p.stdout
    assert "✅ FULL auto-resume enabled" in p.stdout
    assert "FULL_RESUME=1" in p.stdout
    assert "FULL_RESUME_CKPT_PATH=" in p.stdout


def test_run_stage2_script_full_auto_resume_sets_ckpt_path(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "run_stage2.sh"
    assert script.exists()

    # Dummy stage1 ckpt arg (required by interface, even though full resume will be used)
    dummy_stage1 = tmp_path / "stage1.pth"
    dummy_stage1.write_bytes(b"dummy")

    fake_gcs = tmp_path / "fake_gcs"
    fake_gcs.mkdir(parents=True, exist_ok=True)
    (fake_gcs / "full-resume.ckpt").write_bytes(b"dummy-full")
    (fake_gcs / "full-resume.meta.json").write_text(json.dumps({"ctx_len": 512, "ckpt_is_dir": False}) + "\n", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    _write_executable(
        bin_dir / "gsutil",
        """#!/usr/bin/env bash
set -euo pipefail
cmd="${1:-}"; shift || true
case "$cmd" in
  ls)
    uri="${1:-}"
    base="$(basename "$uri")"
    test -f "${FAKE_GCS_DIR}/${base}"
    ;;
  cp)
    if [ "${1:-}" = "-q" ]; then shift; fi
    src="${1:-}"; dst="${2:-}"
    base="$(basename "$src")"
    cp -f "${FAKE_GCS_DIR}/${base}" "${dst}"
    ;;
  -m)
    sub="${1:-}"; shift
    if [ "$sub" != "rsync" ]; then
      echo "unsupported gsutil -m subcmd: $sub" >&2
      exit 2
    fi
    while [[ "${1:-}" == -* ]]; do shift; done
    src="${1:-}"; dst="${2:-}"
    base="$(basename "$src")"
    mkdir -p "${dst}"
    cp -f "${FAKE_GCS_DIR}/${base}" "${dst}"
    ;;
  *)
    echo "unsupported gsutil cmd: $cmd" >&2
    exit 2
    ;;
esac
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": str(bin_dir) + os.pathsep + env.get("PATH", ""),
            "FAKE_GCS_DIR": str(fake_gcs),
            "EXP_ID": "exp-test",
            "GCS_BUCKET": "gs://dummy-bucket",
            "GCS_PREFIX": "radlads-atlas/ablation",
            "FULL_AUTO_RESUME": "1",
            "FORCE_FRESH": "0",
            "DRY_RUN": "1",
        }
    )

    p = subprocess.run(
        ["bash", str(script), str(dummy_stage1), "100", "1", "512"],
        cwd=str(repo_root),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "🔁 Found FULL resume checkpoint in GCS" in p.stdout
    assert "✅ FULL auto-resume enabled" in p.stdout
    assert "FULL_RESUME=1" in p.stdout


def test_run_stage3_script_full_auto_resume_sets_ckpt_path(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "run_stage3.sh"
    assert script.exists()

    dummy_stage2 = tmp_path / "stage2.pth"
    dummy_stage2.write_bytes(b"dummy")

    fake_gcs = tmp_path / "fake_gcs"
    fake_gcs.mkdir(parents=True, exist_ok=True)
    (fake_gcs / "full-resume.ckpt").write_bytes(b"dummy-full")
    (fake_gcs / "full-resume.meta.json").write_text(json.dumps({"ctx_len": 512, "ckpt_is_dir": False}) + "\n", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    _write_executable(
        bin_dir / "gsutil",
        """#!/usr/bin/env bash
set -euo pipefail
cmd="${1:-}"; shift || true
case "$cmd" in
  ls)
    uri="${1:-}"
    base="$(basename "$uri")"
    test -f "${FAKE_GCS_DIR}/${base}"
    ;;
  cp)
    if [ "${1:-}" = "-q" ]; then shift; fi
    src="${1:-}"; dst="${2:-}"
    base="$(basename "$src")"
    cp -f "${FAKE_GCS_DIR}/${base}" "${dst}"
    ;;
  -m)
    sub="${1:-}"; shift
    if [ "$sub" != "rsync" ]; then
      echo "unsupported gsutil -m subcmd: $sub" >&2
      exit 2
    fi
    while [[ "${1:-}" == -* ]]; do shift; done
    src="${1:-}"; dst="${2:-}"
    base="$(basename "$src")"
    mkdir -p "${dst}"
    cp -f "${FAKE_GCS_DIR}/${base}" "${dst}"
    ;;
  *)
    echo "unsupported gsutil cmd: $cmd" >&2
    exit 2
    ;;
esac
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": str(bin_dir) + os.pathsep + env.get("PATH", ""),
            "FAKE_GCS_DIR": str(fake_gcs),
            "EXP_ID": "exp-test",
            "GCS_BUCKET": "gs://dummy-bucket",
            "GCS_PREFIX": "radlads-atlas/ablation",
            "FULL_AUTO_RESUME": "1",
            "FORCE_FRESH": "0",
            "DRY_RUN": "1",
        }
    )

    p = subprocess.run(
        # single ctx schedule for deterministic test
        ["bash", str(script), str(dummy_stage2), "512", "100", "1"],
        cwd=str(repo_root),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "🔁 Found FULL resume checkpoint in GCS for stage3-ctx512" in p.stdout
    assert "✅ FULL auto-resume enabled for stage3-ctx512" in p.stdout
    assert "FULL_RESUME=1" in p.stdout

