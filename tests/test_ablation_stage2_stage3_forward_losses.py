import itertools

import pytest
import torch
import torch.nn.functional as F


def _make_cfg(*, attention_distillation_stage: int, model_kwargs: dict):
    from configs import TrainerCLI_Config, Train_Config, Atlas_Config

    model_cfg = Atlas_Config(**model_kwargs)
    train_cfg = Train_Config(
        attention_distillation_stage=attention_distillation_stage,
        # train_stage value isn't used by Model_atlasqwen2 forward; kept for realism.
        train_stage=3 if attention_distillation_stage == 2 else 2,
        load_model="",
        load_partial=0,
        data_type="binidx",
        data_file="",
        my_exit_tokens=1,
        magic_prime=0,
        micro_bsz=1,
        devices=1,
        num_nodes=1,
        strategy="auto",
    )
    return TrainerCLI_Config(train=train_cfg, model=model_cfg, runtime=None)


def _corrupt_rope_tables(module: torch.nn.Module) -> None:
    # Deterministic "bad buffers" to simulate to_empty() + non-persistent buffers.
    for m in module.modules():
        if m.__class__.__name__ == "RotaryEmbedding":
            # Ensure cache exists and is long enough to not trigger "too short" rebuild.
            # Then corrupt values (this specifically tests corruption handling).
            need_len = 64
            half = m.dim // 2
            m.cos = torch.full((need_len, half), float("nan"), dtype=torch.float32)
            m.sin = torch.full((need_len, half), float("nan"), dtype=torch.float32)


def _forward_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids)
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, tuple):
        return out[0]
    if isinstance(out, torch.Tensor):
        return out
    raise TypeError(f"Unexpected output type: {type(out)}")


def _ids_for_configs(cfgs):
    return [c["name"] for c in cfgs]


def _all_ablation_configs():
    """
    Exhaustive ablation combinations (32):
      poly_degree: 2/3
      qk_norm: False/True
      qkv_conv_kernel: None/4
      use_rope: False/True
      use_groupnorm: False/True
    """
    poly_degrees = [2, 3]
    qk_norms = [False, True]
    convs = [None, 4]
    ropes = [False, True]
    gns = [False, True]

    cfgs = []
    for poly_degree, qk_norm, qkv_conv_kernel, use_rope, use_groupnorm in itertools.product(
        poly_degrees, qk_norms, convs, ropes, gns
    ):
        name = (
            f"poly{poly_degree}-"
            f"qknorm{int(qk_norm)}-"
            f"conv{0 if qkv_conv_kernel is None else qkv_conv_kernel}-"
            f"rope{int(use_rope)}-"
            f"gn{int(use_groupnorm)}"
        )
        cfgs.append(
            dict(
                name=name,
                poly_degree=poly_degree,
                qk_norm=qk_norm,
                qkv_conv_kernel=qkv_conv_kernel,
                use_rope=use_rope,
                use_groupnorm=use_groupnorm,
            )
        )
    return cfgs


@pytest.mark.parametrize("hp", _all_ablation_configs(), ids=_ids_for_configs(_all_ablation_configs()))
def test_stage2_student_forward_and_kl_are_finite_for_all_ablation_configs(hp):
    """
    This test is specifically aimed at catching the GCP failure:
      WITH_ALL_ABLATION=1 → Stage 2 → student logits become all non-finite before KL.

    We simulate the critical behavior:
      - build Model_atlasqwen2 with ablation knobs
      - save state_dict
      - to_empty() (wipes params/buffers)
      - load_state_dict (buffers like RoPE cos/sin are NOT restored)
      - run forward + KL loss

    For use_rope=True we *force* RoPE buffers to be corrupted (NaNs) to make the
    failure deterministic and to ensure the runtime rebuild logic works.
    """
    from models.atlasqwen2 import Model_atlasqwen2

    vocab_size = 256
    model_kwargs = dict(
        vocab_size=vocab_size,
        n_embd=64,
        n_layer=2,
        dim_ffn=128,
        dim_att=64,
        head_size=16,
        ctx_len=32,
        rms_norm_eps=1e-6,
        memory_heads=4,
        memory_dim_head=16,
        num_key_value_heads=2,
        use_momentum=True,
        omega_window=4,
        use_omega_gate=True,
        poly_degree=hp["poly_degree"],
        poly_mode="elementwise",
        qk_norm=hp["qk_norm"],
        qkv_conv_kernel=hp["qkv_conv_kernel"],
        use_rope=hp["use_rope"],
        use_groupnorm=hp["use_groupnorm"],
        atlas_variant="lmm",
        tie_word_embeddings=True,
    )

    cfg2 = _make_cfg(attention_distillation_stage=2, model_kwargs=model_kwargs)
    student = Model_atlasqwen2(cfg2)
    student.configure_model()

    sd = student.state_dict()
    student.to_empty(device=torch.device("cpu"))
    student.load_state_dict(sd, strict=True)
    if hp["use_rope"]:
        _corrupt_rope_tables(student)

    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab_size, (2, 16))
    student_logits = _forward_logits(student, input_ids)
    assert torch.isfinite(student_logits).all()

    teacher_logits = torch.randn_like(student_logits)
    kl = F.kl_div(
        F.log_softmax(student_logits.view(-1, student_logits.size(-1)), dim=-1),
        F.log_softmax(teacher_logits.view(-1, teacher_logits.size(-1)), dim=-1),
        log_target=True,
        reduction="batchmean",
    )
    assert torch.isfinite(kl).all()


@pytest.mark.parametrize("hp", _all_ablation_configs(), ids=_ids_for_configs(_all_ablation_configs()))
def test_stage3_student_forward_and_ce_are_finite_for_all_ablation_configs(hp):
    """
    Paper Stage 3: student-only CE (no teacher), progressive ctx is handled elsewhere.
    Here we validate the forward path and CE loss are finite for all ablation knobs,
    under the same to_empty() + reload behavior.
    """
    from models.atlasqwen2 import Model_atlasqwen2

    vocab_size = 256
    model_kwargs = dict(
        vocab_size=vocab_size,
        n_embd=64,
        n_layer=2,
        dim_ffn=128,
        dim_att=64,
        head_size=16,
        ctx_len=32,
        rms_norm_eps=1e-6,
        memory_heads=4,
        memory_dim_head=16,
        num_key_value_heads=2,
        use_momentum=True,
        omega_window=4,
        use_omega_gate=True,
        poly_degree=hp["poly_degree"],
        poly_mode="elementwise",
        qk_norm=hp["qk_norm"],
        qkv_conv_kernel=hp["qkv_conv_kernel"],
        use_rope=hp["use_rope"],
        use_groupnorm=hp["use_groupnorm"],
        atlas_variant="lmm",
        tie_word_embeddings=True,
    )

    cfg3 = _make_cfg(attention_distillation_stage=-1, model_kwargs=model_kwargs)
    student = Model_atlasqwen2(cfg3)
    student.configure_model()

    sd = student.state_dict()
    student.to_empty(device=torch.device("cpu"))
    student.load_state_dict(sd, strict=True)
    if hp["use_rope"]:
        _corrupt_rope_tables(student)

    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab_size, (2, 16))
    labels = torch.randint(0, vocab_size, (2, 16))
    logits = _forward_logits(student, input_ids)
    assert torch.isfinite(logits).all()

    ce = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    assert torch.isfinite(ce).all()

