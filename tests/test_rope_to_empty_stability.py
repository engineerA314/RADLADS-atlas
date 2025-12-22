import torch


def test_rotary_embedding_rebuilds_after_to_empty_corruption():
    """
    Regression test for: Stage 2 NaNs when WITH_ALL_ABLATION enables use_rope=true.

    Root cause:
      - Lightning calls `module.to_empty(device=...)` on the student model.
      - `to_empty()` also wipes buffers.
      - Our RoPE cos/sin tables are non-persistent buffers (not in state_dict),
        so they are NOT restored by load_state_dict.
      - If cos/sin remain corrupted, apply_rotary_embedding can create NaNs/Inf.

    This test simulates the corruption and verifies the module rebuilds tables.
    """
    from src.rotary import RotaryEmbedding

    emb = RotaryEmbedding(max_sequence_length=32, dim=8, theta=10000.0)

    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 4, 8)

    # Build cache once.
    _ = emb(q, k)

    # Simulate corruption (deterministic).
    emb.to_empty(device=torch.device("cpu"))
    assert emb.cos is not None and emb.sin is not None
    emb.cos.zero_()
    emb.sin.fill_(123.0)

    q2, k2 = emb(q, k)
    assert torch.isfinite(q2).all()
    assert torch.isfinite(k2).all()

    # RoPE invariant restored
    assert torch.allclose(emb.cos[0], torch.ones_like(emb.cos[0]), atol=1e-3, rtol=0.0)
    assert torch.allclose(emb.sin[0], torch.zeros_like(emb.sin[0]), atol=1e-3, rtol=0.0)


def test_atlas_lmm_forward_is_finite_with_rope_after_to_empty_and_reload():
    """
    Simulate the Stage2/3 flow for the student (Model_atlasqwen2):
      init -> save state_dict (no cos/sin) -> to_empty -> load_state_dict -> forward

    We then also *force* RoPE tables to be corrupted to ensure forward rebuilds.
    """
    from configs import TrainerCLI_Config, Train_Config, Atlas_Config
    from models.atlasqwen2 import Model_atlasqwen2

    # Tiny model for fast CPU test
    model_cfg = Atlas_Config(
        vocab_size=256,
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
        poly_degree=3,
        poly_mode="elementwise",
        qk_norm=True,
        qkv_conv_kernel=4,
        use_rope=True,
        use_groupnorm=True,
        atlas_variant="lmm",
        tie_word_embeddings=True,
    )
    train_cfg = Train_Config(
        attention_distillation_stage=2,
        train_stage=3,
        load_model="",  # we'll load manually in test
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
    cfg = TrainerCLI_Config(train=train_cfg, model=model_cfg, runtime=None)

    student = Model_atlasqwen2(cfg)
    student.configure_model()

    # Save weights (cos/sin are non-persistent buffers)
    sd = student.state_dict()

    # Simulate Lightning init behavior
    student.to_empty(device=torch.device("cpu"))
    student.load_state_dict(sd, strict=True)

    # Force RoPE tables to be corrupted to simulate the real-world bug deterministically.
    for module in student.modules():
        if module.__class__.__name__ == "RotaryEmbedding":
            # Ensure cache exists, then corrupt it.
            _ = module(torch.randn(1, 2, module.dim), torch.randn(1, 2, module.dim))
            assert module.cos is not None and module.sin is not None
            module.cos.fill_(float("nan"))
            module.sin.fill_(float("nan"))

    input_ids = torch.randint(0, 256, (1, 16))
    out = student(input_ids)
    logits = out.logits if hasattr(out, "logits") else out[0]
    assert torch.isfinite(logits).all()

