"""
TDD: Atlas grad checkpointing correctness + smoke training (2 steps).
"""

import pytest
import torch
import torch.nn.functional as F

from models.atlasqwen2_core import AtlasConfig, AtlasQwen2ForCausalLM


def _tiny_cfg(*, grad_cp: int) -> AtlasConfig:
    return AtlasConfig(
        vocab_size=256,
        n_embd=64,
        n_layer=2,
        dim_ffn=128,
        rms_norm_eps=1e-6,
        memory_heads=4,
        memory_dim_head=16,
        num_key_value_heads=2,  # exercise GQA path
        use_momentum=True,
        omega_window=4,
        use_omega_gate=True,
        poly_degree=1,
        poly_mode="off",
        qk_norm=True,
        qkv_conv_kernel=None,
        use_rope=False,
        use_groupnorm=False,
        use_accelerated_scan=False,
        memory_scan_chunk_len=8,
        ctx_len=32,
        tie_word_embeddings=True,
        grad_cp=grad_cp,
    )


@pytest.mark.parametrize("grad_cp", [0, 1])
def test_atlas_two_steps_forward_backward_no_hang(grad_cp):
    """
    Runs 2 training steps (forward+backward+optimizer) with and without grad checkpointing.
    If this hangs, it's a strong signal the checkpoint wrapper is invalid.
    """
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    torch.manual_seed(123)
    model = AtlasQwen2ForCausalLM(_tiny_cfg(grad_cp=grad_cp)).to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(2):
        x = torch.randint(0, model.config.vocab_size, (2, 32), device=device)
        y = torch.randint(0, model.config.vocab_size, (2, 32), device=device)

        logits, _, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        assert torch.isfinite(loss).item()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def test_atlas_grad_cp_forward_equivalence():
    """
    Forward logits should match (checkpointing only affects backward activation storage).
    """
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    torch.manual_seed(999)
    m0 = AtlasQwen2ForCausalLM(_tiny_cfg(grad_cp=0)).to(device).train()
    m1 = AtlasQwen2ForCausalLM(_tiny_cfg(grad_cp=1)).to(device).train()
    m1.load_state_dict(m0.state_dict(), strict=True)

    x = torch.randint(0, m0.config.vocab_size, (2, 32), device=device)

    logits0, _, _ = m0(x)
    logits1, _, _ = m1(x)

    assert torch.allclose(logits0, logits1, atol=1e-6, rtol=0.0)


