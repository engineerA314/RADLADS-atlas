import os

import pytest
import torch

from atlasattn import AttentionDistillationWrapper, AtlasSelfAttention


@pytest.mark.skipif(
    str(os.environ.get("RUN_HF_INTEGRATION", "")).lower() not in ("1", "true", "yes"),
    reason="Set RUN_HF_INTEGRATION=1 to enable HF loading test.",
)
def test_qwen2_hf_qkv_transfer_and_forward():
    from transformers import AutoModelForCausalLM

    model_id = os.environ.get("HF_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Grab a teacher attention module from the HF model
    teacher_attn = model.model.layers[0].self_attn

    # Use HF config but inject Atlas knobs (CPU-safe)
    cfg = model.config
    cfg.use_momentum = True
    cfg.omega_window = 4
    cfg.use_omega_gate = True
    cfg.use_cuda = False
    cfg.use_accelerated_scan = False
    cfg.memory_scan_chunk_len = None
    cfg.poly_degree = 2
    cfg.poly_mode = "elementwise"
    cfg.qk_norm = False
    cfg.qkv_conv_kernel = None
    cfg.use_rope = False
    cfg.use_groupnorm = False

    wrapper = AttentionDistillationWrapper(
        teacher_attn, AtlasSelfAttention, cfg, attention_distillation_stage=1
    )

    cell = wrapper.student_attn.memory.cell
    assert torch.allclose(cell.to_q.weight, teacher_attn.q_proj.weight)
    assert torch.allclose(cell.to_k.weight, teacher_attn.k_proj.weight)
    assert torch.allclose(cell.to_v.weight, teacher_attn.v_proj.weight)
    assert torch.allclose(cell.to_out.weight, teacher_attn.o_proj.weight)

    hidden_states = torch.randn(1, 2, cfg.hidden_size, dtype=teacher_attn.q_proj.weight.dtype)

    seq_len = hidden_states.size(1)
    position_embeddings = None
    if hasattr(model.model, "rotary_emb"):
        rotary = model.model.rotary_emb
        try:
            position_embeddings = rotary(hidden_states, seq_len=seq_len)
        except TypeError:
            try:
                position_embeddings = rotary(seq_len=seq_len)
            except TypeError:
                position_ids = torch.arange(seq_len).unsqueeze(0)
                position_embeddings = rotary(hidden_states, position_ids)
    if position_embeddings is None:
        raise RuntimeError("Failed to build position_embeddings for Qwen2Attention")

    outputs = wrapper(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=position_embeddings,
    )
    assert outputs[0].shape == hidden_states.shape
    assert outputs[1] is not None
