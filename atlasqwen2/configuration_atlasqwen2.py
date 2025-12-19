"""Atlas-Qwen2 model configuration for HuggingFace."""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class AtlasQwen2Config(PretrainedConfig):
    r"""
    Configuration class for Atlas-Qwen2 (LMM) models.
    
    Atlas-LMM replaces self-attention with RNN-based memory, keeping the 
    Transformer-style embedding/MLP/norm structure.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size.
        hidden_size (`int`, *optional*, defaults to 896):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 4864):
            Dimension of the MLP intermediate layer.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of decoder layers.
        memory_heads (`int`, *optional*, defaults to 14):
            Number of memory attention heads.
        memory_dim_head (`int`, *optional*, defaults to 64):
            Dimension per memory head.
        use_momentum (`bool`, *optional*, defaults to True):
            Whether to use momentum in memory updates.
        omega_window (`int`, *optional*, defaults to 4):
            Omega window size for sliding window context (must be >= 2).
        use_omega_gate (`bool`, *optional*, defaults to True):
            Whether to use learnable omega gate.
        poly_degree (`int`, *optional*, defaults to 1):
            Polynomial feature map degree.
        poly_mode (`str`, *optional*, defaults to 'off'):
            Polynomial feature map mode ('off', 'elementwise', 'tensor').
        qk_norm (`bool`, *optional*, defaults to True):
            Whether to apply normalization to Q and K.
        qkv_conv_kernel (`int`, *optional*, defaults to None):
            Depthwise conv kernel size for Q/K/V (None = disabled).
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            RMS normalization epsilon.
        use_cache (`bool`, *optional*, defaults to True):
            Whether to return memory state for caching.
        tie_word_embeddings (`bool`, *optional*, defaults to True):
            Whether to tie input/output embeddings.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            Maximum sequence length (for compatibility).
    """

    model_type = "atlasqwen2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=24,
        memory_heads=14,
        memory_dim_head=64,
        use_momentum=True,
        omega_window=4,
        use_omega_gate=True,
        poly_degree=1,
        poly_mode='off',
        qk_norm=True,
        qkv_conv_kernel=None,
        rms_norm_eps=1e-6,
        use_cache=True,
        # Architecture variant
        atlas_variant='lmm',  # 'lmm' (memory only) or 'mal' (memory + attention)
        sliding_window=512,   # For MAL: sliding window attention size
        tie_word_embeddings=True,
        max_position_embeddings=131072,
        initializer_range=0.02,
        hidden_act="silu",
        pad_token_id=None,
        bos_token_id=151643,
        eos_token_id=151643,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.memory_heads = memory_heads
        self.memory_dim_head = memory_dim_head
        self.use_momentum = use_momentum
        self.omega_window = omega_window
        self.use_omega_gate = use_omega_gate
        self.poly_degree = poly_degree
        self.poly_mode = poly_mode
        self.qk_norm = qk_norm
        self.qkv_conv_kernel = qkv_conv_kernel
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.atlas_variant = atlas_variant
        self.sliding_window = sliding_window
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.hidden_act = hidden_act

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
    
    def to_atlas_core_config(self):
        """Convert to internal AtlasConfig for core model."""
        from models.atlasqwen2_core import AtlasConfig
        return AtlasConfig(
            vocab_size=self.vocab_size,
            n_embd=self.hidden_size,
            n_layer=self.num_hidden_layers,
            dim_ffn=self.intermediate_size,
            rms_norm_eps=self.rms_norm_eps,
            memory_heads=self.memory_heads,
            memory_dim_head=self.memory_dim_head,
            use_momentum=self.use_momentum,
            omega_window=self.omega_window,
            use_omega_gate=self.use_omega_gate,
            poly_degree=self.poly_degree,
            poly_mode=self.poly_mode,
            qk_norm=self.qk_norm,
            qkv_conv_kernel=self.qkv_conv_kernel,
            atlas_variant=self.atlas_variant,
            sliding_window=self.sliding_window,
            ctx_len=self.max_position_embeddings,
            tie_word_embeddings=self.tie_word_embeddings,
        )
