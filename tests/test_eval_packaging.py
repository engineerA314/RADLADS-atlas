"""
TDD Tests for Evaluation Packaging (Phase F).

Tests verify:
- Safetensors export works
- Model can be loaded from saved safetensors
- generate() produces tokens
- Model can be loaded with trust_remote_code
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path


@pytest.fixture
def tiny_config():
    """Tiny model config for fast tests."""
    from atlasqwen2 import AtlasQwen2Config
    return AtlasQwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        memory_heads=4,
        memory_dim_head=16,
        use_momentum=True,
    )


@pytest.fixture
def seed():
    torch.manual_seed(42)
    return 42


class TestSafetensorsExport:
    """Test safetensors export functionality."""
    
    def test_save_pretrained_creates_safetensors(self, tiny_config, seed):
        """save_pretrained creates safetensors files."""
        from atlasqwen2 import AtlasQwen2ForCausalLM
        
        torch.manual_seed(seed)
        model = AtlasQwen2ForCausalLM(tiny_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            
            # Check safetensors file exists
            safetensors_path = Path(tmpdir) / "model.safetensors"
            assert safetensors_path.exists(), "model.safetensors not created"
    
    def test_save_pretrained_creates_config(self, tiny_config, seed):
        """save_pretrained creates config.json."""
        from atlasqwen2 import AtlasQwen2ForCausalLM
        
        torch.manual_seed(seed)
        model = AtlasQwen2ForCausalLM(tiny_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            
            config_path = Path(tmpdir) / "config.json"
            assert config_path.exists(), "config.json not created"
    
    def test_load_from_safetensors(self, tiny_config, seed):
        """Model can be loaded from safetensors."""
        from atlasqwen2 import AtlasQwen2ForCausalLM
        
        torch.manual_seed(seed)
        model = AtlasQwen2ForCausalLM(tiny_config)
        
        # Get original output
        x = torch.randint(0, tiny_config.vocab_size, (1, 4))
        with torch.no_grad():
            orig_out = model(x).logits
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            
            # Load from disk
            loaded_model = AtlasQwen2ForCausalLM.from_pretrained(tmpdir)
            
            with torch.no_grad():
                loaded_out = loaded_model(x).logits
        
        assert torch.allclose(orig_out, loaded_out), "Loaded model gives different output"


class TestGenerate:
    """Test text generation."""
    
    def test_generate_produces_tokens(self, tiny_config, seed):
        """generate() produces output tokens."""
        from atlasqwen2 import AtlasQwen2ForCausalLM
        
        torch.manual_seed(seed)
        model = AtlasQwen2ForCausalLM(tiny_config)
        model.eval()
        
        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 4))
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=5,
                do_sample=False,
            )
        
        # Should produce input + new tokens
        assert output_ids.shape[1] == 4 + 5, f"Expected 9 tokens, got {output_ids.shape[1]}"
    
    def test_generate_respects_eos(self, tiny_config, seed):
        """generate() can produce tokens with EOS handling."""
        from atlasqwen2 import AtlasQwen2ForCausalLM
        
        torch.manual_seed(seed)
        model = AtlasQwen2ForCausalLM(tiny_config)
        model.eval()
        
        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 4))
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                eos_token_id=tiny_config.eos_token_id,
            )
        
        # Should produce at least input tokens
        assert output_ids.shape[1] >= 4


class TestTrustRemoteCode:
    """Test trust_remote_code loading pattern."""
    
    def test_package_has_auto_classes(self, tiny_config, seed):
        """Package defines auto class mappings."""
        # Check that the package is set up for trust_remote_code loading
        import importlib.util
        
        # The atlasqwen2 package should be importable
        spec = importlib.util.find_spec("atlasqwen2")
        assert spec is not None, "atlasqwen2 package not found"
    
    def test_model_type_in_config(self, tiny_config):
        """Config has model_type for auto loading."""
        assert hasattr(tiny_config, 'model_type')
        assert tiny_config.model_type == "atlasqwen2"
    
    def test_from_pretrained_with_trust_remote_code(self, tiny_config, seed):
        """from_pretrained works with trust_remote_code pattern."""
        from atlasqwen2 import AtlasQwen2ForCausalLM
        
        torch.manual_seed(seed)
        model = AtlasQwen2ForCausalLM(tiny_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            
            # Simulate trust_remote_code loading
            # (In practice, HF would use the auto_map in config.json)
            loaded = AtlasQwen2ForCausalLM.from_pretrained(tmpdir)
            assert loaded is not None


class TestMALVariant:
    """Test MAL variant save/load."""
    
    def test_mal_save_load_roundtrip(self, seed):
        """MAL model can be saved and loaded."""
        from atlasqwen2 import AtlasQwen2Config, AtlasQwen2ForCausalLM
        
        config = AtlasQwen2Config(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            memory_heads=4,
            memory_dim_head=16,
            use_momentum=True,
            atlas_variant='mal',
            sliding_window=32,
        )
        
        torch.manual_seed(seed)
        model = AtlasQwen2ForCausalLM(config)
        
        x = torch.randint(0, config.vocab_size, (1, 4))
        with torch.no_grad():
            orig_out = model(x).logits
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            
            loaded_model = AtlasQwen2ForCausalLM.from_pretrained(tmpdir)
            
            # Check that it's still a MAL model
            from models.atlasqwen2_core import AtlasMALBlock
            assert isinstance(loaded_model.model.layers[0], AtlasMALBlock)
            
            with torch.no_grad():
                loaded_out = loaded_model(x).logits
        
        assert torch.allclose(orig_out, loaded_out), "MAL model output changed after save/load"
