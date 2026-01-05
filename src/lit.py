import os, math, gc, importlib
import torch
import torch.linalg
import torch.utils.checkpoint
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
import lightning.pytorch as pl
from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_only
from lightning.pytorch.strategies import DeepSpeedStrategy

import pickle
import torch.distributed as dist
from safetensors.torch import load_file
from torch.utils.data import Dataset, DataLoader    

# Optional dependency (only required for DeepSpeed strategies, especially ZeRO-3).
# Keep import safe so CPU-only / non-deepspeed environments can still import this module.
try:
    import deepspeed  # type: ignore
except Exception:  # pragma: no cover
    deepspeed = None

from configs import TrainerCLI_Config, Model_Config, Transformer_Config, Train_Config

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig, FullOptimStateDictConfig

from .state import ModelState

import src.metrics as metrics

from contextlib import nullcontext

from accelerate import init_empty_weights as init_on_meta_device

from src.logger import print0 as print

def console_clear_last_line():
    print('\033[1A', end='\x1b[2K')

def print_gpu_memory(tag: str, rank: int = None):
    """Print GPU memory usage at a specific point. Rank0 only by default."""
    if rank is None and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    elif rank is None:
        rank = 0
    
    if rank != 0:
        return  # Only rank 0 prints
    
    if not torch.cuda.is_available():
        return
    
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9
    print(f"[GPU MEM {tag}] Alloc: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Peak: {max_allocated:.2f}GB")

class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)

class LightningModelWrapper(pl.LightningModule):
    def __init__(self, model:nn.Module, config:TrainerCLI_Config, teacher:nn.Module|None, trainer):
        super().__init__()
        self.model = model
        self.config = config
        self.teacher = teacher
        self.metrics = dict(loss=metrics.Loss(), acc=metrics.Accuracy())
        self.configured = False

        self.trainer = trainer

    def configure_model(self):
        if self.configured:
            return
        self.configured = True

        print("Running configure_model")

        print("configuring student")
        self.configure_specific_model(self.model, maybe_do_reset=True)

        if self.teacher is not None:
            print("configuring teacher")
            self.configure_specific_model(self.teacher, maybe_do_reset=False)
            self.teacher.eval()
            self.teacher.requires_grad_(False)            

        # FIXME - not sure how the resetted and/or loaded weights are getting transferred to the other GPUs on DS1, but it seems to be working??!
        if self.trainer.global_rank == 0:
            if 'deepspeed_stage_3' not in self.config.train.strategy:
                # For full framework resume, Lightning will restore weights from ckpt_path.
                # Don't also do our own weights-loading/reset path.
                if str(os.environ.get("FULL_RESUME", "0")).strip() != "1":
                    self.load_weights()

        # if self.config.train.strategy == 'fsdp':
        #     from functools import partial
        #     from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        #     auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=int(1e6))
        #     #auto_wrap_policy = {models.qwen2.Qwen2DecoderLayer}
        #     activation_checkpointing_policy = None #{ models.qwen2.Qwen2DecoderLayer }

        #     #from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, MixedPrecision, ShardingStrategy
        #     mixed_precision = None
        #     #mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
        #     #mixed_precision = MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
        #     def init_fn(x: torch.nn.Module):
        #         return x.to_empty(device=torch.get_default_device(), recurse=False)

        #     #strategy_obj = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, activation_checkpointing_policy=activation_checkpointing_policy, sync_module_states=True, mixed_precision=mixed_precision, limit_all_gathers=True, use_orig_params=True, param_init_fn=init_fn)
        #     self.model.model = FSDP(self.model.model, device_id=self.trainer.strategy.root_device.index, auto_wrap_policy=auto_wrap_policy, sync_module_states=True, mixed_precision=mixed_precision, limit_all_gathers=True, use_orig_params=True, param_init_fn=init_fn)
        #     if self.teacher is not None:
        #         self.teacher.model = FSDP(self.teacher.model, device_id=self.trainer.strategy.root_device.index, auto_wrap_policy=auto_wrap_policy, sync_module_states=True, mixed_precision=mixed_precision, limit_all_gathers=True, use_orig_params=True, param_init_fn=init_fn)


    def configure_specific_model(self, model, maybe_do_reset):
        do_reset = False
        if maybe_do_reset:
            if self.config.train is not None:
                if self.config.train.load_model == '' or (self.config.train.load_partial and self.config.train.attention_distillation_stage in (0,1)):
                    do_reset = True

        # Full framework resume (trainer.fit(..., ckpt_path=...)) will restore weights/optimizer/step.
        # Do NOT reset params in that case even if load_model is empty.
        if str(os.environ.get("FULL_RESUME", "0")).strip() == "1":
            do_reset = False

        # Strategy helpers
        strategy_name = str(getattr(self.config.train, "strategy", "") or "")
        use_fsdp = "fsdp" in strategy_name
        use_zero3 = "deepspeed_stage_3" in strategy_name

        # HF models (e.g., transformers.Qwen2ForCausalLM) do NOT implement configure_model().
        # They are supported in this repo for HF patching paths (Stage 1) under DS1/DS2, but
        # are NOT supported under FSDP or ZeRO-3 init in our training stack.
        if not hasattr(model, "configure_model"):
            if use_fsdp or use_zero3:
                raise RuntimeError(
                    f"Model {type(model)} does not implement configure_model(), which is required for "
                    f"FSDP/ZeRO-3 init (strategy={strategy_name}). "
                    f"Use a paper-style model implementation with configure_model() for this strategy."
                )
            # DS1/DS2 (and other non-FSDP, non-ZeRO3) path: just move to device+dtype and return.
            dtype_map = {"bf16-true":torch.bfloat16, "bf16-mixed":torch.bfloat16, "16-true":torch.float16, "16-mixed":torch.float16, "32-true":torch.float32}
            dtype = dtype_map[self.trainer.precision]
            print("Moving model to dtype ", dtype)
            model.to(device=self.device, dtype=dtype)
            return

        # Paper behavior for models implementing configure_model():
        # - Most strategies: init on meta, then materialize via to_empty().
        # - ZeRO-3: cannot init on meta.
        if use_zero3:
            model.configure_model()
        else:
            with init_on_meta_device():
                model.configure_model()

        dtype_map = {"bf16-true":torch.bfloat16, "bf16-mixed":torch.bfloat16, "16-true":torch.float16, "16-mixed":torch.float16, "32-true":torch.float32}
        dtype = dtype_map[self.trainer.precision]

        print("Moving model to dtype ", dtype)
        model.to(dtype=dtype)

        # DeepSpeed ZeRO-3: do NOT to_empty() (and do not rely on meta init above).
        # DeepSpeed needs real storage during partitioning; parameters are already materialized.
        if use_zero3:
            return
        if 'fsdp' in self.config.train.strategy:
            # NOTE:
            # The paper-style flow keeps non-rank0 on meta and relies on FSDP sync_module_states to materialize.
            # For Atlas models in this repo, we've observed runtime failures that look like meta/empty storage
            # escaping into backward ("setStorage ... storage of size 0"). To make FSDP robust for our tests,
            # we optionally materialize on ALL ranks (still on CPU by default, so sharding can proceed).
            eager_fsdp_init = str(os.environ.get("FSDP_EAGER_INIT", "")).strip().lower() in ("1", "true")
            if getattr(self.config.model, "classname", "") == "atlasqwen2":
                eager_fsdp_init = True

            if eager_fsdp_init:
                if do_reset:
                    print("FSDP eager init: moving model to empty on", self.device)
                    model.to_empty(device=self.device, recurse=True)
                else:
                    print("FSDP eager init: moving model to empty on cpu (all ranks)")
                    model.to_empty(device=torch.device('cpu'), recurse=True)
            else:
                if self.trainer.global_rank == 0:
                    if do_reset:
                        # NOTE - we could put it on cpu here for truly huge models, but it's a LOT faster to reset parameters on GPU
                        print("Moving model to empty on", self.device)
                        model.to_empty(device=self.device, recurse=True)
                    else:
                        print("Moving model to empty on cpu")
                        model.to_empty(device=torch.device('cpu'), recurse=True)
                # else leave it as a meta tensor on non-zero ranks... FSDP will convert
        else:
            print("Moving model to empty on", self.device)
            model.to_empty(device=self.device, recurse=True)

        if self.trainer.global_rank == 0:
            # reset parameters, if needed
            if maybe_do_reset:
                # NOTE - we do this on GPU because it's a lot faster! (even though it might not fit for truly giant models)
                if self.config.train is not None:
                    if self.config.train.load_model == '' or (self.config.train.load_partial and self.config.train.attention_distillation_stage in (0,1)):
                        print("Resetting parameters")
                        for submodule in model.modules():
                            if hasattr(submodule, 'reset_parameters'):
                                submodule.reset_parameters()

            # Move to CPU for FSDP/DeepSpeed sharding.
            if 'deepspeed_stage_3' in self.config.train.strategy or 'fsdp' in self.config.train.strategy:
                print("Moving model back to CPU, so giant models have room to get sharded")
                model.to(device=torch.device('cpu'))

    def configure_gradient_clipping(
            self,
            optimizer,
            gradient_clip_val = None,
            gradient_clip_algorithm = None,
    ):
        if gradient_clip_val is None or gradient_clip_val <= 0:
            return
        if 'fsdp' in self.config.train.strategy:
            assert gradient_clip_algorithm in ('norm', None), gradient_clip_algorithm
            #self.model.clip_grad_norm_(gradient_clip_val)
            #self.clip_grad_by_norm(optimizer, clip_val)
            #self.clip_gradients(optimizer, gradient_clip_val, gradient_clip_algorithm)
            #if gradient_clip_algorithm == 'norm':
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_val)
            self.trainer.strategy.model.clip_grad_norm_(gradient_clip_val) #self.config.train.gradient_clip_val)
            #self.model.model.clip_grad_norm_(gradient_clip_val) #self.config.train.gradient_clip_val)
        else:
            self.clip_gradients(optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm)


    def save_weights(self, path):
        print("saving ", path)
        config = self.config

        model = self.model
        if 'fsdp' in config.train.strategy:
            # NOTE - this is how we get the FSDP wrapped model - if you use self you won't get the right output saved!!!
            model:nn.Module = self.trainer.strategy.model
            # annoyingly, we are REQUIRED to get the state dict from the FSDP module, which is only the top level LightningModelWrapper
            # so, get it, then edit the dict to remove the `model.` prefix

            assert(any(isinstance(m, FSDP) for m in model.modules()))
            # FIXME - context manager was crashing on release
            FSDP.set_state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            )
            # with FSDP.state_dict_type(
            #     model,
            #     StateDictType.FULL_STATE_DICT,
            #     FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            # ):
            save_dict = model.state_dict()
            if self.trainer.global_rank == 0:
                for k in list(save_dict.keys()):
                    if k.startswith('teacher.'):
                        del save_dict[k]
                    elif k.startswith('model.'):
                        save_dict[k[len('model.'):]] = save_dict[k].bfloat16()
                        del save_dict[k]
        elif 'deepspeed_stage_3' not in config.train.strategy:
            save_dict = model.state_dict()
        else:
            # FIXME - this would save the whole model as well as optimizer state and dataset state
            #self.trainer.save_checkpoint(path, weights_only=True,)

            def save(module: torch.nn.Module, prefix: str = "", save_dict:dict|None=None) -> dict:
                if save_dict is None:
                    save_dict = {}
                #print("saving prefix", prefix)
                with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
                    if deepspeed.comm.get_rank() == 0:
                        for n, p in module.named_parameters():
                            save_dict[prefix + n] = p.detach().cpu()

                for name, child in module._modules.items():
                    if child is not None:
                        save(child, prefix + name + ".", save_dict)
                        
                return save_dict

            save_dict = save(model)

        # Convert Stage 1 checkpoint (HF patching) to Stage 2/3 format (Pure LMM)
        # Stage 1: HF structure with wrapper: model.layers.X.self_attn.memory.*
        # Stage 2/3: Pure LMM structure: model.layers.X.memory.*
        # Also: post_attention_layernorm → post_memory_layernorm
        if config.train.attention_distillation_stage == 1:
            # Stage 1 saving: This checkpoint will be loaded by Stage 2 (Pure LMM)
            # Need to remap HF structure → Pure LMM structure
            new_save_dict = {}
            for k in list(save_dict.keys()):
                # Remove teacher attention weights (not needed in Stage 2+)
                if '.teacher_attn.' in k:
                    continue
                
                k_new = k
                
                # Remove student_attn wrapper (from AttentionDistillationWrapper)
                if '.student_attn.' in k_new:
                    k_new = k_new.replace('.student_attn.', '.')
                
                # Convert HF structure to Pure LMM structure
                # model.layers.X.self_attn.memory.* → model.layers.X.memory.*
                if '.self_attn.memory.' in k_new:
                    k_new = k_new.replace('.self_attn.memory.', '.memory.')
                
                # post_attention_layernorm → post_memory_layernorm
                if 'post_attention_layernorm' in k_new:
                    k_new = k_new.replace('post_attention_layernorm', 'post_memory_layernorm')
                    
                new_save_dict[k_new] = save_dict[k]
            
            save_dict = new_save_dict
        elif config.model.hf_path != '':
            # Stage 2/3 with HF-patched checkpoint: Also remap HF structure → Pure LMM structure
            new_save_dict = {}
            for k in list(save_dict.keys()):
                # Remove teacher attention weights
                if '.teacher_attn.' in k:
                    continue
                
                k_new = k
                    
                # Remove student_attn wrapper (from AttentionDistillationWrapper)
                if '.student_attn.' in k_new:
                    k_new = k_new.replace('.student_attn.', '.')
                    
                # Convert HF patching format to Pure LMM format
                # self_attn.memory.* → memory.*
                if '.self_attn.memory.' in k_new:
                    k_new = k_new.replace('.self_attn.memory.', '.memory.')
                    
                # post_attention_layernorm → post_memory_layernorm
                if 'post_attention_layernorm' in k_new:
                    k_new = k_new.replace('post_attention_layernorm', 'post_memory_layernorm')
                    
                new_save_dict[k_new] = save_dict[k]
            
            save_dict = new_save_dict

        if self.trainer.is_global_zero:
            torch.save(save_dict, path)

    def load_weights(self):
        config = self.config
        ckpt_path = config.train.load_model
        if ckpt_path != '':
            self.load_specific_model_weights(self.model, ckpt_path)
            # Set gradients based on training stage (only for custom models with set_grads method)
            if hasattr(self.model, 'set_grads'):
                self.model.set_grads()
        
        if self.teacher is not None and config.train.teacher is not None:
           teacher_ckpt_path = config.train.teacher.path
           if teacher_ckpt_path != '':
               # Paper expects teacher.path to exist (local checkpoint).
               # Atlas configs may use teacher.path as an enable-flag (or point to a file produced elsewhere).
               # Be robust: only load if it exists; otherwise keep initialized weights.
               if os.path.exists(teacher_ckpt_path):
                   self.load_specific_model_weights(self.teacher, teacher_ckpt_path)
               else:
                   print(f"[teacher] WARNING: teacher.path='{teacher_ckpt_path}' not found; skipping teacher weight load.")

    def load_specific_model_weights(self, model, ckpt_path):
        config = self.config

        if 'fsdp' in config.train.strategy:
            if self.trainer.global_rank != 0:
                return

        print("Loading ", ckpt_path)       
        if 'deepspeed_stage_3' in config.train.strategy and deepspeed.comm.get_rank() != 0:
            load_dict = None
        else:
            if ckpt_path.lower().endswith('.safetensors'):
                load_dict = load_file(ckpt_path, device='cpu')
            else:
                load_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
                
            # FIXME - this provides copies of tied weights, which isn't desirable for all models or when we want them to actually be tied
            if 'lm_head.weight' not in load_dict:
                load_dict['lm_head.weight'] = load_dict['model.embed_tokens.weight']
                
            # FIXME - this gives the inline teacher the copies it needs of the self_attn weights
            if config.train.attention_distillation_stage == 1:
                # Two possible checkpoint formats may be used for Stage 1:
                # 1) HF-style (teacher weights live under `.self_attn.*`) -> we need to duplicate to `.teacher_attn.*`
                # 2) "Stage2-format" produced by save_weights() remapping (`.memory.*`, `post_memory_layernorm`, etc).
                #    For Stage 1 resume we support loading this by mapping keys back into the wrapper's student path:
                #      model.layers.X.memory.* -> model.layers.X.self_attn.student_attn.memory.*
                #      post_memory_layernorm   -> post_attention_layernorm
                keys = list(load_dict.keys())
                is_stage2_format = any('.memory.' in k for k in keys) and not any('.student_attn.' in k for k in keys)
                if is_stage2_format:
                    new_load_dict = {}
                    for k, v in load_dict.items():
                        k_new = k
                        # Memory remap (stage2 -> stage1 wrapper)
                        if '.memory.' in k_new:
                            k_new = k_new.replace('.memory.', '.self_attn.student_attn.memory.')
                        # LayerNorm name remap (stage2 -> stage1 HF layer)
                        if 'post_memory_layernorm' in k_new:
                            k_new = k_new.replace('post_memory_layernorm', 'post_attention_layernorm')
                        new_load_dict[k_new] = v
                    load_dict = new_load_dict
                else:
                    # HF-style teacher duplication (avoid copying student paths)
                    keys = list(load_dict.keys())
                    for k in keys:
                        if '.self_attn.' in k and '.self_attn.student_attn.' not in k:
                            load_dict[k.replace('self_attn', 'teacher_attn')] = load_dict[k]

        strict = not config.train.load_partial #and config.train.attention_distillation_stage != 2

        if 'deepspeed_stage_3' not in config.train.strategy:
            model.load_state_dict(load_dict, strict=strict)
            print("Loaded ", ckpt_path)       
            return

        # # simple version that takes a lot more CPU RAM
        # self.trainer.strategy.model_to_device()
        # with deepspeed.zero.GatheredParameters(list(model.parameters()), modifier_rank=0):
        #     if deepspeed.comm.get_rank() == 0:
        #         model.load_state_dict(load_dict, strict=False)

        # print("Loaded ", ckpt_path)       
        # return

        # see https://github.com/microsoft/DeepSpeed/blob/8cded575a94e296fee751072e862304676c95316/deepspeed/runtime/zero/partition_parameters.py#L2172
        # see trainer.strategy.load_model_state_dict https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/strategies/deepspeed.html

        """Overrides the normal load_state_dict behaviour in PyTorch to ensure we gather parameters that may be sharded
        across processes before loading the state dictionary when using ZeRO stage 3. This is then automatically synced
        across processes.

        Args:
            ckpt: The ckpt file.

        """

        #assert self.lightning_module is not None

        def load(module: torch.nn.Module, prefix: str = "") -> None:
            # because zero3 puts placeholders in model params, this context
            # manager gathers (unpartitions) the params of the current layer, then loads from
            # the state dict and then re-partitions them again
            with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
                if deepspeed.comm.get_rank() == 0:
                    print("loading prefix", prefix)
                    missing_keys = []
                    unexpected_keys = []
                    error_msgs = []

                    # copy state_dict so _load_from_state_dict can modify it
                    metadata = getattr(load_dict, "_metadata", None)
                    state_dict = load_dict.copy()
                    if metadata is not None:
                        state_dict._metadata = metadata

                    local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
                
                    for n, p in module.named_parameters(recurse=False):
                        nn = prefix + n
                        print(nn, nn in state_dict)
                    module._load_from_state_dict(
                        state_dict=state_dict,
                        prefix=prefix,
                        local_metadata=local_metadata,
                        strict=strict,
                        missing_keys=missing_keys,
                        unexpected_keys=unexpected_keys,
                        error_msgs=error_msgs,
                    )
                    if len(error_msgs) > 0:
                        print("ERROR", error_msgs)
                        exit(0)

            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(model, prefix="")
        print("Loaded ", ckpt_path)       

    def forward(self, idx, last_model_state:ModelState|None = None):
        return self.model.forward(idx, last_model_state)
    
    def configure_optimizers(self):
        # what the heck, we had to do this before the optimizers are loaded or the loaded weights wouldn't 'take' into the optimizer!!!
        if 'deepspeed_stage_3' in self.config.train.strategy:
            self.load_weights()

        train_config = self.config.train

        if self.config.model.hf_path != '':
            optim_groups = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if p.requires_grad # (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": train_config.weight_decay, "my_lr_scale": 1.0, 'name':'lr_1x',
                },
                # {
                #     "params": [
                #         p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                #     ],
                #     "weight_decay": 0.0,
                # },
            ]
        else:
            optim_groups = self.model.get_optim_groups()

        print("Configuring optimizers!!!")

        betas = (train_config.beta1, train_config.beta2)
        if train_config.optimizer == 'adamw':
            import torch.optim
            self.optimizer = torch.optim.AdamW(optim_groups, lr=train_config.lr_init, betas=betas, eps=train_config.adam_eps)
            return self.optimizer
        if train_config.optimizer == 'dsfusedadamw':
            from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
            return FusedAdam(optim_groups, lr=train_config.lr_init, betas=betas, eps=train_config.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
        if train_config.optimizer == 'adamwschedulefree':
            import schedulefree
            self.optimizer = schedulefree.AdamWScheduleFree(optim_groups, lr=train_config.lr_init, betas=betas, eps=train_config.adam_eps)
            return self.optimizer
        if train_config.optimizer == 'radamschedulefree':
            import schedulefree
            self.optimizer = schedulefree.RAdamScheduleFree(optim_groups, lr=train_config.lr_init, betas=betas, eps=train_config.adam_eps)
            return self.optimizer
        if train_config.optimizer == 'adam8bit':
            import bitsandbytes as bnb
            return bnb.optim.Adam(optim_groups, train_config.lr_init, betas, optim_bits=8, percentile_clipping=5)
        if train_config.optimizer == 'lion8bit':
            import bitsandbytes as bnb
            return bnb.optim.Lion(optim_groups, train_config.lr_init, betas, optim_bits=8, percentile_clipping=5)
        if train_config.optimizer == 'lion' or train_config.optimizer == 'lionfp16':
            from src.optimizers.lion import Lion
            return Lion(optim_groups, train_config.lr_init, betas, use_fp16=train_config.optimizer == 'lionfp16')
        if self.deepspeed_offload:
            from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
            return DeepSpeedCPUAdam(optim_groups, lr=train_config.lr_init, betas=betas, eps=train_config.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
        return None# FusedAdam(optim_groups, lr=train_config.lr_init, betas=betas, eps=train_config.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)

    def on_fit_start(self) -> None:
        if 'schedulefree' in self.config.train.optimizer:
            self.optimizer.train()

    def on_predict_start(self) -> None:
        if 'schedulefree' in self.config.train.optimizer:
            self.optimizer.eval()

    def on_validation_model_eval(self) -> None:
        if 'schedulefree' in self.config.train.optimizer:
            self.model.eval()
            self.optimizer.eval()

    def on_validation_model_train(self) -> None:
        if 'schedulefree' in self.config.train.optimizer:
            self.model.train()
            self.optimizer.train()

    def on_test_model_eval(self) -> None:
        if 'schedulefree' in self.config.train.optimizer:
            self.model.eval()
            self.optimizer.eval()

    def on_test_model_train(self) -> None:
        if 'schedulefree' in self.config.train.optimizer:
            self.model.train()
            self.optimizer.train()

    def on_predict_model_eval(self) -> None:  # redundant with on_predict_start()
        if 'schedulefree' in self.config.train.optimizer:
            self.model.eval()
            self.optimizer.eval()

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False


    def _get_loss_logits_preds(self, batch, batch_idx, last_model_state):
        # Distillation KL chunking:
        # - Larger = faster, smaller = lower peak memory.
        # - FSDP tends to have slightly higher peaks (all-gathers + allocator fragmentation),
        #   so we default to smaller chunks under FSDP to avoid OOM.
        try:
            distill_chunk_len = int(os.environ.get("DISTILL_KL_CHUNK_LEN", "0") or 0)
        except Exception:
            distill_chunk_len = 0
        if distill_chunk_len <= 0:
            distill_chunk_len = 256 if "fsdp" in str(self.config.train.strategy).lower() else 512

        x, y = batch
        
        print_gpu_memory("step_start")

        B, T = x.shape
        causal_mask = torch.full((T, T), fill_value=-torch.inf, dtype=torch.bfloat16, device=x.device).triu(1)
        causal_mask = causal_mask[None, None, :, :].expand(B, 1, -1, -1)

        token_ids = x
        eos_token_id = 151643 # "<|endoftext|>"
        #loss_mask = token_ids != eos_token_id # shouldn't count loss of prediction from an EOS token, since it's got no information useful for predicting it!
        #loss_mask = torch.ones_like(token_ids)

        # inverted_loss_mask = (token_ids == eos_token_id).long()
        # for i in range(len(inverted_loss_mask)):
        #     row_mask = inverted_loss_mask[i]
        #     first_occurrence = row_mask.argmax() if row_mask.any() else len(inverted_loss_mask[i])
        #     loss_mask[i, first_occurrence + 1:] = False

        if self.training and self.config.train.attention_distillation_stage in (0, 11, 1):
            output_attentions = self.config.train.attention_distillation_stage == 0
            output_post_attention_hidden_states = self.config.train.attention_distillation_stage in (11, 1)
            # special code for attention output and/or attention matrix loss
            if self.config.model.hf_path != '':
                # For HuggingFace models with AttentionDistillationWrapper patched layers,
                # the alignment loss is computed inside each wrapper and stored in _last_alignment_loss
                # Avoid requesting attention weights when not needed. Some HF attention implementations
                # (e.g., sdpa/flash) warn that output_attentions=True is unsupported.
                # Stage 1 alignment uses wrapper-stored _last_alignment_loss and does not require weights.
                results = self.model.forward(x, output_hidden_states=False, output_attentions=output_attentions)
                
                # Collect alignment losses from wrapper layers
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                    losses = []
                    for layer in self.model.model.layers:
                        if hasattr(layer.self_attn, '_last_alignment_loss') and layer.self_attn._last_alignment_loss is not None:
                            losses.append(layer.self_attn._last_alignment_loss)
                    
                    if len(losses) > 0:
                        training_loss = torch.stack(losses, dim=0).mean()
                    elif results.attentions is not None and len(results.attentions) > 0:
                        training_loss = torch.stack(results.attentions, dim=0).mean()
                    else:
                        # Fallback: use logits-based loss
                        training_loss = F.cross_entropy(results.logits.view(-1, results.logits.size(-1)), y.view(-1))
                else:
                    if results.attentions is not None:
                        training_loss = torch.stack(results.attentions, dim=0).mean()
                    else:
                        training_loss = F.cross_entropy(results.logits.view(-1, results.logits.size(-1)), y.view(-1))
                #reported_loss = results.loss
            elif self.config.model.classname != '':
                results = self.model.forward(x, output_hidden_states=False, output_attentions=output_attentions, output_post_attention_hidden_states=output_post_attention_hidden_states)
            else:
                results = self.model.forward(x, return_dict=False, attention_mask=causal_mask, output_hidden_states=False, output_attentions=output_attentions, output_post_attention_hidden_states=output_post_attention_hidden_states)

            if self.config.model.hf_path == '':
                if self.config.train.attention_distillation_stage == 0:
                    #repeated_loss_mask = loss_mask.repeat(len(results.attentions), 1)
                    training_loss = torch.linalg.matrix_norm(torch.cat(results.attentions, dim=0) - torch.cat(results.student_attentions, dim=0))
                    #training_loss = training_loss * repeated_loss_mask
                    training_loss = training_loss.float().mean() / results.attentions[0].size(-1) # FIXME - not quite perfect because the average will be brought down by uncounted EOS tokens
                else: # self.config.train.attention_distillation_stage == 1:
                    #repeated_loss_mask = loss_mask.repeat(len(results.post_attention_hidden_states), 1)
                    training_loss = torch.linalg.vector_norm(torch.cat(results.post_attention_hidden_states, dim=0) - torch.cat(results.student_post_attention_hidden_states, dim=0), dim=-1)
                    #training_loss = training_loss * repeated_loss_mask
                    training_loss = training_loss.float().mean() * (results.post_attention_hidden_states[0].size(-1) ** -0.5) # FIXME - not quite perfect because the average will be brought down by uncounted EOS tokens
            reported_loss = training_loss
            logits = torch.tensor([], device=x.device)
            preds = torch.zeros_like(y)
            next_model_state = last_model_state
        else:
            
            if self.training and self.config.train.attention_distillation_stage == 23:
                print_gpu_memory("before_student_forward")
                results = self.model.forward(x, output_hidden_states=True, output_attentions=False, output_post_attention_hidden_states=False)
                print_gpu_memory("after_student_forward")
                
                self.teacher.eval()
                with torch.no_grad():
                    teacher_results = self.teacher.forward(x, output_hidden_states=True)
                print_gpu_memory("after_teacher_forward")
                #reported_loss = training_loss = torch.linalg.vector_norm(torch.cat(teacher_results.hidden_states[1:], dim=0) - torch.cat(results.hidden_states[1:], dim=0), dim=-1).mean() * (results.hidden_states[0].size(-1) ** -0.5)

                student_logits = results.logits
                flat_student_logits = student_logits.view(-1, student_logits.size(-1))
                with torch.no_grad():
                    preds = student_logits.argmax(dim=-1)

                teacher_logits = teacher_results.logits
                flat_teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))

                chunk_len = distill_chunk_len
                n_chunks = (flat_student_logits.size(0) + chunk_len - 1) // chunk_len

                # memory saving measure, because otherwise kl_div tried to allocate everything all at once
                distillation_loss = torch.tensor(0.0, device=flat_student_logits.device, dtype=flat_student_logits.dtype)
                for c in range(0, flat_student_logits.size(0), chunk_len):
                    student_log_softmax = F.log_softmax(flat_student_logits[c:c+chunk_len], dim=-1)
                    teacher_log_softmax = F.log_softmax(flat_teacher_logits[c:c+chunk_len], dim=-1)
                    distillation_loss = distillation_loss + F.kl_div(
                        student_log_softmax,
                        teacher_log_softmax,
                        log_target=True,
                        reduction='batchmean'
                    )
                distillation_loss = distillation_loss / n_chunks
                print_gpu_memory("after_kl_loss")

                #reported_loss = training_loss = distillation_loss + torch.linalg.vector_norm(teacher_results.hidden_states[-1] - results.hidden_states[-1], dim=-1).mean() * (results.hidden_states[0].size(-1) ** -0.5)
                hidden_states_loss = torch.tensor(0.0, device=flat_student_logits.device, dtype=flat_student_logits.dtype)
                for layer_id in range(1,len(results.hidden_states)-1):
                    hidden_states_loss = hidden_states_loss + torch.linalg.vector_norm(teacher_results.hidden_states[layer_id] - results.hidden_states[layer_id], dim=-1).mean() / (len(results.hidden_states)-2) * (results.hidden_states[0].size(-1) ** -0.5)
                print_gpu_memory("after_hidden_loss")
                reported_loss = training_loss = distillation_loss + hidden_states_loss
                logits = torch.tensor([], device=x.device)
                return reported_loss, training_loss, logits, preds, last_model_state

            # if self.training and self.config.train.attention_distillation_stage == 2:
            #     results = self.model.forward(x, output_hidden_states=True, output_attentions=False, output_post_attention_hidden_states=False)
            #     self.teacher.eval()
            #     with torch.no_grad():
            #         teacher_results = self.teacher.forward(x, output_hidden_states=True)
            #     # FIXME - argh this doesn't work because FSDP slices up the lm_head.weight so we can't use it here
            #     print(results.hidden_states[-1].view(-1, results.hidden_states[-1].size(-1)).shape, teacher_results.hidden_states[-1].view(-1, teacher_results.hidden_states[-1].size(-1)).shape, self.model.lm_head.weight.shape, self.teacher.lm_head.weight.shape)
            #     exit()
            #     reported_loss = training_loss = self.kl_div_loss(results.hidden_states[-1].view(-1, results.hidden_states[-1].size(-1)), teacher_results.hidden_states[-1].view(-1, teacher_results.hidden_states[-1].size(-1)), self.model.lm_head.weight, self.teacher.lm_head.weight)
            #     logits = torch.tensor([], device=x.device)
            #     preds = torch.zeros_like(y)
            #     return reported_loss, training_loss, logits, preds, last_model_state

            if self.config.model.hf_path != '':
                results = self.model(x, output_hidden_states=False)
            elif self.config.model.classname != '':
                results = self.model(x, last_model_state, output_hidden_states=False)
            elif self.config.model.tmix.lower().startswith('qwen2'):
                results = self.model(x, attention_mask=causal_mask, output_hidden_states=False)
            else:
                results = self.model(x, last_model_state)
            if isinstance(results, tuple):
                logits = results[0]
                next_model_state = results[1]
            elif isinstance(results, torch.Tensor):
                logits = results
                next_model_state = last_model_state
            else:
                logits = results.logits
                next_model_state = last_model_state

            flat_student_logits = logits.view(-1, logits.size(-1))
            flat_labels = y.view(-1)
            #flat_loss_mask = loss_mask.view(-1)

            reported_loss = training_loss = distillation_loss = ce_loss = torch.tensor(0.0, device=flat_student_logits.device, dtype=flat_student_logits.dtype)

            chunk_loss_calcs = self.config.train.attention_distillation_stage in (-1, 2, 21, 31)
            chunk_len = distill_chunk_len
            n_chunks = (flat_student_logits.size(0) + chunk_len - 1) // chunk_len
            if not self.training or self.teacher is None or self.config.train.teacher.ce_weight > 0:
                if not chunk_loss_calcs:
                    ce_loss = F.cross_entropy(flat_student_logits, flat_labels) #, reduction='mean') #, ignore_index=eos_token_id)
                else:
                    # memory saving measure, because otherwise cross_entropy tried to allocate everything all at once
                    ce_loss = torch.tensor(0.0, device=flat_student_logits.device, dtype=torch.float) #flat_student_logits.dtype)
                    for c in range(0, flat_student_logits.size(0), chunk_len):
                        ce_loss = ce_loss + F.cross_entropy(flat_student_logits[c:c+chunk_len], flat_labels[c:c+chunk_len]) #, reduction='sum')
                            #ignore_index=eos_token_id, reduction='sum')
                    #ce_loss = ce_loss / flat_student_logits.size(0) 
                    ce_loss = ce_loss / n_chunks
                    # / (flat_loss_mask.sum() + 1e-8) # FIXME - this isn't right to divide by this
                reported_loss = training_loss = ce_loss

            with torch.no_grad():
                preds = logits.argmax(dim=-1)

            if self.training and self.teacher is not None:
                self.teacher.eval()
                # IMPORTANT:
                # For FSDP, materializing full teacher logits [B*T, vocab] can easily OOM (~300MB+ just for logits).
                # Instead, prefer computing teacher hidden states once, then apply teacher.lm_head + log_softmax in
                # token chunks while accumulating KL, without ever storing full teacher logits.
                teacher_hidden_flat = None
                flat_teacher_logits = None  # fallback only
                with torch.no_grad():
                    if hasattr(self.teacher, "model") and hasattr(self.teacher, "lm_head"):
                        t_core = None
                        # Paper-style qwen2 teacher: teacher.model(token_ids, last_model_state=..., ...)
                        try:
                            t_core = self.teacher.model(
                                x,
                                last_model_state=None,
                                output_hidden_states=False,
                                output_attentions=False,
                                output_post_attention_hidden_states=False,
                            )
                        except TypeError:
                            # HF-style teacher: teacher.model(input_ids=..., attention_mask=...)
                            try:
                                t_core = self.teacher.model(input_ids=x)
                            except TypeError:
                                # Last resort: call as positional (some HF models)
                                try:
                                    t_core = self.teacher.model(x)
                                except Exception:
                                    t_core = None

                        if t_core is not None:
                            t_hidden = getattr(t_core, "logits", None)
                            if t_hidden is None:
                                t_hidden = getattr(t_core, "last_hidden_state", None)
                            if t_hidden is None and isinstance(t_core, (tuple, list)) and len(t_core) > 0 and torch.is_tensor(t_core[0]):
                                t_hidden = t_core[0]
                            # Expect [B, T, D]
                            if torch.is_tensor(t_hidden) and t_hidden.dim() == 3:
                                teacher_hidden_flat = t_hidden.reshape(-1, t_hidden.size(-1))

                    # Fallback path: get full teacher logits (may OOM under FSDP).
                    if teacher_hidden_flat is None:
                        teacher_results = self.teacher.forward(x)
                        if isinstance(teacher_results, tuple):
                            teacher_logits = teacher_results[0]
                        elif isinstance(results, torch.Tensor):
                            teacher_logits = teacher_results
                        else:
                            teacher_logits = teacher_results.logits
                        flat_teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))

                if not chunk_loss_calcs:
                    # Prefer chunked KL even in the "non-chunk" mode if we can avoid full teacher logits.
                    if teacher_hidden_flat is not None:
                        distillation_loss = torch.tensor(0.0, device=flat_student_logits.device, dtype=torch.float)
                        for c in range(0, flat_student_logits.size(0), chunk_len):
                            student_log_softmax = F.log_softmax(flat_student_logits[c:c+chunk_len], dim=-1)
                            teacher_logits_chunk = self.teacher.lm_head(teacher_hidden_flat[c:c+chunk_len])
                            teacher_log_softmax = F.log_softmax(teacher_logits_chunk, dim=-1)
                            distillation_loss = distillation_loss + F.kl_div(
                                student_log_softmax,
                                teacher_log_softmax,
                                log_target=True,
                                reduction='sum',
                            )
                        distillation_loss = distillation_loss / flat_labels.size(0)
                    else:
                        distillation_loss = F.kl_div(
                            F.log_softmax(flat_student_logits, dim=-1),
                            F.log_softmax(flat_teacher_logits, dim=-1),
                            log_target=True,
                            reduction='batchmean'
                        )
                    # distillation_loss = F.kl_div(
                    #     torch.log( F.softmax(flat_student_logits, dim=-1) * flat_loss_mask + 1e-8 ),
                    #     F.softmax(flat_teacher_logits, dim=-1) * flat_loss_mask + 1e-8,
                    #     log_target=False,
                    #     reduction='sum'
                    # )
                    # distillation_loss = distillation_loss / (flat_loss_mask.sum() + 1e-8)
                else:
                    # memory saving measure, because otherwise kl_div tried to allocate everything all at once
                    distillation_loss = torch.tensor(0.0, device=flat_student_logits.device, dtype=torch.float) #flat_student_logits.dtype)
                    for c in range(0, flat_student_logits.size(0), chunk_len):
                        # chunk_loss_mask = flat_loss_mask[c:c+chunk_len].unsqueeze(-1)
                        # student_log_softmax = torch.log( F.softmax(flat_student_logits[c:c+chunk_len], dim=-1) * chunk_loss_mask + 1e-8 )
                        # teacher_softmax = F.softmax(flat_teacher_logits[c:c+chunk_len], dim=-1) * chunk_loss_mask + 1e-8
                        # distillation_loss = distillation_loss + F.kl_div(
                        #     student_log_softmax,
                        #     teacher_softmax,
                        #     log_target=False,
                        #     reduction='sum',
                        # )
                        # student_log_softmax = torch.log_softmax(flat_student_logits[c:c+chunk_len], dim=-1) * chunk_loss_mask
                        # teacher_log_softmax = torch.log_softmax(flat_teacher_logits[c:c+chunk_len], dim=-1) * chunk_loss_mask
                        # distillation_loss = distillation_loss + F.kl_div(
                        #     student_log_softmax,
                        #     teacher_log_softmax,
                        #     log_target=True,
                        #     reduction='sum',
                        # )
                        student_log_softmax = F.log_softmax(flat_student_logits[c:c+chunk_len], dim=-1)
                        if teacher_hidden_flat is not None:
                            teacher_logits_chunk = self.teacher.lm_head(teacher_hidden_flat[c:c+chunk_len])
                            teacher_log_softmax = F.log_softmax(teacher_logits_chunk, dim=-1)
                        else:
                            teacher_log_softmax = F.log_softmax(flat_teacher_logits[c:c+chunk_len], dim=-1)
                        distillation_loss = distillation_loss + F.kl_div(
                             student_log_softmax,
                             teacher_log_softmax,
                             log_target=True,
                             reduction='sum'
                        )
                    distillation_loss = distillation_loss / flat_labels.size(0)
                    # distillation_loss = distillation_loss / (flat_loss_mask.sum() + 1e-8)

                training_loss = distillation_loss * self.config.train.teacher.kl_weight
                if self.config.train.teacher.ce_weight > 0:
                    training_loss = training_loss + ce_loss * self.config.train.teacher.ce_weight
                # FIXME - reporting disillation loss, but we can still see accuracy
                reported_loss = training_loss
                if batch_idx % 10 == 0:
                    print(f"kl_div:{distillation_loss.item()}, ce_loss:{ce_loss.item()}")

        if reported_loss.isinf().any():
            raise Exception("reported loss was infinite")

        if reported_loss.isnan().any():
            raise Exception("reported loss was NaN")

        if training_loss.isinf().any():
            raise Exception("loss was infinite")

        if training_loss.isnan().any():
            raise Exception("loss was NaN")

        return reported_loss, training_loss, logits, preds, next_model_state
    
    def get_real_global_step(self): return int(self.trainer.global_step + self.config.train.epoch_begin * self.config.runtime.epoch_global_steps)
    def get_real_tokens(self): return self.get_real_global_step() * self.config.model.ctx_len * self.config.runtime.global_step_bsz
    def get_real_progress(self):
        config = self.config
        progress = self.get_real_tokens() / abs(config.train.my_exit_tokens)
        progress = max(0, min(1, progress))
        return progress
    def get_lr_progress(self):
        config = self.config
        warmup_tokens = config.train.warmup_steps * config.model.ctx_len * config.runtime.global_step_bsz
        tokens_ex_warmup = abs(config.train.my_exit_tokens) - warmup_tokens
        real_progress_ex_warmup = (self.get_real_tokens() - warmup_tokens) / tokens_ex_warmup
        real_progress_ex_warmup = max(0, min(1, real_progress_ex_warmup))
        lr_progress = (real_progress_ex_warmup - config.train.lr_wait) / (config.train.lr_endpoint - config.train.lr_wait)
        lr_progress = max(0, min(1, lr_progress))
        return lr_progress


    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        model_state = None

        loss, training_loss, logits, preds, model_state = self._get_loss_logits_preds((inputs, labels), batch_idx, model_state)
        margs = metrics.MetricArgs(inputs, logits, preds, labels, loss)
        # FIXME - sync from other devices/nodes here
        for metric in self.metrics.values():
            metric.update(margs)
        if self.trainer.is_global_zero:
            self.log("loss", float(loss), prog_bar=True, on_step=True)#, rank_zero_only=True)
            if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
                if (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0:
                    logdict = dict(tokens = self.get_real_tokens())
                    #str = f"epoch:{self.current_epoch} token:{self.all_nodes_tokens_processed:,} step:{batch_idx} "
                    for name, metric in self.metrics.items():
                        metric_value = metric.compute()
                        logdict['train/' + name] = metric_value
                        metric.clear()
                        #str += f'{name}:{metric_value:.4f} '
                    #str += f"{gb:.1f}gb {int(ms_per)}ms {ktok_per_sec:.2f}kT/s {self.total_runtime:.1f}sec"
                    #print(str)
                    if len(self.config.train.wandb) > 0:
                        self.trainer.my_wandb.log(logdict, step=self.get_real_global_step(), commit=True)

        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))

        #if logits.size(0) > 0:
        #    return L2Wrap.apply(training_loss, logits)
        #else:
        return training_loss

    def on_validation_epoch_start(self):
        if self.trainer.is_global_zero:
            print(f"STARTING VALIDATION")
            print()

            # clear metrics
            for metric in self.metrics.values():
                metric.compute()

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            logdict = dict(tokens = self.get_real_tokens())
            str = f"VALIDATION COMPLETE. "
            for name, metric in self.metrics.items():
                metric_value = metric.compute()
                logdict["val/" + name] = metric_value
                str += f"{metric_value:.4f} "
                metric.clear()
            if len(self.config.train.wandb) > 0:
                self.trainer.my_wandb.log(logdict, step=self.get_real_global_step(), commit=True)

            console_clear_last_line()
            print(str)
            print()

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        loss, training_loss, logits, preds, next_block_states = self._get_loss_logits_preds(batch, batch_idx, None)
        margs = metrics.MetricArgs(inputs, logits, preds, labels, loss)
        for name, metric in self.metrics.items():
            metric.update(margs)
            # on_epoch causes this to be logged in aggregate rather than per batch
            #self.log('val/'+name, metric.compute(), on_epoch=True, rank_zero_only=True)
            #metric.clear()
        #self.log("tokens", float(self.all_nodes_tokens_processed), on_epoch=True, rank_zero_only=True)
        return loss
    
    def predict_dataloader(self):
        return DataLoader(mnist_predict, batch_size=self.batch_size)
    
    # def training_step_end(self, batch_parts):
    #     if pl.__version__[0]!='2':
    #         all = self.all_gather(batch_parts)
    #         if self.trainer.is_global_zero:
    #             self.trainer.my_loss_all = all
    