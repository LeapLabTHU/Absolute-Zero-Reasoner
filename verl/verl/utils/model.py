# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utilities to create common models from huggingface
"""
import os
import warnings
from typing import Dict, Type, Optional

import numpy as np
import torch
from torch import nn
# from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, MistralForSequenceClassification, GenerationConfig # DELAYED
from verl.models.registry import ModelRegistry

# --- AZR DIAGNOSTIC: Removed global print here ---
# It's better to trace imports via function calls within the Ray worker lifecycle.
# PROC_PID_MODEL_PY = os.getpid()


class LambdaLayer(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def squeeze(x):
    return torch.squeeze(x, dim=-1)


def update_model_config(module_config, override_config_kwargs):
    # This function itself doesn't need transformers, it operates on a config object
    for key, val in override_config_kwargs.items():
        setattr(module_config, key, val)


def get_huggingface_actor_config(model_name: str, override_config_kwargs=None, trust_remote_code=False) -> Dict:
    pid = os.getpid()
    from transformers import AutoConfig # DELAYED IMPORT
    if override_config_kwargs is None:
        override_config_kwargs = {}
    assert isinstance(override_config_kwargs, Dict), \
        f'override_config_kwargs must be a dict, got {type(override_config_kwargs)}'
    
    module_config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    update_model_config(module_config, override_config_kwargs)
    return module_config


def get_generation_config(
    model: str,
    trust_remote_code: bool = False,
) -> Optional["GenerationConfig"]: # Forward reference with string
    pid = os.getpid()
    from transformers import GenerationConfig # DELAYED IMPORT
    try:
        return GenerationConfig.from_pretrained(model)
    except OSError:  # Not found
        try:
            config = get_huggingface_actor_config( # This will have its own prints
                model,
                trust_remote_code=trust_remote_code,
            )
            return GenerationConfig.from_model_config(config)
        except OSError:  # Not found
            return None


def create_huggingface_actor(model_name: str, override_config_kwargs=None, automodel_kwargs=None) -> nn.Module:
    pid = os.getpid()
    from transformers import AutoModelForCausalLM # DELAYED IMPORT
    
    if override_config_kwargs is None:
        override_config_kwargs = {}
    if automodel_kwargs is None:
        automodel_kwargs = {}
    assert isinstance(override_config_kwargs, Dict), \
        f'override_config_kwargs must be a dict, got {type(override_config_kwargs)}'
    
    module_config = get_huggingface_actor_config(model_name, # This will have its own prints
                                                 override_config_kwargs,
                                                 trust_remote_code=automodel_kwargs.get('trust_remote_code', False))
    
    module: nn.Module = AutoModelForCausalLM.from_config(module_config, **automodel_kwargs)
    return module


def create_huggingface_critic(model_name: str, override_config_kwargs=None, automodel_kwargs=None) -> nn.Module:
    pid = os.getpid()
    
    critic_module: nn.Module = create_huggingface_actor(model_name, # This will have its own prints
                                                        override_config_kwargs=override_config_kwargs,
                                                        automodel_kwargs=automodel_kwargs)
    if automodel_kwargs is None:
        automodel_kwargs = {}
    torch_dtype = automodel_kwargs.get('torch_dtype', torch.float32)
    
    if hasattr(critic_module, 'lm_head'):
        critic_module.lm_head = nn.Sequential(nn.Linear(critic_module.config.hidden_size, 1, dtype=torch_dtype),
                                              LambdaLayer(fn=squeeze))
    elif hasattr(critic_module, 'score') and isinstance(getattr(critic_module, 'score'), nn.Linear):
        critic_module.score = nn.Sequential(nn.Linear(critic_module.config.hidden_size, 1, dtype=torch_dtype),
                                             LambdaLayer(fn=squeeze))
    else:
        critic_module.value_head = nn.Sequential(nn.Linear(critic_module.config.hidden_size, 1, dtype=torch_dtype),
                                               LambdaLayer(fn=squeeze))

    return critic_module


def get_model_size(model: nn.Module, scale='auto'):
    n_params = sum(p.numel() for p in model.parameters())
    if scale == 'auto':
        if n_params > 1e9: scale = 'B'
        elif n_params > 1e6: scale = 'M'
        elif n_params > 1e3: scale = 'K'
        else: scale = ''
    if scale == 'B': n_params /= 1e9
    elif scale == 'M': n_params /= 1e6
    elif scale == 'K': n_params /= 1e3
    elif scale == '': pass
    else: raise NotImplementedError(f'Unknown scale {scale}')
    return n_params, scale

def print_model_size(model: nn.Module, name: str = None):
    n_params, scale = get_model_size(model, scale='auto')
    if name is None: name = model.__class__.__name__
    print(f'{name} contains {n_params:.2f}{scale} parameters')


def create_random_mask(input_ids: torch.Tensor,
                       max_ratio_of_valid_token: float,
                       max_ratio_of_left_padding: float,
                       min_ratio_of_valid_token: float = 0):
    assert max_ratio_of_valid_token > 0 and max_ratio_of_valid_token <= 1.
    assert max_ratio_of_left_padding >= 0 and max_ratio_of_left_padding < 1.
    assert min_ratio_of_valid_token <= max_ratio_of_valid_token
    batch_size, sequence_length = input_ids.shape
    max_num_valid_tokens = int(sequence_length * max_ratio_of_valid_token)
    min_num_valid_tokens = max(1, int(sequence_length * min_ratio_of_valid_token))
    max_left_padding = int(sequence_length * max_ratio_of_left_padding)
    assert max_num_valid_tokens + max_left_padding <= sequence_length
    assert max_num_valid_tokens > 0 # Corrected: was max_ratio_of_valid_token
    masks = torch.ones_like(input_ids, dtype=torch.int64)
    for i in range(batch_size):
        num_left_padding = np.random.randint(low=0, high=max_left_padding + 1, dtype=np.int64)
        num_valid = np.random.randint(low=min_num_valid_tokens, high=max_num_valid_tokens + 1, dtype=np.int64)
        masks[i, :num_left_padding] = 0
        masks[i, num_left_padding + num_valid:] = 0
    return masks

def compute_position_id_with_mask(mask):
    return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0)


def normalize_pp_vpp_params(params, num_hidden_layers, layer_name='layers'):
    def normalize_model_name(name_in, pp_rank, vpp_rank, pp_size, vpp_size, num_layers_total):
        if vpp_size > 1:
            layers_per_pp_stage = num_layers_total // pp_size
            layers_per_vpp_chunk = layers_per_pp_stage // vpp_size
            # Corrected offset calculation logic
            # Layer index within a PP stage starts from 0 for each VPP segment.
            # Global layer index = (pp_rank * layers_per_pp_stage) + (vpp_rank * layers_per_vpp_chunk * pp_size) + local_layer_index_in_vpp_chunk
            # This is complex. The original simpler offset might be based on how Megatron names layers internally.
            # Let's stick to the original offset logic if it was working for Megatron.
            # The key is that 'name_in' contains a local layer index like 'layers.0.xxx'
            # and this '0' needs to be offset globally.

            # Simplified: local layer index + offset due to PP rank + offset due to VPP rank within its global position
            # This assumes layers are numbered 0 to N-1 globally.
            # And each VPP chunk within a PP stage has a contiguous block of layers.
            # Example: 2 PP, 2 VPP. 32 layers.
            # PP0: VPP0 (layers 0-7), VPP1 (layers 8-15)
            # PP1: VPP0 (layers 16-23), VPP1 (layers 24-31)
            # This means layer_offset = pp_rank * (num_layers_total // pp_size) + vpp_rank * (num_layers_total // (pp_size * vpp_size))
            # This is still not quite right. The original was:
            # pp_offset = layers_per_vpp * pp_rank  (where layers_per_vpp = (num_layers // pp_size) // vpp_size)
            # vpp_offset = (layers_per_vpp * pp_size) * vpp_rank
            # layer_offset = pp_offset + vpp_offset
            # This implies VPP chunks are interleaved globally, which is unusual for standard PP+VPP.
            # Let's assume standard contiguous layer blocks for PP, and then VPP divides those blocks.
            
            layers_per_pp_stage = num_layers_total // pp_size
            base_layer_for_pp_stage = pp_rank * layers_per_pp_stage
            
            layers_per_vpp_chunk_in_stage = layers_per_pp_stage // vpp_size
            layer_offset_within_pp_stage_due_to_vpp = vpp_rank * layers_per_vpp_chunk_in_stage
            
            layer_offset = base_layer_for_pp_stage + layer_offset_within_pp_stage_due_to_vpp

        else: # Only PP
            layers_per_pp_stage = num_layers_total // pp_size
            layer_offset = pp_rank * layers_per_pp_stage

        if layer_name in name_in:
            split_name = name_in.split('.')
            layer_num_idx = -1
            for i_split, item_name in enumerate(split_name):
                if item_name == layer_name:
                    layer_num_idx = i_split + 1
                    break
            if layer_num_idx != -1 and layer_num_idx < len(split_name) and split_name[layer_num_idx].isdigit():
                local_layer_idx = int(split_name[layer_num_idx])
                global_layer_idx = local_layer_idx + layer_offset # Add the calculated offset
                split_name[layer_num_idx] = str(global_layer_idx)
                return '.'.join(split_name)
        return name_in

    pp_size_val = len(params)
    normalized_name_to_param = {}
    for pp_rank_val in range(pp_size_val):
        vpp_size_val = len(params[pp_rank_val])
        for vpp_rank_val in range(vpp_size_val):
            for name_param, param_val in params[pp_rank_val][vpp_rank_val].items():
                normalized_name = normalize_model_name(name_param, pp_rank_val, vpp_rank_val, pp_size_val, vpp_size_val, num_hidden_layers)
                normalized_name_to_param[normalized_name] = param_val
    return normalized_name_to_param


def get_parallel_model_from_config(config_hf: "PretrainedConfig", # Explicitly transformers.PretrainedConfig
                                   megatron_config_mp: "ModelParallelConfig", # Explicitly megatron.core.ModelParallelConfig
                                   pre_process=None,
                                   post_process=None,
                                   share_embeddings_and_output_weights=False,
                                   value=False):
    pid = os.getpid()
    from megatron.core import ModelParallelConfig # DELAYED IMPORT (or ensure it's safe if already imported elsewhere)
    from transformers import PretrainedConfig # DELAYED IMPORT (for type hint)

    assert isinstance(megatron_config_mp, ModelParallelConfig)
    assert isinstance(config_hf, PretrainedConfig)
    
    model_class = _get_parallel_model_architecture_from_config(config_hf, value) # config_hf is PretrainedConfig
    model = model_class(config_hf, 
                        megatron_config_mp,
                        pre_process=pre_process,
                        post_process=post_process,
                        share_embeddings_and_output_weights=share_embeddings_and_output_weights)
    return model


def _get_parallel_model_architecture_from_config(config_hf: "PretrainedConfig", value=False) -> Type[nn.Module]:
    pid = os.getpid()
    # Type hint import already handled by caller or can be added here if strictly needed for this scope
    # from transformers import PretrainedConfig 

    architectures = getattr(config_hf, "architectures", [])
    for arch in architectures:
        model_cls = ModelRegistry.load_model_cls(arch, value)
        if model_cls is not None:
            return model_cls
    raise ValueError(f"Model architectures {architectures} are not supported for now. "
                     f"Supported architectures: {ModelRegistry.get_supported_archs()}")


def load_megatron_model_weights(config_hydra, # This is DictConfig (Hydra/OmegaConf)
                                model_config_hf: "PretrainedConfig", # This is transformers.PretrainedConfig
                                parallel_model_megatron, 
                                params_dtype,
                                is_value_model=False,
                                local_cache_path='~/.cache/verl/rlhf'):
    pid = os.getpid()
    from transformers import AutoModelForCausalLM, MistralForSequenceClassification, PretrainedConfig # DELAYED IMPORT
    
    assert hasattr(model_config_hf, "architectures"), "model_config_hf (PretrainedConfig) must have 'architectures' attribute."
    architectures = getattr(model_config_hf, "architectures", [])
    local_cache_path = os.path.expanduser(local_cache_path)

    model_path_from_hydra_config = config_hydra.model.path

    if model_path_from_hydra_config.startswith("hdfs:"):
        from verl.utils.fs import copy_local_path_from_hdfs # Assuming this is safe
        local_model_path = copy_local_path_from_hdfs(src=model_path_from_hydra_config, cache_dir=local_cache_path)
    else:
        local_model_path = model_path_from_hydra_config

    if 'mistral7b-rm' in model_path_from_hydra_config:
        model_hf_loaded = MistralForSequenceClassification.from_pretrained(local_model_path)
        state_dict = model_hf_loaded.state_dict()
        if 'score.weight' in state_dict: # Check if score.weight exists before assigning
            state_dict['lm_head.weight'] = state_dict['score.weight']
        else:

            if 'model.embed_tokens.weight' in state_dict and state_dict['model.embed_tokens.weight'].shape[0] > 32000:
                 state_dict['model.embed_tokens.weight'] = state_dict['model.embed_tokens.weight'][:32000]
                 is_value_model = True
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        model_hf_loaded = AutoModelForCausalLM.from_pretrained(local_model_path)
        state_dict = model_hf_loaded.state_dict()

    from verl.models.weight_loader_registry import get_weight_loader # Assuming this is safe
    for arch in architectures:
        weight_loader = get_weight_loader(arch)
        weight_loader(state_dict=state_dict,
                      wrapped_models=parallel_model_megatron,
                      config=model_hf_loaded.config, 
                      params_dtype=params_dtype,
                      is_value_model=is_value_model)


def pad_packed_inputs(unpad_tokens: torch.Tensor, cu_seqlens, max_seqlen_in_batch, size):
    F = nn.functional
    total_nnz = unpad_tokens.shape[0]
    pad_size = (size - total_nnz % size) % size # Ensures pad_size is 0 if total_nnz is multiple of size
    if pad_size > 0:
        if unpad_tokens.ndim == 1:
            unpad_tokens = F.pad(unpad_tokens, (0, pad_size))
        elif unpad_tokens.ndim == 2:
            unpad_tokens = F.pad(unpad_tokens, (0, 0, 0, pad_size))
        else:
            raise NotImplementedError(f'Padding dim {unpad_tokens.ndim()} is not supported')
        cu_seqlens = F.pad(cu_seqlens, (0, 1), value=pad_size + cu_seqlens[-1])
        max_seqlen_in_batch = max(max_seqlen_in_batch, pad_size)
    return unpad_tokens, cu_seqlens, max_seqlen_in_batch
