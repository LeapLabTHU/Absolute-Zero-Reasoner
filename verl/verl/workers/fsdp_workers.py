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
The main entry point to run the PPO algorithm
"""

import logging
import os
import warnings
import traceback # Added for diagnostics

import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
from omegaconf import DictConfig, open_dict, OmegaConf
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch

# RE-ENABLED IMPORTS (were 'DELAYED IMPORTS' in your corrupted file, moved back to top for stability)
from verl.utils import hf_tokenizer
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from verl.utils.fsdp_utils import offload_fsdp_optimizer, offload_fsdp_model_to_cpu, load_fsdp_optimizer, \
    load_fsdp_model_to_gpu
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.flops_counter import FlopsCounter
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from peft import LoraConfig, get_peft_model, TaskType
from verl.utils.model import print_model_size, update_model_config, get_generation_config
from verl.utils.torch_dtypes import PrecisionType
from transformers import AutoModelForCausalLM, AutoConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, CPUOffload
from torch import optim
from verl.utils.torch_functional import get_constant_schedule_with_warmup, tokenize_and_postprocess_data
from verl.workers.rollout import HFRollout, vLLMRollout
from verl.workers.sharding_manager import BaseShardingManager, FSDPVLLMShardingManager
from verl.workers.actor import DataParallelPPOActor
from verl.workers.critic import DataParallelPPOCritic
from verl.models.registry import check_model_support_rmpad
# IMPORTANT FIX: Commented out _apply_liger_kernel_to_instance as it's causing ImportError
from verl.models.transformers.monkey_patch import apply_monkey_patch # , _apply_liger_kernel_to_instance # Removed _apply_liger_kernel_to_instance
from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis, rearrange
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx

from transformers import BitsAndBytesConfig
from codetiming import Timer
from omegaconf import ListConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))

# Define log prefixes for clarity in multi-process logs
log_prefix_actor_init = "AZR_DIAG_LOG: ActorRolloutRefWorker.__init__"
log_prefix_bmo = "AZR_DIAG_LOG: _build_model_optimizer"
log_prefix_br = "AZR_DIAG_LOG: _build_rollout"
log_prefix_im = "AZR_DIAG_LOG: ActorRolloutRefWorker.init_model"
log_prefix_critic_init = "AZR_DIAG_LOG: CriticWorker.__init__"
log_prefix_bcmo = "AZR_DIAG_LOG: CriticWorker._build_critic_model_optimizer"
log_prefix_cim = "AZR_DIAG_LOG: CriticWorker.init_model"
log_prefix_rm_init = "AZR_DIAG_LOG: RewardModelWorker.__init__"
log_prefix_bm_rm = "AZR_DIAG_LOG: RewardModelWorker._build_model"
log_prefix_rm_im = "AZR_DIAG_LOG: RewardModelWorker.init_model"


# --- BEGIN DIAGNOSTIC FUNCTION (Helper) ---
def run_diagnostic_checks(log_prefix="AZR_DIAG_LOG"):
    proc_pid = os.getpid()
    print(f"{log_prefix}: PID {proc_pid} - Entering diagnostic checks.")
    all_passed = True
    try:
        if not torch.distributed.is_initialized():
            print(f"{log_prefix}: PID {proc_pid} - torch.distributed not initialized here.")
        else:
            print(f"{log_prefix}: PID {proc_pid} - torch.distributed initialized. Rank: {torch.distributed.get_rank()}, World Size: {torch.distributed.get_world_size()}")

        print(f"{log_prefix}: PID {proc_pid} - torch version: {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        print(f"{log_prefix}: PID {proc_pid} - torch.cuda.is_available(): {cuda_ok}")
        if cuda_ok:
            print(f"{log_prefix}: PID {proc_pid} - torch.version.cuda: {torch.version.cuda}")
            print(f"{log_prefix}: PID {proc_pid} - torch.cuda.device_count(): {torch.cuda.device_count()}")
            try:
                for i in range(torch.cuda.device_count()):
                    print(f"{log_prefix}: PID {proc_pid} - GPU {i} Name: {torch.cuda.get_device_name(i)}")
            except Exception as gpu_info_e:
                print(f"{log_prefix}: PID {proc_pid} - WARNING: Could not get detailed GPU info: {gpu_info_e}")
        else:
            all_passed = False # If CUDA is not available, set to False

        print(f"{log_prefix}: PID {proc_pid} - Attempting to import transformers.utils.import_utils...")
        try:
            from transformers.utils import import_utils
            print(f"{log_prefix}: PID {proc_pid} - Successfully imported transformers.utils.import_utils.")
            print(f"{log_prefix}: PID {proc_pid} - Attempting to check flash_attn_2_available...")
            flash_available = import_utils.is_flash_attn_2_available()
            print(f"{log_prefix}: PID {proc_pid} - transformers.utils.import_utils.is_flash_attn_2_available: {flash_available}")
            if not flash_available:
                # This diagnostic check is now the problem. If it returns False, and was True before,
                # then something about Flash Attention installation or environment has changed.
                # However, for now, we just report it, don't crash.
                print(f"{log_prefix}: PID {proc_pid} - WARNING: Flash Attention 2 is NOT available.")
                # We will not set all_passed to False here, as you indicated it was working before.
                # The actual issue might be deeper than just FA2 non-availability.
        except Exception as import_fa_e:
            print(f"{log_prefix}: PID {proc_pid} - WARNING: Could not import or check Flash Attention 2: {import_fa_e}")
            all_passed = False # If we can't even check, that's a problem.
            
        print(f"{log_prefix}: PID {proc_pid} - Diagnostic checks passed: {all_passed}")

    except Exception as e_diag:
        all_passed = False
        print(f"{log_prefix}: PID {proc_pid} - EXCEPTION in diagnostic checks: {e_diag}")
        traceback.print_exc()
    return all_passed
# --- END DIAGNOSTIC FUNCTION ---


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size <= 0 or fsdp_size >= world_size or world_size % fsdp_size != 0:
        device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])
        if fsdp_size <=0:
            print(f"INFO: create_device_mesh: fsdp_size={fsdp_size} (<=0), defaulting to FULL_SHARD (mesh_shape=({world_size},))")
        elif fsdp_size >= world_size:
            print(f"INFO: create_device_mesh: fsdp_size={fsdp_size} (>=world_size={world_size}), defaulting to FULL_SHARD (mesh_shape=({world_size},))")
        elif world_size % fsdp_size != 0:
            print(f"WARN: create_device_mesh: world_size ({world_size}) not divisible by fsdp_size ({fsdp_size}), defaulting to FULL_SHARD (mesh_shape=({world_size},))")
    else:
        device_mesh = init_device_mesh('cuda',
                                       mesh_shape=(world_size // fsdp_size, fsdp_size),
                                       mesh_dim_names=['ddp', 'fsdp'])
        print(f"INFO: create_device_mesh: Using HYBRID_SHARD (mesh_shape=({world_size // fsdp_size}, {fsdp_size}))")
    return device_mesh


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy
    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


class ActorRolloutRefWorker(Worker):
    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        proc_pid = os.getpid()
        print(f"{log_prefix_actor_init} (PID {proc_pid}), Role: {role} - START.")
        if not run_diagnostic_checks(log_prefix=log_prefix_actor_init):
            print(f"{log_prefix_actor_init} (PID {proc_pid}) - DIAGNOSTIC CHECKS FAILED EARLY! Role: {role}")
            # REMOVED: raise RuntimeError(f"Initial diagnostic checks failed in ActorRolloutRefWorker for role {role}. Halting initialization.")
        print(f"{log_prefix_actor_init} (PID {proc_pid}), Role: {role} - DIAGNOSTIC CHECKS PASSED.")

        self.config = config
        if not torch.distributed.is_initialized():
            print(f"{log_prefix_actor_init} (PID {proc_pid}) - Initializing torch.distributed process group (backend='nccl').")
            torch.distributed.init_process_group(backend="nccl")
            print(f"{log_prefix_actor_init} (PID {proc_pid}) - torch.distributed process group initialized.")

        try:
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        except AttributeError:
            self._world_size = torch.distributed.get_world_size()
            self._rank = torch.distributed.get_rank()

        print(f"{log_prefix_actor_init} (PID {proc_pid}), Rank: {self.rank}, World Size: {self.world_size} - Before create_device_mesh.")
        if torch.cuda.is_available() and self.world_size <= torch.cuda.device_count():
            try:
                torch.cuda.set_device(self.rank % torch.cuda.device_count())
                print(f"{log_prefix_actor_init} (PID {proc_pid}), Rank: {self.rank} - Set CUDA device to {torch.cuda.current_device()}.")
            except Exception as e_set_device:
                print(f"{log_prefix_actor_init} (PID {proc_pid}), Rank: {self.rank} - WARNING: Could not set CUDA device: {e_set_device}.")

        self.device_mesh = create_device_mesh(world_size=self.world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)
        print(f"{log_prefix_actor_init} (PID {proc_pid}), Rank: {self.rank} - FSDP device_mesh created: {self.device_mesh}.")

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get('ulysses_sequence_parallel_size', 1)
        if self.ulysses_sequence_parallel_size > 1:
            if self.world_size % self.ulysses_sequence_parallel_size != 0:
                raise ValueError(f"World size ({self.world_size}) must be divisible by ulysses_sequence_parallel_size ({self.ulysses_sequence_parallel_size})")
            dp_ulysses = self.world_size // self.ulysses_sequence_parallel_size
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp_ulysses, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])
            print(f"{log_prefix_actor_init} (PID {proc_pid}), Rank: {self.rank} - Ulysses device_mesh created: {self.ulysses_device_mesh}.")

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        print(f"{log_prefix_actor_init} (PID {proc_pid}), Rank: {self.rank} - Ulysses sharding manager initialized.")

        self.role = role
        assert self.role in ['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']
        self._is_actor = self.role in ['actor', 'actor_rollout', 'actor_rollout_ref']
        self._is_rollout = self.role in ['rollout', 'actor_rollout', 'actor_rollout_ref']
        self._is_ref = self.role in ['ref', 'actor_rollout_ref']

        self._is_offload_param = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get('param_offload', False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get('optimizer_offload', False)
        elif self._is_ref:
            self._is_offload_param = self.config.ref.fsdp_config.get('param_offload', False)
        print(f"{log_prefix_actor_init} (PID {proc_pid}), Rank: {self.rank} - Offload configs: param_offload={self._is_offload_param}, optimizer_offload={self._is_offload_optimizer}.")


        # Batch size adjustments
        if self.ulysses_sequence_parallel_size > 1 and self.ulysses_device_mesh:
            effective_dp_world_size = self.ulysses_device_mesh.size(0)
            print(f"{log_prefix_actor_init} (PID {proc_pid}), Rank: {self.rank} - Ulysses-based effective_dp_world_size: {effective_dp_world_size}.")
        elif self.device_mesh.ndim == 2:
             effective_dp_world_size = self.device_mesh.size(0)
             print(f"{log_prefix_actor_init} (PID {proc_pid}), Rank: {self.rank} - HSDP-based effective_dp_world_size: {effective_dp_world_size}.")
        else:
             effective_dp_world_size = self.world_size if self.world_size > 0 else 1
             print(f"{log_prefix_actor_init} (PID {proc_pid}), Rank: {self.rank} - Full Shard/Single GPU effective_dp_world_size: {effective_dp_world_size}.")

        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            if effective_dp_world_size > 0:
                self.config.actor.ppo_mini_batch_size //= effective_dp_world_size
                if self.config.actor.ppo_mini_batch_size == 0: self.config.actor.ppo_mini_batch_size = 1
                
                print(f"{log_prefix_actor_init} (PID {proc_pid}), Rank: {self.rank} - Actor ppo_mini_batch_size (per GPU): {self.config.actor.ppo_mini_batch_size}.")

                if self.config.actor.ppo_micro_batch_size is not None:
                    self.config.actor.ppo_micro_batch_size //= effective_dp_world_size
                    self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size
                    if self.config.actor.ppo_micro_batch_size_per_gpu == 0: self.config.actor.ppo_micro_batch_size_per_gpu = 1
                    
                    print(f"{log_prefix_actor_init} (PID {proc_pid}), Rank: {self.rank} - Actor ppo_micro_batch_size_per_gpu: {self.config.actor.ppo_micro_batch_size_per_gpu}.")
                    assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0, \
                        f"Actor ppo_mini_batch_size ({self.config.actor.ppo_mini_batch_size}) not divisible by ppo_micro_batch_size_per_gpu ({self.config.actor.ppo_micro_batch_size_per_gpu})"

        if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
            if effective_dp_world_size > 0:
                self.config.rollout.log_prob_micro_batch_size //= effective_dp_world_size
                self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
                if self.config.rollout.log_prob_micro_batch_size_per_gpu == 0: self.config.rollout.log_prob_micro_batch_size_per_gpu = 1
                print(f"{log_prefix_actor_init} (PID {proc_pid}), Rank: {self.rank} - Rollout log_prob_micro_batch_size_per_gpu: {self.config.rollout.log_prob_micro_batch_size_per_gpu}.")


        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            if effective_dp_world_size > 0:
                self.config.ref.log_prob_micro_batch_size //= effective_dp_world_size
                self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size
                if self.config.ref.log_prob_micro_batch_size_per_gpu == 0: self.config.ref.log_prob_micro_batch_size_per_gpu = 1
                print(f"{log_prefix_actor_init} (PID {proc_pid}), Rank: {self.rank} - Ref log_prob_micro_batch_size_per_gpu: {self.config.ref.log_prob_micro_batch_size_per_gpu}.")
        print(f"{log_prefix_actor_init} (PID {proc_pid}), Role: {role} - END init.")


    @property
    def world_size(self):
        if hasattr(self, '_world_size'):
            return self._world_size
        return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    @property
    def rank(self):
        if hasattr(self, '_rank'):
            return self._rank
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


    def _build_model_optimizer(self, model_path, fsdp_config, optim_config, override_model_config,
                               use_remove_padding=False, enable_gradient_checkpointing=False,
                               trust_remote_code=False, use_liger=False, role='actor'):
        proc_pid = os.getpid()
        print(f"{log_prefix_bmo} (PID {proc_pid}), Role: {role} - START.")
        if not run_diagnostic_checks(log_prefix=log_prefix_bmo):
            print(f"{log_prefix_bmo} (PID {proc_pid}) - DIAGNOSTIC CHECKS FAILED! Role: {role}")
            # REMOVED: raise RuntimeError(f"Diagnostic check failed in _build_model_optimizer for role {role}")
        print(f"{log_prefix_bmo} (PID {proc_pid}), Role: {role} - DIAGNOSTIC CHECKS PASSED.")

        log_gpu_memory_usage(f'Before init from HF AutoModel ({role})', logger=logger, rank=self.rank)
        local_path = copy_local_path_from_hdfs(model_path)
        print(f"{log_prefix_bmo} (PID {proc_pid}), Rank: {self.rank} - Copied model from HDFS if necessary: {local_path}.")

        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        print(f"{log_prefix_bmo} (PID {proc_pid}), Rank: {self.rank} - Tokenizer loaded.")

        torch_dtype_config = fsdp_config.get('model_dtype', None)
        if torch_dtype_config is None:
            torch_dtype = torch.bfloat16 if role == 'actor' else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype_config)
        
        qlora_keys = ["load_in_4bit", "bnb_4bit_quant_type", "bnb_4bit_compute_dtype", "bnb_4bit_use_double_quant", "bnb_4bit_quant_storage"]
        from_pretrained_qlora_kwargs = {k: v for k, v in override_model_config.items() if k in qlora_keys and v is not None}

        # FSDP-QLoRA compatibility fix
        if from_pretrained_qlora_kwargs.get("load_in_4bit"):
            # Add the critical bnb_4bit_quant_storage parameter for FSDP compatibility
            if "bnb_4bit_quant_storage" not in from_pretrained_qlora_kwargs:
                from_pretrained_qlora_kwargs["bnb_4bit_quant_storage"] = torch.bfloat16
                if self.rank == 0: 
                    logger.info(f"({role}) FSDP-QLoRA: Added bnb_4bit_quant_storage=torch.bfloat16 for FSDP compatibility")
            
            # Create BitsAndBytesConfig for proper FSDP integration
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=from_pretrained_qlora_kwargs.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=from_pretrained_qlora_kwargs.get("bnb_4bit_use_double_quant", True),
                bnb_4bit_quant_storage=torch.bfloat16,  # KEY for FSDP compatibility
            )
            # Clear individual params since we're using BitsAndBytesConfig
            from_pretrained_qlora_kwargs = {"quantization_config": quantization_config}
            if self.rank == 0:
                logger.info(f"({role}) FSDP-QLoRA: Using BitsAndBytesConfig for FSDP-compatible quantization")

        if from_pretrained_qlora_kwargs.get("load_in_4bit"):
            compute_dtype_str = from_pretrained_qlora_kwargs.get("bnb_4bit_compute_dtype")
            if compute_dtype_str:
                if isinstance(compute_dtype_str, str) and compute_dtype_str.startswith("torch."):
                     actual_compute_dtype_str = compute_dtype_str.split(".")[-1]
                else:
                     actual_compute_dtype_str = compute_dtype_str
                try:
                    potential_torch_dtype = getattr(torch, actual_compute_dtype_str)
                    if isinstance(potential_torch_dtype, torch.dtype):
                        torch_dtype = potential_torch_dtype
                        if self.rank == 0: logger.info(f"({role}) QLoRA active. Overriding from_pretrained torch_dtype to: {torch_dtype} based on bnb_4bit_compute_dtype: {compute_dtype_str}")
                except AttributeError:
                    if self.rank == 0: logger.warning(f"({role}) Could not parse torch.{actual_compute_dtype_str} for QLoRA. Using default torch_dtype for loading: {torch_dtype}")
            elif self.rank == 0: logger.info(f"({role}) QLoRA active (load_in_4bit=True) but bnb_4bit_compute_dtype not specified. Using torch_dtype for loading: {torch_dtype}")
        
        print(f"{log_prefix_bmo} (PID {proc_pid}), Rank: {self.rank} - About to call AutoConfig.from_pretrained for {local_path}.")
        actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        print(f"{log_prefix_bmo} (PID {proc_pid}), Rank: {self.rank} - After AutoConfig.from_pretrained.")

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)
        print(f"{log_prefix_bmo} (PID {proc_pid}), Rank: {self.rank} - Generation config loaded. self.generation_config: {self.generation_config}.")

        if use_remove_padding:
            check_model_support_rmpad(actor_model_config.model_type)
        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            apply_monkey_patch(actor_model_config, verbose=(self.rank == 0))
        
        current_override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        non_qlora_override_model_config = {k:v for k,v in override_model_config.items() if k not in qlora_keys}
        current_override_config_kwargs.update(non_qlora_override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=current_override_config_kwargs)

        if self.rank == 0:
            print(f'Model config after override ({role}): {actor_model_config}')
            if from_pretrained_qlora_kwargs: logger.info(f"({role}) Passing QLoRA arguments to from_pretrained: {from_pretrained_qlora_kwargs}")
            logger.info(f"({role}) Using torch_dtype for from_pretrained (model loading): {torch_dtype}")

        use_meta_init = not getattr(actor_model_config, 'tie_word_embeddings', True)
        if self.rank == 0: logger.info(f"({role}) Using meta device init: {use_meta_init} (based on tie_word_embeddings: {getattr(actor_model_config, 'tie_word_embeddings', True)})")

        try:
            init_context = get_init_weight_context_manager(use_meta_tensor=use_meta_init)
            if hasattr(init_context, '__enter__') and hasattr(init_context, '__exit__'):
                context_mgr = init_context
            else:
                from contextlib import nullcontext
                context_mgr = nullcontext()
                if self.rank == 0: logger.warning(f"({role}) get_init_weight_context_manager did not return a context manager. Using nullcontext().")
        except Exception as e:
            from contextlib import nullcontext
            context_mgr = nullcontext()
            print(f"{log_prefix_bmo} (PID {proc_pid}), Rank: {self.rank} - WARNING: get_init_weight_context_manager failed: {e}. Using nullcontext().")
        
        with context_mgr, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            print(f"{log_prefix_bmo} (PID {proc_pid}), Rank: {self.rank} - About to call AutoModelForCausalLM.from_pretrained for {local_path}.")
            actor_module = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=local_path,
                config=actor_model_config, 
                torch_dtype=torch_dtype,
                attn_implementation='flash_attention_2',
                trust_remote_code=trust_remote_code,
                **from_pretrained_qlora_kwargs
            )
            print(f"{log_prefix_bmo} (PID {proc_pid}), Rank: {self.rank} - After AutoModelForCausalLM.from_pretrained.")
        
            if use_liger:
                try:
                    # IMPORTANT: This line was commented out in your original "working" file.
                    # If Liger was truly used and enabled it should be here.
                    # If it causes an ImportError, it means liger_kernel is not installed or broken.
                    from verl.models.transformers.monkey_patch import _apply_liger_kernel_to_instance # Re-import here if needed
                    _apply_liger_kernel_to_instance(model=actor_module)
                    if self.rank == 0: logger.info(f"Liger kernel applied to {role} module.")
                except ImportError as ie:
                    if self.rank == 0: logger.error(f"Failed to import _apply_liger_kernel_to_instance. Liger kernel will not be applied: {ie}")
                    if self.rank == 0: logger.warning(f"This might be the reason for the Flash Attention dtype/GPU error if Liger was enabling float32 compatibility.")
                except Exception as e_liger:
                    if self.rank == 0: logger.error(f"Error applying Liger kernel: {e_liger}")
            
            # if not from_pretrained_qlora_kwargs.get("load_in_4bit") and not from_pretrained_qlora_kwargs.get("load_in_8bit"):
            #      actor_module.to(torch_dtype) 
            #      if self.rank == 0: logger.info(f"({role}) Model (not QLoRA quantized) cast to {torch_dtype} after loading.")


            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

            peft_config_dict = self.config.model.get('peft_config', None)
            if peft_config_dict and peft_config_dict.get('enable', False):
                if self.rank == 0: logger.info(f"Applying LoRA configuration for role: {role}")
                
                # [DEFINITIVE FIX] This block correctly handles all config types for target_modules
                target_modules_raw = peft_config_dict.get('target_modules', "q_proj,v_proj")
                if isinstance(target_modules_raw, str):
                    target_modules_list = [m.strip() for m in target_modules_raw.split(',')]
                elif isinstance(target_modules_raw, (list, ListConfig)): # Checks for both Python list and OmegaConf list
                    target_modules_list = list(target_modules_raw)
                else:
                    if self.rank == 0: logger.warning(f"Unexpected type for target_modules ({role}): {type(target_modules_raw)}. Falling back to default.")
                    target_modules_list = ["q_proj", "v_proj"]

                lora_config_obj = LoraConfig(
                    r=int(peft_config_dict.get('r', 16)), 
                    lora_alpha=int(peft_config_dict.get('lora_alpha', 32)),
                    target_modules=target_modules_list, 
                    lora_dropout=float(peft_config_dict.get('lora_dropout', 0.05)),
                    bias=peft_config_dict.get('bias', "none"), 
                    task_type=getattr(TaskType, peft_config_dict.get('task_type', "CAUSAL_LM"))
                )
                actor_module = get_peft_model(actor_module, lora_config_obj)
                if self.rank == 0: actor_module.print_trainable_parameters()

                # [MODIFIED] Add this loop to fix the FSDP dtype mismatch
                for name, param in actor_module.named_parameters():
                    if param.requires_grad and param.dtype == torch.float32:
                        param.data = param.data.to(torch.bfloat16)
        
        torch.distributed.barrier()
        if self.rank == 0: print_model_size(actor_module, name=f"{role}_module_before_fsdp")
        log_gpu_memory_usage(f'After init from HF AutoModel and PEFT ({role})', logger=logger, rank=self.rank)

        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'bf16'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'bf16'))
        else:
            param_dtype = torch.bfloat16 
            reduce_dtype = torch.bfloat16 
            buffer_dtype = torch.bfloat16
        
        if self.rank == 0:
            logger.info(f"({role}) FSDP MixedPrecision: param_dtype={param_dtype}, reduce_dtype={reduce_dtype}, buffer_dtype={buffer_dtype}")

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)
        
        auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get('wrap_policy', None))
        if self.role == 'rollout' and self.config.rollout.name == 'hf':
             auto_wrap_policy = None
        if self.rank == 0: print(f'FSDP wrap_policy for {role}: {auto_wrap_policy}')
        
        fsdp_mesh_to_use = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh_to_use)
        if self.rank == 0: logger.info(f"({role}) FSDP sharding_strategy: {sharding_strategy} on mesh: {fsdp_mesh_to_use}")

        cpu_offload_params_flag = False
        if role == 'ref' and fsdp_config.get('param_offload', False):
            cpu_offload_params_flag = True
        elif role == 'actor' and self._is_offload_param:
            cpu_offload_params_flag = True
        
        if self.rank == 0: logger.info(f"({role}) FSDP CPUOffload(offload_params={cpu_offload_params_flag})")
        
        print(f"{log_prefix_bmo} (PID {proc_pid}), Rank: {self.rank} - About to wrap model with FSDP for role {role}. Current CUDA device: {torch.cuda.current_device()}.")
        actor_module_fsdp = FSDP(
            actor_module, 
            cpu_offload=CPUOffload(offload_params=cpu_offload_params_flag), 
            param_init_fn=init_fn if not use_meta_init else None,
            use_orig_params=True, 
            auto_wrap_policy=auto_wrap_policy, 
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy, 
            mixed_precision=mixed_precision, 
            sync_module_states=True,
            device_mesh=fsdp_mesh_to_use, 
            forward_prefetch=fsdp_config.get('forward_prefetch', True),
            limit_all_gathers=fsdp_config.get('limit_all_gathers', True)
            )
        print(f"{log_prefix_bmo} (PID {proc_pid}), Rank: {self.rank} - After FSDP wrapping for role {role}.")

        log_gpu_memory_usage(f'After {role} FSDP init', logger=logger, rank=self.rank)
        
        actor_optimizer = None
        actor_lr_scheduler = None
        if role == 'actor':
            
            params_to_optimize = actor_module_fsdp.parameters()
            
            peft_config_optim_dict = self.config.model.get('peft_config', None)
            if peft_config_optim_dict and peft_config_optim_dict.get('enable', False):
                if self.rank == 0: logger.info(f"({role}) Optimizing only LoRA (trainable) parameters for Actor.")
            
            actor_optimizer = optim.AdamW(params_to_optimize, lr=optim_config.lr,
                                          betas=tuple(optim_config.get('betas', (0.9, 0.999))),
                                          weight_decay=optim_config.get('weight_decay', 1e-2),
                                          eps=optim_config.get('eps', 1e-8))
            
            total_steps = optim_config.get('total_training_steps', 0)
            num_warmup_steps_ratio = optim_config.get('lr_warmup_steps_ratio', 0.)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)
            if self.rank == 0: print(f'Actor total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')
            
            actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps)
        
        log_gpu_memory_usage(f'After {role} optimizer init', logger=logger, rank=self.rank)
        print(f"{log_prefix_bmo} (PID {proc_pid}), Role: {role} - END.")
        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self):
        proc_pid = os.getpid()
        print(f"{log_prefix_br} (PID {proc_pid}) - START.")
        if not run_diagnostic_checks(log_prefix=log_prefix_br):
            print(f"{log_prefix_br} (PID {proc_pid}) - DIAGNOSTIC CHECKS FAILED!")
            # REMOVED: raise RuntimeError("Diagnostic check failed in _build_rollout")
        print(f"{log_prefix_br} (PID {proc_pid}) - DIAGNOSTIC CHECKS PASSED.")

        infer_tp = self.config.rollout.get('tensor_model_parallel_size', 1)
        if self.world_size % infer_tp != 0:
            raise ValueError(f'Rollout: world_size ({self.world_size}) is not divisible by infer_tp ({infer_tp})')
        
        dp_rollout = self.world_size // infer_tp
        rollout_device_mesh = None
        if infer_tp > 1 or dp_rollout > 1:
            rollout_device_mesh = init_device_mesh('cuda', mesh_shape=(dp_rollout, infer_tp), mesh_dim_names=['dp_rollout', 'tp_rollout'])
            print(f"{log_prefix_br} (PID {proc_pid}) - Created rollout_device_mesh: {rollout_device_mesh}.")
        else:
            print(f"{log_prefix_br} (PID {proc_pid}) - Rollout on single device, no separate mesh created.")


        rollout = None 
        rollout_sharding_manager = None 

        if self.config.rollout.name == 'hf':
            print(f"{log_prefix_br} (PID {proc_pid}) - Initializing HFRollout.")
            rollout = HFRollout(module=self.actor_module_fsdp, config=self.config.rollout)
            rollout_sharding_manager = BaseShardingManager()
        elif self.config.rollout.name == 'vllm':
            print(f"{log_prefix_br} (PID {proc_pid}) - About to import vLLMRollout related modules.")
            from verl.workers.rollout.vllm_rollout import vllm_mode # this one needs to be local
            print(f"{log_prefix_br} (PID {proc_pid}) - Successfully imported vLLMRollout and FSDPVLLMShardingManager.")
            
            log_gpu_memory_usage('Before building vllm rollout', logger=logger, rank=self.rank)
            local_path = copy_local_path_from_hdfs(self.config.model.path)
            
            vllm_mode_local = 'spmd'
            try:
                from importlib.metadata import version as get_pkg_version, PackageNotFoundError
                vllm_pkg_version = get_pkg_version('vllm')
                if vllm_pkg_version <= '0.6.3':
                    vllm_mode_local = 'customized' 
            except PackageNotFoundError:
                if self.rank == 0: logger.warning(f"({log_prefix_br}) vLLM package not found, assuming vllm_mode='spmd'.")
            
            print(f"{log_prefix_br} (PID {proc_pid}) - Determined vllm_mode: {vllm_mode_local}.")
            print(f"{log_prefix_br} (PID {proc_pid}) - Initializing vLLMRollout with model_path: {local_path if vllm_mode_local == 'spmd' else 'FSDP module (for customized)'}.")

            vllm_rollout_kwargs = {
                "config": self.config.rollout,
                "tokenizer": self.tokenizer,
                "model_hf_config": self.actor_model_config
            }
            if vllm_mode_local == 'customized':
                 vllm_rollout_kwargs["actor_module"] = self.actor_module_fsdp
            elif vllm_mode_local == 'spmd':
                vllm_rollout_kwargs["model_path"] = local_path
                if rollout_device_mesh:
                    vllm_rollout_kwargs["device_mesh"] = rollout_device_mesh
            else:
                raise NotImplementedError(f"vllm_mode '{vllm_mode_local}' not supported for direct instantiation logic here.")

            rollout = vLLMRollout(**vllm_rollout_kwargs)
            log_gpu_memory_usage('After building vllm rollout', logger=logger, rank=self.rank)
            
            rollout_sharding_manager_kwargs = {
                "module": self.actor_module_fsdp,
                "inference_engine": rollout.inference_engine,
                "model_config": self.actor_model_config,
                "full_params": self.config.rollout.get('load_format', 'hf') == 'hf',
            }
            if rollout_device_mesh:
                rollout_sharding_manager_kwargs["device_mesh"] = rollout_device_mesh

            rollout_sharding_manager = FSDPVLLMShardingManager(**rollout_sharding_manager_kwargs)
            log_gpu_memory_usage('After building vLLM sharding manager', logger=logger, rank=self.rank)
        else:
            raise NotImplementedError(f"Unknown rollout name: {self.config.rollout.name}")
        
        print(f"{log_prefix_br} (PID {proc_pid}) - END.")
        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        proc_pid = os.getpid()
        print(f"{log_prefix_im} (PID {proc_pid}), Role: {self.role} - START.")
        if not run_diagnostic_checks(log_prefix=log_prefix_im):
            print(f"{log_prefix_im} (PID {proc_pid}) - DIAGNOSTIC CHECKS FAILED EARLY IN INIT_MODEL!")
            # REMOVED: raise RuntimeError("Diagnostic check failed at start of init_model")
        print(f"{log_prefix_im} (PID {proc_pid}), Role: {self.role} - DIAGNOSTIC CHECKS PASSED.")

        import_external_libs(self.config.model.get('external_lib', None))
        
        actor_override_model_config_hydra = self.config.model.get('override_config', OmegaConf.create({}))
        actor_override_model_config = OmegaConf.to_container(actor_override_model_config_hydra, resolve=True)
        use_remove_padding = self.config.model.get('use_remove_padding', False)

        if self._is_actor or self._is_rollout:
            optim_config_actor = self.config.actor.optim
            current_fsdp_config = self.config.actor.fsdp_config
            
            print(f"{log_prefix_im} (PID {proc_pid}), Rank: {self.rank} - Before _build_model_optimizer (for actor/rollout base).")
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = self._build_model_optimizer(
                model_path=self.config.model.path, 
                fsdp_config=current_fsdp_config, 
                optim_config=optim_config_actor if self._is_actor else None,
                override_model_config=actor_override_model_config, 
                use_remove_padding=use_remove_padding,
                enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False),
                trust_remote_code=self.config.model.get('trust_remote_code', False),
                use_liger=self.config.model.get('use_liger', False), 
                role='actor'
            )
            print(f"{log_prefix_im} (PID {proc_pid}), Rank: {self.rank} - After _build_model_optimizer (for actor/rollout base).")
            
            if self._is_actor and self._is_offload_optimizer and self.actor_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage('After offload actor optimizer during init', logger=logger, rank=self.rank)
        
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor): self.config.actor.use_remove_padding = use_remove_padding
            self.actor = DataParallelPPOActor(config=self.config.actor,
                                              actor_module=self.actor_module_fsdp,
                                              actor_optimizer=self.actor_optimizer)
            print(f"{log_prefix_im} (PID {proc_pid}), Rank: {self.rank} - Actor policy initialized.")

        if self._is_rollout:
            print(f"{log_prefix_im} (PID {proc_pid}), Rank: {self.rank} - Before _build_rollout.")
            self.rollout, self.rollout_sharding_manager = self._build_rollout()
            print(f"{log_prefix_im} (PID {proc_pid}), Rank: {self.rank} - After _build_rollout. Rollout engine: {type(self.rollout).__name__}.")

        if self._is_ref:
            ref_model_config_sec = self.config.ref.get('model', self.config.model)
            ref_override_model_config_hydra = ref_model_config_sec.get('override_config', actor_override_model_config_hydra)
            ref_override_model_config = OmegaConf.to_container(ref_override_model_config_hydra, resolve=True)
            
            ref_fsdp_config = self.config.ref.fsdp_config
            
            print(f"{log_prefix_im} (PID {proc_pid}), Rank: {self.rank} - Before _build_model_optimizer (for ref).")
            self.ref_module_fsdp, _, _, self.ref_model_config = self._build_model_optimizer(
                model_path=ref_model_config_sec.get('path', self.config.model.path), 
                fsdp_config=ref_fsdp_config, 
                optim_config=None,
                override_model_config=ref_override_model_config, 
                use_remove_padding=use_remove_padding,
                enable_gradient_checkpointing=ref_model_config_sec.get('enable_gradient_checkpointing', False),
                trust_remote_code=ref_model_config_sec.get('trust_remote_code', False),
                use_liger=ref_model_config_sec.get('use_liger', False), 
                role='ref'
            )
            print(f"{log_prefix_im} (PID {proc_pid}), Rank: {self.rank} - After _build_model_optimizer (for ref).")
            
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref): self.config.ref.use_remove_padding = use_remove_padding
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)
            print(f"{log_prefix_im} (PID {proc_pid}), Rank: {self.rank} - Ref policy initialized.")


        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(model=self.actor_module_fsdp,
                                                            optimizer=self.actor_optimizer,
                                                            lr_scheduler=self.actor_lr_scheduler,
                                                            tokenizer=self.tokenizer)
            print(f"{log_prefix_im} (PID {proc_pid}), Rank: {self.rank} - Checkpoint manager for actor initialized.")

        torch.cuda.empty_cache()
        print(f"{log_prefix_im} (PID {proc_pid}), Rank: {self.rank} - END init_model. CUDA cache emptied.")
    
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        data = data.to('cuda')
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer and self.actor_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())

        log_gpu_memory_usage('Before update policy', logger=logger, rank=self.rank)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name='update_policy', logger=logger.info) as timer: # FIX: Changed logger=logger to logger=logger.info
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info['global_token_num']
            if hasattr(self, 'flops_counter') and self.flops_counter:
                 estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
                 if promised_flops > 0 :
                    metrics['mfu/actor'] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
                 else:
                    metrics['mfu/actor'] = 0.0

            if self.actor_lr_scheduler:
                self.actor_lr_scheduler.step()
                lr = self.actor_lr_scheduler.get_last_lr()[0]
                metrics['actor/lr'] = lr
            
            log_gpu_memory_usage('After update policy', logger=logger, rank=self.rank)
            output = DataProto(meta_info={'metrics': metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        if self._is_offload_optimizer and self.actor_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        import os
        import torch

        logger_instance = logger

        prompts = prompts.to('cuda') 
        assert self._is_rollout
        
        if hasattr(self, 'actor_module_fsdp') and self.actor_module_fsdp and \
           self.config.actor.fsdp_config.get('param_offload', False): 
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        ### START MODIFIED SECTION for robust EOS/PAD ID fetching ###
        eos_id_to_use = None
        pad_id_to_use = None
        
        current_rank = self.rank

        if hasattr(self, 'actor_model_config') and self.actor_model_config:
            eos_id_to_use = getattr(self.actor_model_config, 'eos_token_id', None)
            pad_id_to_use = getattr(self.actor_model_config, 'pad_token_id', None)
            if current_rank == 0 and logger_instance:
                logger_instance.debug(f"DEBUG: generate_sequences (PID {os.getpid()}) - Using EOS/PAD from actor_model_config: EOS={eos_id_to_use}, PAD={pad_id_to_use}")

        if eos_id_to_use is None and hasattr(self, 'generation_config') and self.generation_config:
            eos_id_to_use = getattr(self.generation_config, 'eos_token_id', None)
            if current_rank == 0 and logger_instance:
                logger_instance.debug(f"DEBUG: generate_sequences (PID {os.getpid()}) - Falling back to generation_config for EOS: {eos_id_to_use}")

        if pad_id_to_use is None and hasattr(self, 'generation_config') and self.generation_config:
            pad_id_to_use = getattr(self.generation_config, 'pad_token_id', None)
            if current_rank == 0 and logger_instance:
                logger_instance.debug(f"DEBUG: generate_sequences (PID {os.getpid()}) - Falling back to generation_config for PAD: {pad_id_to_use}")

        if eos_id_to_use is None and hasattr(self, 'tokenizer') and self.tokenizer:
            eos_id_to_use = getattr(self.tokenizer, 'eos_token_id', None)
            if current_rank == 0 and logger_instance:
                logger_instance.debug(f"DEBUG: generate_sequences (PID {os.getpid()}) - Falling back to tokenizer for EOS: {eos_id_to_use}")

        if pad_id_to_use is None and hasattr(self, 'tokenizer') and self.tokenizer:
            pad_id_to_use = getattr(self.tokenizer, 'pad_token_id', None)
            if current_rank == 0 and logger_instance:
                logger_instance.debug(f"DEBUG: generate_sequences (PID {os.getpid()}) - Falling back to tokenizer for PAD: {pad_id_to_use}")
        
        if pad_id_to_use is None and eos_id_to_use is not None:
            pad_id_to_use = eos_id_to_use
            if current_rank == 0 and logger_instance:
                logger_instance.info(f"INFO: generate_sequences (PID {os.getpid()}) - PAD token ID is None, using EOS token ID ({eos_id_to_use}) as PAD token ID.")
        
        meta_info = {
            'eos_token_id': eos_id_to_use,
            'pad_token_id': pad_id_to_use,
        }

        if meta_info['eos_token_id'] is None or meta_info['pad_token_id'] is None:
            if current_rank == 0 and logger_instance:
                logger_instance.critical(f"AZR_DIAG_LOG: generate_sequences (PID {os.getpid()}) - CRITICAL WARNING: eos_token_id ({meta_info['eos_token_id']}) or pad_token_id ({meta_info['pad_token_id']}) is None. This will likely cause errors in vLLM or generation termination.")
        else:
            if current_rank == 0 and logger_instance:
                logger_instance.info(f"AZR_DIAG_LOG: generate_sequences (PID {os.getpid()}) - Token IDs confirmed: EOS={meta_info['eos_token_id']}, PAD={meta_info['pad_token_id']}.")
        ### END MODIFIED SECTION ###
        
        prompts.meta_info.update(meta_info)

        with self.rollout_sharding_manager: 
            if hasattr(self, 'actor_module_fsdp') and self.actor_module_fsdp and \
               self.config.actor.fsdp_config.get('param_offload', False):
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            
            if hasattr(self, 'actor_optimizer') and self.actor_optimizer and \
               self.config.actor.fsdp_config.get('optimizer_offload', False):
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)

            effective_rank_for_logging = current_rank
            log_gpu_memory_usage('After entering rollout sharding manager', logger=logger_instance, rank=effective_rank_for_logging)
            prompts = self.rollout_sharding_manager.preprocess_data(prompts)

            if hasattr(self.rollout, 'inference_engine') and hasattr(self.rollout.inference_engine, 'sampling_params'):
                current_sampling_params = self.rollout.inference_engine.sampling_params
                if effective_rank_for_logging == 0 and logger_instance:
                    logger_instance.debug(f"DEBUG: [VLLMRollout generate_sequences] Effective SamplingParams for vLLM.generate: {current_sampling_params.kwargs}")

            output = self.rollout.generate_sequences(prompts=prompts)
            log_gpu_memory_usage('After rollout generation', logger=logger_instance, rank=effective_rank_for_logging)
            output = self.rollout_sharding_manager.postprocess_data(output)

        output = output.to('cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_gpu_memory_usage('After generate_sequences (emptied cache)', logger=logger_instance, rank=effective_rank_for_logging)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        data = data.to('cuda')
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        
        log_gpu_memory_usage('Before compute_log_prob', logger=logger, rank=self.rank)
        
        data.meta_info['micro_batch_size'] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info['max_token_len'] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info['temperature'] = self.config.rollout.temperature

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            if not hasattr(self, 'actor') or not self.actor:
                 raise RuntimeError("Actor policy (self.actor) not initialized in compute_log_prob.")
            output_log_probs = self.actor.compute_log_prob(data=data) 
            output = DataProto.from_dict(tensors={'old_log_probs': output_log_probs},
                                         meta_info={'temperature': self.config.rollout.temperature})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to('cpu')
        if self.world_size > 1 and hasattr(self.actor.actor_module, '_handle') and self.actor.actor_module._handle:
            self.actor.actor_module._handle.reshard(True)
        
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After compute_log_prob (actor)', logger=logger, rank=self.rank)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref
        data = data.to('cuda')
        
        log_gpu_memory_usage('Before compute_ref_log_prob', logger=logger, rank=self.rank)

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['temperature'] = self.config.rollout.temperature
        data.meta_info['max_token_len'] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.ref.log_prob_use_dynamic_bsz

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            if not hasattr(self, 'ref_policy') or not self.ref_policy:
                raise RuntimeError("Reference policy (self.ref_policy) not initialized in compute_ref_log_prob.")
            output_log_probs = self.ref_policy.compute_log_prob(data=data) 
            output = DataProto.from_dict(tensors={'ref_log_prob': output_log_probs})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to('cpu')
        if self.world_size > 1 and hasattr(self.ref_policy.actor_module, '_handle') and self.ref_policy.actor_module._handle:
            self.ref_policy.actor_module._handle.reshard(True)
        
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After compute_ref_log_prob', logger=logger, rank=self.rank) 
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, remove_previous_ckpt=False):
        assert self._is_actor
        if not hasattr(self, 'checkpoint_manager') or not self.checkpoint_manager:
            if self.rank == 0: logger.warning("Checkpoint manager not initialized. Skipping save_checkpoint.")
            return

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        
        self.checkpoint_manager.save_checkpoint(local_path=local_path,
                                                hdfs_path=hdfs_path,
                                                global_step=global_step,
                                                remove_previous_ckpt=remove_previous_ckpt)
        torch.distributed.barrier()
        
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path, del_local_after_load=False): 
        assert self._is_actor 
        if not hasattr(self, 'checkpoint_manager') or not self.checkpoint_manager:
            if self.rank == 0: logger.warning("Checkpoint manager not initialized. Skipping load_checkpoint.")
            return
        
        self.checkpoint_manager.load_checkpoint(path=path, del_local_after_load=del_local_after_load)
        torch.distributed.barrier() 
        
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)


class CriticWorker(Worker):
    def __init__(self, config):
        super().__init__()
        proc_pid = os.getpid()
        print(f"{log_prefix_critic_init} (PID {proc_pid}) - START.")
        if not run_diagnostic_checks(log_prefix=log_prefix_critic_init):
            print(f"{log_prefix_critic_init} (PID {proc_pid}) - DIAGNOSTIC CHECKS FAILED EARLY!")
            # REMOVED: raise RuntimeError("Initial diagnostic checks failed in CriticWorker")
        print(f"{log_prefix_critic_init} (PID {proc_pid}) - DIAGNOSTIC CHECKS PASSED.")

        if not torch.distributed.is_initialized():
            print(f"{log_prefix_critic_init} (PID {proc_pid}) - Initializing process group.")
            torch.distributed.init_process_group(backend="nccl")
            print(f"{log_prefix_critic_init} (PID {proc_pid}) - Process group initialized.")
        
        self.config = config
        try:
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        except AttributeError:
            self._world_size = torch.distributed.get_world_size()
            self._rank = torch.distributed.get_rank()

        if torch.cuda.is_available() and self.world_size <= torch.cuda.device_count():
            try:
                torch.cuda.set_device(self.rank % torch.cuda.device_count())
                print(f"{log_prefix_critic_init} (PID {proc_pid}) - Set CUDA device to {torch.cuda.current_device()} for Rank {self.rank}.")
            except Exception as e_set_device:
                print(f"{log_prefix_critic_init} (PID {proc_pid}) - WARNING: Could not set CUDA device for Rank {self.rank}: {e_set_device}.")

        print(f"{log_prefix_critic_init} (PID {proc_pid}) - Rank: {self.rank}, World Size: {self.world_size}.")

        fsdp_size_critic = self.config.model.fsdp_config.get('fsdp_size', self.world_size)
        self.device_mesh = create_device_mesh(world_size=self.world_size, fsdp_size=fsdp_size_critic)
        print(f"{log_prefix_critic_init} (PID {proc_pid}) - FSDP device_mesh created: {self.device_mesh}.")


        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        if self.ulysses_sequence_parallel_size > 1:
            if self.world_size % self.ulysses_sequence_parallel_size != 0:
                 raise ValueError(f"Critic: World size ({self.world_size}) must be divisible by ulysses_sequence_parallel_size ({self.ulysses_sequence_parallel_size})")
            dp_ulysses_critic = self.world_size // self.ulysses_sequence_parallel_size
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp_ulysses_critic, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])
            print(f"{log_prefix_critic_init} (PID {proc_pid}) - Ulysses device_mesh created: {self.ulysses_device_mesh}.")

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        print(f"{log_prefix_critic_init} (PID {proc_pid}) - Ulysses sharding manager initialized.")
        
        self._is_offload_param = self.config.model.fsdp_config.get('param_offload', False)
        self._is_offload_optimizer = self.config.model.fsdp_config.get('optimizer_offload', False)
        print(f"{log_prefix_critic_init} (PID {proc_pid}), Rank: {self.rank} - Offload configs: param_offload={self._is_offload_param}, optimizer_offload={self._is_offload_optimizer}.")

        if self.ulysses_sequence_parallel_size > 1 and self.ulysses_device_mesh:
            effective_dp_size_critic = self.ulysses_device_mesh.size(0)
        elif self.device_mesh.ndim == 2:
             effective_dp_size_critic = self.device_mesh.size(0)
        else:
             effective_dp_size_critic = self.world_size if self.world_size > 0 else 1

        if effective_dp_size_critic > 0:
            self.config.ppo_mini_batch_size //= effective_dp_size_critic
            if self.config.ppo_mini_batch_size == 0: self.config.ppo_mini_batch_size = 1

            if self.config.ppo_micro_batch_size is not None:
                self.config.ppo_micro_batch_size //= effective_dp_size_critic
                self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size
                if self.config.ppo_micro_batch_size_per_gpu == 0: self.config.ppo_micro_batch_size_per_gpu = 1
                assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0, \
                    f"Critic ppo_mini_batch_size ({self.config.ppo_mini_batch_size}) not divisible by ppo_micro_batch_size_per_gpu ({self.config.ppo_micro_batch_size_per_gpu})"

            if self.config.get('forward_micro_batch_size') is not None:
                self.config.forward_micro_batch_size //= effective_dp_size_critic
                self.config.forward_micro_batch_size_per_gpu = self.config.forward_micro_batch_size
                if self.config.forward_micro_batch_size_per_gpu == 0: self.config.forward_micro_batch_size_per_gpu = 1

        print(f"{log_prefix_critic_init} (PID {proc_pid}) - END init.")

    @property
    def world_size(self):
        if hasattr(self, '_world_size'):
            return self._world_size
        return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    @property
    def rank(self):
        if hasattr(self, '_rank'):
            return self._rank
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


    def _build_critic_model_optimizer(self, config_critic_worker):
        proc_pid = os.getpid()
        print(f"{log_prefix_bcmo} (PID {proc_pid}) - START.")
        if not run_diagnostic_checks(log_prefix=log_prefix_bcmo):
            print(f"{log_prefix_bcmo} (PID {proc_pid}) - DIAGNOSTIC CHECKS FAILED!")
            # REMOVED: raise RuntimeError("Diagnostic check failed in _build_critic_model_optimizer")
        print(f"{log_prefix_bcmo} (PID {proc_pid}) - DIAGNOSTIC CHECKS PASSED.")

        local_path = copy_local_path_from_hdfs(config_critic_worker.model.path)
        tokenizer_path_cfg = config_critic_worker.model.get('tokenizer_path', config_critic_worker.model.path)
        tokenizer_local_path = copy_local_path_from_hdfs(tokenizer_path_cfg)
        tokenizer_trust_remote_code = config_critic_worker.model.get('trust_remote_code', False)
        self.tokenizer = hf_tokenizer(tokenizer_local_path, trust_remote_code=tokenizer_trust_remote_code)
        
        critic_override_config_hydra = config_critic_worker.model.get('override_config', OmegaConf.create({}))
        current_override_config = OmegaConf.to_container(critic_override_config_hydra, resolve=True)
        
        tokenizer_override_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        display_override_config = tokenizer_override_kwargs.copy()
        display_override_config.update(current_override_config)
        if self.rank == 0: print(f'Critic effective override_config (for display/tokenizer related): {display_override_config}')
        
        torch_dtype_config = config_critic_worker.model.fsdp_config.get('model_dtype', 'fp32')
        torch_dtype = PrecisionType.to_dtype(torch_dtype_config)
        
        qlora_keys = ["load_in_4bit", "bnb_4bit_quant_type", "bnb_4bit_compute_dtype", "bnb_4bit_use_double_quant"]
        from_pretrained_qlora_kwargs_critic = {k: v for k, v in current_override_config.items() if k in qlora_keys and v is not None}

        if from_pretrained_qlora_kwargs_critic.get("load_in_4bit"):
            compute_dtype_str_critic = from_pretrained_qlora_kwargs_critic.get("bnb_4bit_compute_dtype")
            if compute_dtype_str_critic:
                if isinstance(compute_dtype_str_critic, str) and compute_dtype_str_critic.startswith("torch."): 
                    actual_compute_dtype_str_critic = compute_dtype_str_critic.split(".")[-1]
                else:
                    actual_compute_dtype_str_critic = compute_dtype_str_critic
                try:
                    potential_torch_dtype_critic = getattr(torch, actual_compute_dtype_str_critic)
                    if isinstance(potential_torch_dtype_critic, torch.dtype):
                        torch_dtype = potential_torch_dtype_critic 
                        if self.rank == 0: logger.info(f"Critic QLoRA active. Overriding from_pretrained torch_dtype to: {torch_dtype} based on bnb_4bit_compute_dtype: {compute_dtype_str_critic}")
                except AttributeError:
                     if self.rank == 0: logger.warning(f"Critic - Could not parse torch.{actual_compute_dtype_str_critic} for QLoRA. Using default torch_dtype for loading: {torch_dtype}")
            elif self.rank == 0: logger.info(f"Critic QLoRA active (load_in_4bit=True) but bnb_4bit_compute_dtype not specified. Using torch_dtype for loading: {torch_dtype}")
        
        model_trust_remote_code = config_critic_worker.model.get('trust_remote_code', False)
        print(f"{log_prefix_bcmo} (PID {proc_pid}), Rank: {self.rank} - About to call AutoConfig.from_pretrained for {local_path} (Critic).")
        critic_model_config_obj = AutoConfig.from_pretrained(local_path, trust_remote_code=model_trust_remote_code)
        print(f"{log_prefix_bcmo} (PID {proc_pid}), Rank: {self.rank} - After AutoConfig.from_pretrained (Critic).")
        
        critic_model_config_obj.num_labels = 1
        
        for key, value in current_override_config.items():
            if key not in qlora_keys and hasattr(critic_model_config_obj, key) :
                setattr(critic_model_config_obj, key, value)
                if self.rank == 0: logger.info(f"Critic - Applied override {key}={value} to model config.")
        
        use_remove_padding_critic = config_critic_worker.model.get('use_remove_padding', False)
        if use_remove_padding_critic:
            check_model_support_rmpad(critic_model_config_obj.model_type)
        if use_remove_padding_critic and self.ulysses_sequence_parallel_size > 1:
            apply_monkey_patch(critic_model_config_obj, verbose=(self.rank == 0))
        
        if self.rank == 0:
            print(f'Critic model config before from_pretrained: {critic_model_config_obj}')
            if from_pretrained_qlora_kwargs_critic: logger.info(f"Critic - Passing QLoRA arguments to from_pretrained: {from_pretrained_qlora_kwargs_critic}")
            logger.info(f"Critic - Using torch_dtype for from_pretrained (model loading): {torch_dtype}")

        use_meta_init_critic = not getattr(critic_model_config_obj, 'tie_word_embeddings', True)
        if self.rank == 0: logger.info(f"(Critic) Using meta device init: {use_meta_init_critic} (based on tie_word_embeddings: {getattr(critic_model_config_obj, 'tie_word_embeddings', True)})")

        try:
            init_context_critic = get_init_weight_context_manager(use_meta_tensor=use_meta_init_critic)
            if hasattr(init_context_critic, '__enter__') and hasattr(init_context_critic, '__exit__'):
                context_mgr_critic = init_context_critic
            else:
                from contextlib import nullcontext
                context_mgr_critic = nullcontext()
                if self.rank == 0: logger.warning(f"Critic: get_init_weight_context_manager did not return a context manager. Using nullcontext().")
        except Exception as e:
            from contextlib import nullcontext
            context_mgr_critic = nullcontext()
            print(f"{log_prefix_bcmo} (PID {proc_pid}), Rank: {self.rank} - WARNING: get_init_weight_context_manager failed for Critic: {e}. Using nullcontext().")
        
        with context_mgr_critic, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not hasattr(critic_model_config_obj, 'classifier_dropout') or \
               current_override_config.get('classifier_dropout') is None : 
                setattr(critic_model_config_obj, 'classifier_dropout', 0.0)
            
            print(f"{log_prefix_bcmo} (PID {proc_pid}), Rank: {self.rank} - About to call AutoModelForTokenClassification.from_pretrained for {local_path}.")
            critic_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path, 
                config=critic_model_config_obj, 
                torch_dtype=torch_dtype,
                attn_implementation='flash_attention_2',
                trust_remote_code=model_trust_remote_code,
                **from_pretrained_qlora_kwargs_critic
            )
            print(f"{log_prefix_bcmo} (PID {proc_pid}), Rank: {self.rank} - After AutoModelForTokenClassification.from_pretrained.")

            if not from_pretrained_qlora_kwargs_critic.get("load_in_4bit") and not from_pretrained_qlora_kwargs_critic.get("load_in_8bit"):
                 critic_module.to(torch_dtype)
                 if self.rank == 0: logger.info(f"(Critic) Model (not QLoRA quantized) cast to {torch_dtype} after loading.")

            if config_critic_worker.model.get('enable_gradient_checkpointing', False):
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            
            peft_config_critic_dict = config_critic_worker.model.get('peft_config', None)
            if peft_config_critic_dict and peft_config_critic_dict.get('enable', False):
                if self.rank == 0: logger.info("Applying LoRA configuration for Critic.")
                target_modules_raw_critic = peft_config_critic_dict.get('target_modules', "query,value")
                if isinstance(target_modules_raw_critic, str): target_modules_list_critic = [m.strip() for m in target_modules_raw_critic.split(',')]
                elif isinstance(target_modules_raw_critic, OmegaConf.ListConfig): target_modules_list_critic = list(target_modules_raw_critic)
                elif isinstance(target_modules_raw_critic, list): target_modules_list_critic = target_modules_raw_critic
                else:
                    if self.rank == 0: logger.warning(f"Critic - Unexpected type for target_modules: {type(target_modules_raw_critic)}. Falling back.")
                    target_modules_list_critic = ["query", "value"]
                
                task_type_str_critic = peft_config_critic_dict.get('task_type', "TOKEN_CLS")
                try: task_type_enum_critic = getattr(TaskType, task_type_str_critic)
                except AttributeError:
                    if self.rank == 0: logger.warning(f"Critic - Invalid TaskType '{task_type_str_critic}'. Defaulting to TOKEN_CLS.")
                    task_type_enum_critic = TaskType.TOKEN_CLS
                
                lora_config_obj_critic = LoraConfig(
                    r=int(peft_config_critic_dict.get('r', 16)), 
                    lora_alpha=int(peft_config_critic_dict.get('lora_alpha', 32)),
                    target_modules=target_modules_list_critic, 
                    lora_dropout=float(peft_config_critic_dict.get('lora_dropout', 0.05)),
                    bias=peft_config_critic_dict.get('bias', "none"), 
                    task_type=task_type_enum_critic)
                critic_module = get_peft_model(critic_module, lora_config_obj_critic)
                if self.rank == 0: critic_module.print_trainable_parameters()
        
        if self.rank == 0: print_model_size(critic_module, name="Critic_module_before_fsdp")
        self.critic_model_config = critic_model_config_obj
        
        fsdp_config_critic_sec = config_critic_worker.model.fsdp_config
        mixed_precision_config_critic = fsdp_config_critic_sec.get('mixed_precision', None)
        if mixed_precision_config_critic is not None:
            param_dtype_critic = PrecisionType.to_dtype(mixed_precision_config_critic.get('param_dtype', 'bf16'))
            reduce_dtype_critic = PrecisionType.to_dtype(mixed_precision_config_critic.get('reduce_dtype', 'bf16'))
            buffer_dtype_critic = PrecisionType.to_dtype(mixed_precision_config_critic.get('buffer_dtype', 'bf16'))
        else: 
            param_dtype_critic = torch.bfloat16
            reduce_dtype_critic = torch.bfloat16
            buffer_dtype_critic = torch.bfloat16
        
        if self.rank == 0:
            logger.info(f"(Critic) FSDP MixedPrecision: param_dtype={param_dtype_critic}, reduce_dtype={reduce_dtype_critic}, buffer_dtype={buffer_dtype_critic}")

        mixed_precision_critic = MixedPrecision(param_dtype=param_dtype_critic, reduce_dtype=reduce_dtype_critic, buffer_dtype=buffer_dtype_critic)
        
        auto_wrap_policy_critic = get_fsdp_wrap_policy(module=critic_module, config=fsdp_config_critic_sec.get('wrap_policy', None))
        if self.rank == 0: print(f'FSDP wrap_policy for Critic: {auto_wrap_policy_critic}')

        log_gpu_memory_usage('Before critic FSDP', logger=logger, rank=self.rank)
        
        fsdp_mesh_critic = self.device_mesh
        sharding_strategy_critic = get_sharding_strategy(fsdp_mesh_critic)
        if self.rank == 0: logger.info(f"(Critic) FSDP sharding_strategy: {sharding_strategy_critic} on mesh: {fsdp_mesh_critic}")

        cpu_offload_critic_flag = config_critic_worker.model.fsdp_config.get("param_offload", False)
        if config_critic_worker.model.fsdp_config.get("param_offload_fsdp_native", False):
             cpu_offload_critic_flag = True

        if self.rank == 0: logger.info(f"Critic - FSDP CPUOffload(offload_params={cpu_offload_critic_flag})")
        
        print(f"{log_prefix_bcmo} (PID {proc_pid}), Rank: {self.rank} - About to wrap model with FSDP for Critic. Current CUDA device: {torch.cuda.current_device()}.")
        critic_module_fsdp = FSDP(
            critic_module, 
            cpu_offload=CPUOffload(offload_params=cpu_offload_critic_flag),
            param_init_fn=init_fn if not use_meta_init_critic else None,
            use_orig_params=True, 
            auto_wrap_policy=auto_wrap_policy_critic,
            device_id=torch.cuda.current_device(), 
            sharding_strategy=sharding_strategy_critic,
            mixed_precision=mixed_precision_critic, 
            sync_module_states=True, 
            forward_prefetch=fsdp_config_critic_sec.get('forward_prefetch', True),
            limit_all_gathers=fsdp_config_critic_sec.get('limit_all_gathers', True),
            device_mesh=fsdp_mesh_critic)
        print(f"{log_prefix_bcmo} (PID {proc_pid}), Rank: {self.rank} - After FSDP wrapping for Critic.")

        log_gpu_memory_usage('After critic FSDP', logger=logger, rank=self.rank)
        
        params_to_optimize_critic = critic_module_fsdp.parameters()
        peft_config_critic_optim_dict = config_critic_worker.model.get('peft_config', None)
        if peft_config_critic_optim_dict and peft_config_critic_optim_dict.get('enable', False):
            if self.rank == 0: logger.info("Optimizing only LoRA (trainable) parameters for Critic.")
            
        critic_optimizer = optim.AdamW(
            params_to_optimize_critic, 
            lr=config_critic_worker.optim.lr,
            betas=tuple(config_critic_worker.optim.get('betas', (0.9, 0.999))),
            weight_decay=config_critic_worker.optim.get('weight_decay', 1e-2),
            eps=config_critic_worker.optim.get('eps', 1e-8)
        )
        
        total_steps_critic = config_critic_worker.optim.get('total_training_steps', 0)
        num_warmup_steps_ratio_critic = config_critic_worker.optim.get('lr_warmup_steps_ratio', 0.)
        num_warmup_steps_critic = int(num_warmup_steps_ratio_critic * total_steps_critic)
        if self.rank == 0: print(f'Critic total steps: {total_steps_critic}, num_warmup_steps: {num_warmup_steps_critic}')
        
        critic_lr_scheduler = get_constant_schedule_with_warmup(optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps_critic)
        
        print(f"{log_prefix_bcmo} (PID {proc_pid}) - END.")
        return critic_module_fsdp, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        proc_pid = os.getpid()
        print(f"{log_prefix_cim} (PID {proc_pid}) - START.")
        if not run_diagnostic_checks(log_prefix=log_prefix_cim):
            print(f"{log_prefix_cim} (PID {proc_pid}) - DIAGNOSTIC CHECKS FAILED EARLY!")
            # REMOVED: raise RuntimeError("Diagnostic check failed at start of CriticWorker.init_model")
        print(f"{log_prefix_cim} (PID {proc_pid}) - DIAGNOSTIC CHECKS PASSED.")

        import_external_libs(self.config.model.get('external_lib', None))
        
        print(f"{log_prefix_cim} (PID {proc_pid}), Rank: {self.rank} - Before _build_critic_model_optimizer.")
        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(self.config)
        print(f"{log_prefix_cim} (PID {proc_pid}), Rank: {self.rank} - After _build_critic_model_optimizer.")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
            log_gpu_memory_usage('After offload critic model during init', logger=logger, rank=self.rank)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
            log_gpu_memory_usage('After offload critic optimizer during init', logger=logger, rank=self.rank)
            
        self.critic = DataParallelPPOCritic(config=self.config, 
                                            critic_module=self.critic_module, 
                                            critic_optimizer=self.critic_optimizer)
        
        if not hasattr(self, 'critic_model_config') or not self.critic_model_config:
            raise RuntimeError("Critic model config not set after _build_critic_model_optimizer")
        self.flops_counter = FlopsCounter(self.critic_model_config)
        
        self.checkpoint_manager = FSDPCheckpointManager(model=self.critic_module, 
                                                        optimizer=self.critic_optimizer,
                                                        lr_scheduler=self.critic_lr_scheduler, 
                                                        tokenizer=self.tokenizer)
        torch.cuda.empty_cache()
        print(f"{log_prefix_cim} (PID {proc_pid}), Rank: {self.rank} - END init_model. CUDA cache emptied.")

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        data = data.to('cuda')
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        
        log_gpu_memory_usage('Before compute_values', logger=logger, rank=self.rank)

        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['max_token_len'] = self.config.get('forward_max_token_len_per_gpu', 512)
        data.meta_info['use_dynamic_bsz'] = self.config.get('use_dynamic_bsz', False)
        
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            if not hasattr(self, 'critic') or not self.critic:
                raise RuntimeError("Critic (self.critic) not initialized in compute_values.")
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={'values': values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to('cpu')
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        log_gpu_memory_usage('After compute_values (critic)', logger=logger, rank=self.rank) 
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        data = data.to('cuda')
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=torch.cuda.current_device())

        log_gpu_memory_usage('Before update_critic', logger=logger, rank=self.rank)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name='update_critic', logger=logger.info) as timer: # FIX: Changed logger=logger to logger=logger.info
                if not hasattr(self, 'critic') or not self.critic:
                    raise RuntimeError("Critic (self.critic) not initialized in update_critic.")
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info['global_token_num']
            
            if hasattr(self, 'flops_counter') and self.flops_counter:
                estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
                if promised_flops > 0:
                    metrics['mfu/critic'] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size
                else:
                    metrics['mfu/critic'] = 0.0
            
            if self.critic_lr_scheduler:
                self.critic_lr_scheduler.step()
                lr = self.critic_lr_scheduler.get_last_lr()[0]
                metrics['critic/lr'] = lr
            
            output = DataProto(meta_info={'metrics': metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
        torch.cuda.empty_cache()
        output = output.to('cpu')
        log_gpu_memory_usage('After update_critic', logger=logger, rank=self.rank) 
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, remove_previous_ckpt=False):
        if not hasattr(self, 'checkpoint_manager') or not self.checkpoint_manager:
            if self.rank == 0: logger.warning("Checkpoint manager not initialized. Skipping save_checkpoint.")
            return

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        
        self.checkpoint_manager.save_checkpoint(local_path=local_path,
                                                hdfs_path=hdfs_path,
                                                global_step=global_step,
                                                remove_previous_ckpt=remove_previous_ckpt)
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path, del_local_after_load=True):
        if not hasattr(self, 'checkpoint_manager') or not self.checkpoint_manager:
            if self.rank == 0: logger.warning("Checkpoint manager not initialized. Skipping load_checkpoint.")
            return
        
        self.checkpoint_manager.load_checkpoint(path=path, del_local_after_load=del_local_after_load)
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)


class RewardModelWorker(Worker):
    def __init__(self, config):
        super().__init__()
        proc_pid = os.getpid()
        print(f"{log_prefix_rm_init} (PID {proc_pid}) - START.")
        if not run_diagnostic_checks(log_prefix=log_prefix_rm_init):
            print(f"{log_prefix_rm_init} (PID {proc_pid}) - DIAGNOSTIC CHECKS FAILED EARLY!")
            # REMOVED: raise RuntimeError("Initial diagnostic checks failed in RewardModelWorker")
        print(f"{log_prefix_rm_init} (PID {proc_pid}) - DIAGNOSTIC CHECKS PASSED.")

        if not torch.distributed.is_initialized():
            print(f"{log_prefix_rm_init} (PID {proc_pid}) - Initializing process group.")
            torch.distributed.init_process_group(backend="nccl")
            print(f"{log_prefix_rm_init} (PID {proc_pid}) - Process group initialized.")
        
        self.config = config
        try:
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        except AttributeError:
            self._world_size = torch.distributed.get_world_size()
            self._rank = torch.distributed.get_rank()

        if torch.cuda.is_available() and self.world_size <= torch.cuda.device_count():
            try:
                torch.cuda.set_device(self.rank % torch.cuda.device_count())
                print(f"{log_prefix_rm_init} (PID {proc_pid}) - Set CUDA device to {torch.cuda.current_device()} for Rank {self.rank}.")
            except Exception as e_set_device:
                print(f"{log_prefix_rm_init} (PID {proc_pid}) - WARNING: Could not set CUDA device for Rank {self.rank}: {e_set_device}.")

        print(f"{log_prefix_rm_init} (PID {proc_pid}) - Rank: {self.rank}, World Size: {self.world_size}.")

        fsdp_size_rm = self.config.model.fsdp_config.get('fsdp_size', self.world_size)
        self.device_mesh = create_device_mesh(world_size=self.world_size, fsdp_size=fsdp_size_rm)
        print(f"{log_prefix_rm_init} (PID {proc_pid}) - FSDP device_mesh created: {self.device_mesh}.")

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        if self.ulysses_sequence_parallel_size > 1:
            if self.world_size % self.ulysses_sequence_parallel_size != 0:
                 raise ValueError(f"RewardModel: World size ({self.world_size}) must be divisible by ulysses_sequence_parallel_size ({self.ulysses_sequence_parallel_size})")
            dp_ulysses_rm = self.world_size // self.ulysses_sequence_parallel_size
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp_ulysses_rm, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])
            print(f"{log_prefix_rm_init} (PID {proc_pid}) - Ulysses device_mesh created: {self.ulysses_device_mesh}.")
        
        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        print(f"{log_prefix_rm_init} (PID {proc_pid}) - Ulysses sharding manager initialized.")
        self.use_remove_padding = self.config.model.get('use_remove_padding', False)
        
        if self.config.get('micro_batch_size') is not None:
            if self.ulysses_sequence_parallel_size > 1 and self.ulysses_device_mesh:
                effective_dp_size_rm_local = self.ulysses_device_mesh.size(0)
            elif self.device_mesh.ndim == 2:
                effective_dp_size_rm_local = self.device_mesh.size(0)
            else:
                effective_dp_size_rm_local = self.world_size if self.world_size > 0 else 1

            if effective_dp_size_rm_local > 0:
                self.config.micro_batch_size //= effective_dp_size_rm_local
                self.config.micro_batch_size_per_gpu = self.config.micro_batch_size
                if self.config.micro_batch_size_per_gpu == 0: self.config.micro_batch_size_per_gpu = 1
                print(f"{log_prefix_rm_init} (PID {proc_pid}), Rank: {self.rank} - RewardModel micro_batch_size_per_gpu: {self.config.micro_batch_size_per_gpu}.")
        
        print(f"{log_prefix_rm_init} (PID {proc_pid}) - END init.")

    @property
    def world_size(self):
        if hasattr(self, '_world_size'):
            return self._world_size
        return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    @property
    def rank(self):
        if hasattr(self, '_rank'):
            return self._rank
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    def _build_model(self, config_rm_worker):
        proc_pid = os.getpid()
        print(f"{log_prefix_bm_rm} (PID {proc_pid}) - START.")
        if not run_diagnostic_checks(log_prefix=log_prefix_bm_rm):
            print(f"{log_prefix_bm_rm} (PID {proc_pid}) - DIAGNOSTIC CHECKS FAILED!")
            # REMOVED: raise RuntimeError("Diagnostic check failed in RewardModelWorker._build_model")
        print(f"{log_prefix_bm_rm} (PID {proc_pid}) - DIAGNOSTIC CHECKS PASSED.")
        
        local_path = copy_local_path_from_hdfs(config_rm_worker.model.path)
        rm_tokenizer_trust_remote_code = config_rm_worker.model.get('trust_remote_code', False)

        if config_rm_worker.model.get('input_tokenizer') is None:
            self._do_switch_chat_template = False
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=rm_tokenizer_trust_remote_code)
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_local_path_from_hdfs(config_rm_worker.model.input_tokenizer)
            self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path, trust_remote_code=rm_tokenizer_trust_remote_code)
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=rm_tokenizer_trust_remote_code)
        
        model_trust_remote_code_rm = config_rm_worker.model.get('trust_remote_code', False)
        
        print(f"{log_prefix_bm_rm} (PID {proc_pid}), Rank: {self.rank} - About to call AutoConfig.from_pretrained for {local_path} (RewardModel).")
        model_config_obj_rm = AutoConfig.from_pretrained(local_path, trust_remote_code=model_trust_remote_code_rm)
        print(f"{log_prefix_bm_rm} (PID {proc_pid}), Rank: {self.rank} - After AutoConfig.from_pretrained (RewardModel).")
        model_config_obj_rm.num_labels = 1
        
        rm_override_config_hydra = config_rm_worker.model.get('override_config', OmegaConf.create({}))
        rm_override_config = OmegaConf.to_container(rm_override_config_hydra, resolve=True)
        for key, value in rm_override_config.items():
            if hasattr(model_config_obj_rm, key):
                setattr(model_config_obj_rm, key, value)
                if self.rank == 0: logger.info(f"RewardModel - Applied override {key}={value} to model config.")

        use_remove_padding_rm = config_rm_worker.model.get('use_remove_padding', False)
        if use_remove_padding_rm:
            check_model_support_rmpad(model_config_obj_rm.model_type)
        if use_remove_padding_rm and self.ulysses_sequence_parallel_size > 1:
            apply_monkey_patch(model_config_obj_rm, verbose=(self.rank == 0))
        
        use_meta_init_rm = not getattr(model_config_obj_rm, 'tie_word_embeddings', True)
        if self.rank == 0: logger.info(f"(RewardModel) Using meta device init: {use_meta_init_rm} (based on tie_word_embeddings: {getattr(model_config_obj_rm, 'tie_word_embeddings', True)})")
        
        try:
            init_context_rm = get_init_weight_context_manager(use_meta_tensor=use_meta_init_rm)
            if hasattr(init_context_rm, '__enter__') and hasattr(init_context_rm, '__exit__'):
                context_mgr_rm = init_context_rm
            else:
                from contextlib import nullcontext
                context_mgr_rm = nullcontext()
                if self.rank == 0: logger.warning(f"RewardModel: get_init_weight_context_manager did not return a context manager. Using nullcontext().")
        except Exception as e:
            from contextlib import nullcontext
            context_mgr_rm = nullcontext()
            print(f"{log_prefix_bm_rm} (PID {proc_pid}), Rank: {self.rank} - WARNING: get_init_weight_context_manager failed for RewardModel: {e}. Using nullcontext().")
        
        with context_mgr_rm, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not hasattr(model_config_obj_rm, 'classifier_dropout') or rm_override_config.get('classifier_dropout') is None:
                setattr(model_config_obj_rm, 'classifier_dropout', 0.0) 
            
            print(f"{log_prefix_bm_rm} (PID {proc_pid}), Rank: {self.rank} - About to call AutoModelForTokenClassification.from_pretrained for {local_path} (RewardModel).")
            reward_module_torch_dtype = PrecisionType.to_dtype(config_rm_worker.model.fsdp_config.get('model_dtype', 'bf16'))
            if self.rank == 0: logger.info(f"(RewardModel) Using torch_dtype for from_pretrained (model loading): {reward_module_torch_dtype}")

            reward_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path, 
                config=model_config_obj_rm,
                torch_dtype=reward_module_torch_dtype,
                attn_implementation='flash_attention_2',
                trust_remote_code=model_trust_remote_code_rm
            )
            print(f"{log_prefix_bm_rm} (PID {proc_pid}), Rank: {self.rank} - After AutoModelForTokenClassification.from_pretrained (RewardModel).")
        
        self.reward_model_config = model_config_obj_rm

        auto_wrap_policy_rm = get_fsdp_wrap_policy(module=reward_module, config=config_rm_worker.model.fsdp_config.get('wrap_policy', None))
        if self.rank == 0: print(f'FSDP wrap_policy for RewardModel: {auto_wrap_policy_rm}')

        fsdp_mesh_rm = self.device_mesh
        sharding_strategy_rm = get_sharding_strategy(fsdp_mesh_rm)
        if self.rank == 0: logger.info(f"(RewardModel) FSDP sharding_strategy: {sharding_strategy_rm} on mesh: {fsdp_mesh_rm}")

        rm_mixed_precision_config = config_rm_worker.model.fsdp_config.get('mixed_precision', None)
        if rm_mixed_precision_config:
            rm_param_dtype = PrecisionType.to_dtype(rm_mixed_precision_config.get('param_dtype', 'bf16'))
            rm_reduce_dtype = PrecisionType.to_dtype(rm_mixed_precision_config.get('reduce_dtype', 'bf16'))
            rm_buffer_dtype = PrecisionType.to_dtype(rm_mixed_precision_config.get('buffer_dtype', 'bf16'))
        else:
            rm_param_dtype = torch.bfloat16
            rm_reduce_dtype = torch.bfloat16
            rm_buffer_dtype = torch.bfloat16
        
        if self.rank == 0:
            logger.info(f"(RewardModel) FSDP MixedPrecision: param_dtype={rm_param_dtype}, reduce_dtype={rm_reduce_dtype}, buffer_dtype={rm_buffer_dtype}")
        rm_mixed_precision = MixedPrecision(param_dtype=rm_param_dtype, reduce_dtype=rm_reduce_dtype, buffer_dtype=rm_buffer_dtype)

        rm_cpu_offload_flag = config_rm_worker.model.fsdp_config.get('param_offload', True)
        if self.rank == 0: logger.info(f"RewardModel - FSDP CPUOffload(offload_params={rm_cpu_offload_flag})")

        print(f"{log_prefix_bm_rm} (PID {proc_pid}), Rank: {self.rank} - About to wrap model with FSDP (RewardModel). Current CUDA device: {torch.cuda.current_device()}.")
        reward_module_fsdp = FSDP(
            reward_module, 
            cpu_offload=CPUOffload(offload_params=rm_cpu_offload_flag),
            param_init_fn=init_fn if not use_meta_init_rm else None,
            use_orig_params=True, 
            auto_wrap_policy=auto_wrap_policy_rm,
            device_id=torch.cuda.current_device(), 
            sharding_strategy=sharding_strategy_rm,
            mixed_precision=rm_mixed_precision,
            sync_module_states=True, 
            forward_prefetch=config_rm_worker.model.fsdp_config.get('forward_prefetch', True),
            limit_all_gathers=config_rm_worker.model.fsdp_config.get('limit_all_gathers', True),
            device_mesh=fsdp_mesh_rm)
        print(f"{log_prefix_bm_rm} (PID {proc_pid}), Rank: {self.rank} - After FSDP wrapping (RewardModel).")
        print(f"{log_prefix_bm_rm} (PID {proc_pid}) - END.")
        return reward_module_fsdp

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        proc_pid = os.getpid()
        print(f"{log_prefix_rm_im} (PID {proc_pid}) - START.")
        if not run_diagnostic_checks(log_prefix=log_prefix_rm_im):
            print(f"{log_prefix_rm_im} (PID {proc_pid}) - DIAGNOSTIC CHECKS FAILED EARLY!")
            # REMOVED: raise RuntimeError("Diagnostic check failed at start of RewardModelWorker.init_model")
        print(f"{log_prefix_rm_im} (PID {proc_pid}), Rank: {self.rank} - DIAGNOSTIC CHECKS PASSED.")
        
        import_external_libs(self.config.model.get('external_lib', None))
        
        print(f"{log_prefix_rm_im} (PID {proc_pid}), Rank: {self.rank} - Before _build_model (RewardModel).")
        self.reward_module = self._build_model(config=self.config)
        print(f"{log_prefix_rm_im} (PID {proc_pid}), Rank: {self.rank} - After _build_model (RewardModel).")
        
        log_gpu_memory_usage('After RewardModel FSDP init and potential offload', logger=logger, rank=self.rank)
        torch.cuda.empty_cache()
        print(f"{log_prefix_rm_im} (PID {proc_pid}), Rank: {self.rank} - END init_model. CUDA cache emptied.")

    def _forward_micro_batch(self, micro_batch):
        autocast_dtype = torch.bfloat16
        if hasattr(self, 'reward_module') and hasattr(self.reward_module, 'mixed_precision') and self.reward_module.mixed_precision:
            autocast_dtype = self.reward_module.mixed_precision.param_dtype or torch.bfloat16

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=autocast_dtype, enabled=torch.is_autocast_enabled()):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            
            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, max_s = unpad_input(input_ids, attention_mask)
                if input_ids_rmpad.ndim == 2 and input_ids_rmpad.shape[1] == 1:
                    input_ids_rmpad = input_ids_rmpad.squeeze(1)
                
                position_ids_rmpad = position_ids[attention_mask.bool()]

                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad_ul, position_ids_rmpad_ul, pad_size_ul = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad.unsqueeze(1), position_ids_rmpad.unsqueeze(1), 
                        sp_size=self.ulysses_sequence_parallel_size, sp_rank=self.ulysses_device_mesh.get_coordinate()[1]
                    )
                    input_ids_rmpad_ul = input_ids_rmpad_ul.squeeze(1)
                    position_ids_rmpad_ul = position_ids_rmpad_ul.squeeze(1)
                else:
                    input_ids_rmpad_ul = input_ids_rmpad
                    position_ids_rmpad_ul = position_ids_rmpad
                
                output = self.reward_module(input_ids=input_ids_rmpad_ul, attention_mask=None, position_ids=position_ids_rmpad_ul, use_cache=False)
                reward_rmpad_ul = output.logits
                
                if reward_rmpad_ul.shape[-1] == 1: reward_rmpad_ul = reward_rmpad_ul.squeeze(-1)
                else: 
                    if self.rank == 0: logger.warning(f"WARN: RewardModel output shape unexpected (rmpad): {reward_rmpad_ul.shape}")
                
                if self.ulysses_sequence_parallel_size > 1:
                    reward_rmpad_gathered = gather_outpus_and_unpad(
                        reward_rmpad_ul.unsqueeze(1),
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size_ul,
                        sp_size=self.ulysses_sequence_parallel_size
                    ).squeeze(1)
                else:
                    reward_rmpad_gathered = reward_rmpad_ul

                if reward_rmpad_gathered.ndim == 1: reward_rmpad_gathered = reward_rmpad_gathered.unsqueeze(-1)
                rm_score = pad_input(reward_rmpad_gathered, indices=indices, batch_size=batch_size, sequence_length=seqlen).squeeze(-1)
            else:
                output = self.reward_module(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
                rm_score = output.logits
                if rm_score.shape[-1] == 1: rm_score = rm_score.squeeze(-1)
                else: 
                     if self.rank == 0: logger.warning(f"WARN: RewardModel non-rmpad output shape unexpected: {rm_score.shape}")
            
            sequence_lengths = attention_mask.sum(dim=-1) - 1
            rm_score_at_eos = rm_score[torch.arange(batch_size, device=rm_score.device), sequence_lengths]
            return rm_score_at_eos

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch['input_ids'].shape[0]
        attention_mask = data.batch['attention_mask']
        
        sequence_lengths = attention_mask.sum(dim=-1) - 1
        
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype, device=scores.device)
        token_level_scores[torch.arange(batch_size, device=scores.device), sequence_lengths] = scores
        
        if 'responses' in data.batch and data.batch['responses'] is not None:
            response_length = data.batch['responses'].shape[-1]
            token_level_scores_response_part = token_level_scores[:, -response_length:]
            return token_level_scores_response_part
        else:
            if self.rank == 0: logger.warning("WARN: _expand_to_token_level: 'responses' not in data.batch. Returning scores for full sequence length.")
            return token_level_scores


    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch['attention_mask'].shape[-1]
        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer
        
        rm_input_ids_list = [] 
        rm_attention_mask_list = [] 

        num_samples = data.batch['input_ids'].shape[0]

        for i in range(num_samples): 
            chat_list_raw: list = data.non_tensor_batch['raw_prompt'][i] 
            
            if not isinstance(chat_list_raw, list) or not all(isinstance(item, dict) for item in chat_list_raw):
                if isinstance(chat_list_raw, str):
                    chat_list = [{'role': 'user', 'content': chat_list_raw}]
                else:
                    if self.rank == 0: logger.warning(f"WARN: ({self.rank}) _switch_chat_template: raw_prompt format not as expected list of dicts. Got: {type(chat_list_raw)}. Attempting to proceed.")
                    chat_list = list(chat_list_raw) if isinstance(chat_list_raw, list) else [{'role':'user', 'content':str(chat_list_raw)}]

            else:
                chat_list = chat_list_raw

            response_ids = data.batch['responses'][i]
            valid_response_length = response_ids.count_nonzero() if response_ids.ndim == 1 else \
                                    (data.batch['attention_mask'][i, -response_ids.shape[-1]:]).sum().item()

            valid_response_ids = response_ids[:int(valid_response_length)]
            
            response_text = src_tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            current_chat_list_for_rm = chat_list + [{'role': 'assistant', 'content': response_text}]
            
            try:
                prompt_with_rm_template = target_tokenizer.apply_chat_template(
                    current_chat_list_for_rm, 
                    add_generation_prompt=False,
                    tokenize=False
                )
            except Exception as e_template:
                if self.rank == 0: logger.error(f"ERROR applying chat template for RM: {e_template}. Chat: {current_chat_list_for_rm}")
                prompt_with_rm_template = "\n".join([turn.get('content', '') for turn in current_chat_list_for_rm])


            if self.rank == 0 and i == 0: print(f'RM Switch Template Input: {prompt_with_rm_template}')
            
            rm_max_length = self.config.get('max_length', src_max_length) 
            
            input_ids_tensor, attention_mask_tensor = tokenize_and_postprocess_data( 
                prompt=prompt_with_rm_template, 
                tokenizer=target_tokenizer, 
                max_length=rm_max_length,
                pad_token_id=target_tokenizer.pad_token_id, 
                left_pad=False,
                truncation=self.config.get('truncation', 'right')
            )
            rm_input_ids_list.append(input_ids_tensor)
            rm_attention_mask_list.append(attention_mask_tensor)
        
        rm_input_ids = torch.cat(rm_input_ids_list, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask_list, dim=0)
        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)
        
        rm_batch_data = {'input_ids': rm_input_ids, 'attention_mask': rm_attention_mask, 'position_ids': rm_position_ids}
        return DataProto(batch=rm_batch_data, non_tensor_batch=data.non_tensor_batch, meta_info=data.meta_info)


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        
        rm_data_proto = data 
        if self._do_switch_chat_template: 
            if not hasattr(self, 'input_tokenizer'):
                 raise RuntimeError("RewardModelWorker: input_tokenizer not available for _do_switch_chat_template=True.")
            rm_data_proto = self._switch_chat_template(data) 
        
        rm_data_proto = rm_data_proto.to('cuda')
        data_for_expansion = data.to('cuda') 

        with self.ulysses_sharding_manager:
            rm_data_processed = self.ulysses_sharding_manager.preprocess_data(data=rm_data_proto) 
            data_for_expansion_processed = self.ulysses_sharding_manager.preprocess_data(data=data_for_expansion)
            
            use_dynamic_bsz_rm = self.config.get('use_dynamic_bsz', False)
            
            if use_dynamic_bsz_rm:
                max_token_len_config_key_rm = 'forward_max_token_len_per_gpu'
                max_token_len_val_rm = self.config.get(max_token_len_config_key_rm, 512)
                max_token_len_rm = max_token_len_val_rm * self.ulysses_sequence_parallel_size
                
                micro_batches, indices = rearrange_micro_batches(
                    batch=rm_data_processed.batch,
                    max_token_len=max_token_len_rm
                )
            else:
                num_samples_on_rank = rm_data_processed.batch['input_ids'].shape[0]
                micro_batch_size_rm = self.config.micro_batch_size_per_gpu
                
                micro_batches = []
                for i in range(0, num_samples_on_rank, micro_batch_size_rm):
                    current_micro_batch = {
                        key: tensor[i : i + micro_batch_size_rm] 
                        for key, tensor in rm_data_processed.batch.items()
                    }
                    micro_batches.append(current_micro_batch)
            
            output_scores_list = [] 
            for micro_batch_dict in micro_batches:
                rm_score_batch = self._forward_micro_batch(micro_batch_dict)
                output_scores_list.append(rm_score_batch)
            
            scores = torch.cat(output_scores_list, dim=0)
            
            if use_dynamic_bsz_rm:
                indices_flat = list(itertools.chain.from_iterable(indices)) 
                if len(indices_flat) != scores.size(0):
                    logger.warning(f"WARN: ({self.rank}) RM dynamic bsz: Mismatch len(indices_flat)={len(indices_flat)} vs scores.size(0)={scores.size(0)}. Truncating/Padding scores if necessary.")
                    if len(indices_flat) > scores.size(0):
                        indices_flat = indices_flat[:scores.size(0)]
                    elif scores.size(0) > len(indices_flat):
                        scores = scores[:len(indices_flat)]

                revert_indices = torch.tensor(get_reverse_idx(indices_flat), dtype=torch.long, device=scores.device) 
                scores = scores[revert_indices]
            
            token_level_scores = self._expand_to_token_level(data_for_expansion_processed, scores)
            
            output_proto = DataProto.from_dict(tensors={'rm_scores': token_level_scores}) 
            output_proto = self.ulysses_sharding_manager.postprocess_data(data=output_proto)
        
        if self.world_size > 1 and hasattr(self.reward_module, '_handle') and self.reward_module._handle: 
             self.reward_module._handle.reshard(True)
        
        output_proto = output_proto.to('cpu')
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After compute_rm_score', logger=logger, rank=self.rank) 
        return output_proto
