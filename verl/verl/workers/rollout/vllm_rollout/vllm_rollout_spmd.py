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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
# from torch import nn # nn is not used in this file

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index_tensor = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)
    if non_pad_index_tensor.numel() > 0: # Check if tensor is not empty
        non_pad_index = non_pad_index_tensor[0][0]
        token_ids = prompt_token_ids[non_pad_index:].tolist()
    else: # Handle case where prompt_token_ids might be all pad_token_id or empty
        token_ids = prompt_token_ids.tolist() # Or handle as an error/empty list as appropriate
    return token_ids


class vLLMRollout(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            model_path: Path to the model.
            config: DictConfig for rollout settings.
            tokenizer: The task/model tokenizer. # THIS IS THE TOKENIZER OBJECT
            model_hf_config: The huggingface config to initialize the generating model in vllm. # THIS IS THE MODEL'S HF CONFIG
            **kwargs: Additional arguments, e.g., train_tp for Megatron Backend.
        """
        super().__init__()
        self.config = config
        # self.tokenizer = tokenizer # Store tokenizer if needed for other methods, though not directly used in this snippet
        # self.model_hf_config = model_hf_config # Store model_hf_config if needed

        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                              num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        # Get vLLM quantization setting from Hydra config (config.rollout.vllm_quantization)
        # Default to None if not specified. This 'config' is self.config.rollout from ActorRolloutRefWorker
        vllm_quantization_method = config.get("vllm_quantization", None)
        # if torch.distributed.get_rank() == 0: # Log only on rank 0
        #     if vllm_quantization_method:
        #         print(f"INFO: [VLLMRollout] vLLM quantization method specified: {vllm_quantization_method}")
        #     else:
        #         print("INFO: [VLLMRollout] No vLLM quantization method specified in config.")

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True, 
            tensor_parallel_size=tensor_parallel_size, 
            distributed_executor_backend="external_launcher", 
            dtype=config.dtype, 
            enforce_eager=config.enforce_eager, 
            gpu_memory_utilization=config.gpu_memory_utilization, 
            disable_custom_all_reduce=True, 
            skip_tokenizer_init=False, 
            max_model_len=config.prompt_length + config.response_length, 
            disable_log_stats=config.disable_log_stats, 
            max_num_batched_tokens=max_num_batched_tokens, 
            max_num_seqs=config.get("max_num_seqs", 256),
            enable_chunked_prefill=config.enable_chunked_prefill, 
            quantization=vllm_quantization_method,    
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        ### START MODIFIED SECTION ###
        # Determine EOS token ID reliably
        actual_eos_token_id = None
        if tokenizer and hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            actual_eos_token_id = tokenizer.eos_token_id
            # if torch.distributed.get_rank() == 0:
            #     print(f"INFO: [VLLMRollout] Using EOS token ID from tokenizer: {actual_eos_token_id}")
        elif model_hf_config and hasattr(model_hf_config, 'eos_token_id') and model_hf_config.eos_token_id is not None:
            actual_eos_token_id = model_hf_config.eos_token_id
            # if torch.distributed.get_rank() == 0:
            #     print(f"INFO: [VLLMRollout] Using EOS token ID from model_hf_config: {actual_eos_token_id}")
        # else:
        #     if torch.distributed.get_rank() == 0:
        #         print("CRITICAL WARNING: [VLLMRollout] Could not determine EOS token ID from tokenizer or model_hf_config. Generation might not stop correctly.")
        
        # Determine PAD token ID reliably for _pre_process_inputs
        actual_pad_token_id = None
        if tokenizer and hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            actual_pad_token_id = tokenizer.pad_token_id
            # if torch.distributed.get_rank() == 0:
            #     print(f"INFO: [VLLMRollout] Using PAD token ID from tokenizer: {actual_pad_token_id}")
        elif model_hf_config and hasattr(model_hf_config, 'pad_token_id') and model_hf_config.pad_token_id is not None:
            actual_pad_token_id = model_hf_config.pad_token_id
            # if torch.distributed.get_rank() == 0:
            #     print(f"INFO: [VLLMRollout] Using PAD token ID from model_hf_config: {actual_pad_token_id}")
        elif actual_eos_token_id is not None: # Fallback pad_token_id to eos_token_id if pad is not explicitly set
            actual_pad_token_id = actual_eos_token_id
            # if torch.distributed.get_rank() == 0:
            #     print(f"INFO: [VLLMRollout] PAD token ID not found, defaulting to EOS token ID: {actual_pad_token_id}")
        else:
            # if torch.distributed.get_rank() == 0:
            #     print("CRITICAL WARNING: [VLLMRollout] Could not determine PAD token ID. Using 0 as default for preprocessing, this might be incorrect.")
            actual_pad_token_id = 0 # Absolute fallback, check if this is appropriate for your models
        
        self.pad_token_id = actual_pad_token_id


        # Initialize sampling_kwargs with some defaults that can be overridden by Hydra config
        sampling_kwargs = dict( 
            n=config.get('n', 1), # Default to 1 if not in config
            logprobs=config.get('logprobs', 1),  # Default to 1 if not in config
            max_tokens=config.response_length, # This should be defined in config
            ignore_eos=config.get('ignore_eos', False) # Default to False
        )

        # Explicitly set stop_token_ids if EOS is known
        if actual_eos_token_id is not None:
            sampling_kwargs['stop_token_ids'] = [actual_eos_token_id]
            # If ignore_eos is True in config but we are setting stop_token_ids with EOS,
            # it's a bit contradictory. VLLM's ignore_eos would likely take precedence.
            # For clarity, if stop_token_ids contains EOS, ignore_eos should ideally be False.
            # if sampling_kwargs['ignore_eos'] is True and torch.distributed.get_rank() == 0:
            #      print(f"WARNING: [VLLMRollout] 'ignore_eos' is True in config, but 'stop_token_ids' is being set with EOS token ({actual_eos_token_id}). 'ignore_eos' will likely cause vLLM to ignore this EOS in stop_token_ids.")
        
        if vllm_version != '0.3.1': 
            sampling_kwargs['detokenize'] = False
        ### END MODIFIED SECTION ###

        # supporting adding any sampling params from the config file
        # This loop will override any values set above if they are also in the hydra `config` (config is self.config.rollout)
        for k_sampling in config.keys(): 
            if hasattr(SamplingParams(), str(k_sampling)):
                # Log if a pre-set value is being overridden by Hydra config
                # if k_sampling in sampling_kwargs and sampling_kwargs[k_sampling] != config.get(k_sampling) and torch.distributed.get_rank() == 0:
                #     print(f"INFO: [VLLMRollout] Overriding sampling_param '{k_sampling}' with value from Hydra config: {config.get(k_sampling)} (was: {sampling_kwargs[k_sampling]})")
                # elif k_sampling not in sampling_kwargs and torch.distributed.get_rank() == 0: # Log new params from config
                #      print(f"INFO: [VLLMRollout] Setting sampling_param '{k_sampling}' from Hydra config: {config.get(k_sampling)}")
                sampling_kwargs[k_sampling] = config.get(k_sampling)

        # if torch.distributed.get_rank() == 0: 
        #     print(f"INFO: [VLLMRollout] Final vLLM SamplingParams kwargs: {sampling_kwargs}")
        self.sampling_params = SamplingParams(**sampling_kwargs)

        # self.pad_token_id = tokenizer.pad_token_id # Moved reliable fetching logic above

    @contextmanager
    def update_sampling_params(self, **kwargs_update): 
        old_sampling_params_args = {}
        if kwargs_update:
            for key, value in kwargs_update.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        if len(old_sampling_params_args): 
            for key, value in old_sampling_params_args.items():
                setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs_generate) -> DataProto: 
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine: 
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        
        ### START MODIFIED SECTION for eos_token_for_masking ###
        # eos_token_id should now be reliably set in self.sampling_params.stop_token_ids
        # However, get_eos_mask still needs an explicit eos_token.
        # Let's try to get it from prompts.meta_info as before, but with a fallback.
        eos_token_for_masking = prompts.meta_info.get('eos_token_id', None)
        if eos_token_for_masking is None:
            # Fallback to the one we set for sampling_params if available
            if self.sampling_params.stop_token_ids and len(self.sampling_params.stop_token_ids) > 0:
                 # Assuming the first one in stop_token_ids is the primary EOS.
                eos_token_for_masking = self.sampling_params.stop_token_ids[0] 
                # if torch.distributed.get_rank() == 0:
                #     print(f"DEBUG: [VLLMRollout generate_sequences] EOS for masking from self.sampling_params.stop_token_ids: {eos_token_for_masking}")
            # else:
            #     # This is a critical issue if we reach here without an EOS for masking.
            #     if torch.distributed.get_rank() == 0:
            #         print(f"CRITICAL WARNING: [VLLMRollout generate_sequences] eos_token_id not found in prompts.meta_info and not in self.sampling_params.stop_token_ids. Masking might be incorrect.")
        ### END MODIFIED SECTION for eos_token_for_masking ###
        
        batch_size = idx.size(0)

        idx_list = []
        for i in range(batch_size):
            # Use self.pad_token_id which should now be reliably set in __init__
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))


        do_sample = prompts.meta_info.get('do_sample', True)
        sampling_params_override = {}
        if not do_sample:
            sampling_params_override = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0.0, 
                'n': 1
            }
        
        final_sampling_kwargs = {**sampling_params_override, **kwargs_generate}

        with self.update_sampling_params(**final_sampling_kwargs):
            # Debug print the sampling params being used for this specific call
            # if torch.distributed.get_rank() == 0:
            #     current_params_dict = {}
            #     try:
            #         # For Pydantic V2 style models (common in newer libraries like recent vLLM)
            #         if hasattr(self.sampling_params, 'model_fields'):
            #             current_params_dict = {field_name: getattr(self.sampling_params, field_name, None) for field_name in self.sampling_params.model_fields.keys()}
            #         # For Pydantic V1 style models or classes with __fields__
            #         elif hasattr(self.sampling_params, '__fields__'): # Pydantic V1
            #              current_params_dict = {field_name: getattr(self.sampling_params, field_name, None) for field_name in self.sampling_params.__fields__.keys()}
            #         # Fallback for general objects if they have a __dict__
            #         elif hasattr(self.sampling_params, '__dict__'):
            #             current_params_dict = vars(self.sampling_params).copy() # Use copy to avoid modifying original __dict__
            #         else: # Last resort, just get the repr
            #             current_params_dict = {"__repr__": repr(self.sampling_params)}
            #     except Exception as e:
            #         current_params_dict = {"error_inspecting_sampling_params": str(e), "__repr__": repr(self.sampling_params)}
                
            #     print(f"DEBUG: [VLLMRollout generate_sequences] Effective SamplingParams for vLLM.generate: {current_params_dict}")

            outputs = self.inference_engine.generate(
                prompts=None,
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=False)

        response_sequences = [] 
        for output_item in outputs: 
            for sample_output in output_item.outputs: 
                response_sequences.append(sample_output.token_ids)

        response_tensor = pad_2d_list_to_length(response_sequences, self.pad_token_id, 
                                         max_length=self.config.response_length).to(idx.device)

        num_responses_per_prompt = 1
        if outputs and outputs[0].outputs:
            num_responses_per_prompt = len(outputs[0].outputs)

        if num_responses_per_prompt > 1 and do_sample: 
            idx = idx.repeat_interleave(num_responses_per_prompt, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_responses_per_prompt, dim=0)
            position_ids = position_ids.repeat_interleave(num_responses_per_prompt, dim=0)
            effective_batch_size = batch_size * num_responses_per_prompt 
        else:
            effective_batch_size = batch_size

        seq = torch.cat([idx, response_tensor], dim=-1)
        response_length = response_tensor.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(effective_batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        final_position_ids = torch.cat([position_ids, response_position_ids], dim=-1) 
        
        ### START MODIFIED SECTION for get_eos_mask call ###
        # Use the eos_token_for_masking determined above
        # if eos_token_for_masking is None and torch.distributed.get_rank() == 0:
        #     print(f"WARNING: [VLLMRollout generate_sequences] Calling get_eos_mask with eos_token=None. This might lead to unexpected behavior or errors.")
        response_attention_mask = get_eos_mask(response_id=response_tensor, eos_token=eos_token_for_masking, dtype=attention_mask.dtype)
        ### END MODIFIED SECTION for get_eos_mask call ###

        final_attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1) 

        batch_tensordict = TensorDict( 
            {
                'prompts': idx,
                'responses': response_tensor,
                'input_ids': seq,
                'attention_mask': final_attention_mask,
                'position_ids': final_position_ids
            },
            batch_size=effective_batch_size) 

        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine: 
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch_tensordict)
