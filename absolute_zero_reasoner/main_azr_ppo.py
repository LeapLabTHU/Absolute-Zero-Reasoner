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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
import ray
import hydra
from pathlib import Path
from pprint import pprint
import os
import torch
import time
import traceback
import subprocess

from omegaconf import OmegaConf
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils import hf_tokenizer

from absolute_zero_reasoner.trainer.ppo.azr_ray_trainer import CodeIORayPPOTrainer
from absolute_zero_reasoner.rewards.reward_managers import CodeIORewardManager
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from transformers import AutoTokenizer


@hydra.main(config_path='configs', config_name='azr_ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config, compute_score=None):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(address=None, runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}}, object_store_memory=1000000000, _plasma_directory="/tmp", num_cpus=4, ignore_reinit_error=True)

    ray.get(main_task.remote(config, compute_score))


@ray.remote
def main_task(config, compute_score_fn):
    # import time
    # proc_pid = os.getpid()
    # print(f"{log_prefix}: Entering main_task (orchestrator - no GPU needed).")
    # pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    if config.trainer.debug:
        import debugpy
        debugpy.listen(("0.0.0.0", config.trainer.debug_port))
        print(f"Debugger listening on port {config.trainer.debug_port}")
        debugpy.wait_for_client()
        print("Debugger attached!")

    # generator one batch, solver one batch
    config.actor_rollout_ref.actor.ppo_mini_batch_size = config.data.train_batch_size * len(config.azr.problem_types) * (2 if config.azr.train_propose else 1)
    pprint(f"auto setting ppo_mini_batch_size: {config.actor_rollout_ref.actor.ppo_mini_batch_size}")
    config.azr.data_selection_strategy.data_len = config.data.train_batch_size * config.azr.data_selection_strategy.update_iteration
    pprint(f"auto setting data_len: {config.azr.data_selection_strategy.data_len}")

    config.trainer.default_local_dir = (Path(config.trainer.default_local_dir) / config.data.train_files.split('/')[-1].split('.')[0] / config.actor_rollout_ref.model.path.split('/')[-1] / config.reward_fn.extraction_type).as_posix()

    assert not (not config.azr.reward.generation_reward_config.reject_multiple_functions and config.azr.data_selection_strategy.composite_function_n_min > 0), "If reject_multiple_functions is False, composite_function_n_min must be 0"

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    tokenizer = hf_tokenizer(local_path)

    # base model chat template
    if config.actor_rollout_ref.model.pretrained_tokenizer:
        tokenizer.chat_template = "{%- for message in messages -%}{{- '\n' if not loop.first -}}{{- message['content'] -}}{%- endfor -%}"

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError


    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }
    if config.actor_rollout_ref.ref.include_ref:
        mapping[Role.RefPolicy] = global_pool_id
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)

    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = CodeIORewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        reward_fn_extraction_type=config.reward_fn.extraction_type,
        math_metric=config.reward_fn.math_metric,
        split='train',
        splitter=config.reward_fn.splitter,
        output_path=config.trainer.default_local_dir,
        max_prompt_length=config.data.max_prompt_length,
        generation_reward_config=config.azr.reward.generation_reward_config,
        valid_program_filter=config.azr.data_selection_strategy.valid_program_filter,
        debug=config.trainer.debug,
        extract_code_block=config.azr.reward.extract_code_block,
        code_f_reward_type=config.azr.reward.code_f_reward_type,
        boxed_retry=config.reward_fn.boxed_retry,
    )

    val_reward_fn = CodeIORewardManager(
        tokenizer=tokenizer,
        num_examine=1,
        reward_fn_extraction_type=config.reward_fn.extraction_type,
        math_metric=config.reward_fn.math_metric,
        split='test',
        splitter=config.reward_fn.splitter,
        output_path=config.trainer.default_local_dir,
        max_prompt_length=config.data.max_prompt_length,
        generation_reward_config=config.azr.reward.generation_reward_config,
        valid_program_filter=config.azr.data_selection_strategy.valid_program_filter,
        debug=config.trainer.debug,
        extract_code_block=config.azr.reward.extract_code_block,
        code_f_reward_type=config.azr.reward.code_f_reward_type,
        boxed_retry=config.reward_fn.boxed_retry,
    )

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    wandb_tags = [
        'codeio', config.azr.pred_data_mix_strategy, 'executor-' + config.azr.executor,
        config.azr.data_selection_strategy.valid_program_filter, config.azr.gen_data_probabilities_strategy,
    ]
    wandb_tags.extend(config.azr.problem_types)
    if config.trainer.wandb_tags is not None:
        config.trainer.wandb_tags = wandb_tags + config.trainer.wandb_tags.split(',')
    else:
        config.trainer.wandb_tags = wandb_tags

    trainer = CodeIORayPPOTrainer(
        # past_epoch_window=config.azr.past_epoch_window,
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        # ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,  # Add this line
        val_reward_fn=val_reward_fn  # Add this line if validation rewards are needed
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                # Better Ray cleanup before retry
                # import time
                print(f"Cleaning up Ray state before retry {attempt}...")
                
                # Force shutdown and cleanup
                try:
                    ray.shutdown()
                    time.sleep(3)
                except:
                    pass
                
                # Kill any remaining Ray processes
                try:
                    subprocess.run(["ray", "stop", "--force"], check=False, capture_output=True)
                    time.sleep(2)
                except:
                    pass
                
                # Clean up placement groups
                try:
                    subprocess.run(["pkill", "-f", "ray"], check=False, capture_output=True)
                    time.sleep(1)
                except:
                    pass
                
                # Reinitialize Ray with different parameters
                ray.init(
                    address=None, 
                    runtime_env={
                        'env_vars': {
                            'TOKENIZERS_PARALLELISM': 'true', 
                            'NCCL_DEBUG': 'WARN',
                            'RAY_DISABLE_IMPORT_WARNING': '1'
                        }
                    }, 
                    object_store_memory=1500000000 + (attempt * 500000000),  # Increase memory each retry
                    _plasma_directory="/tmp", 
                    num_cpus=4, 
                    ignore_reinit_error=True,
                    log_to_driver=False  # Reduce logging to avoid conflicts
                )
                
            trainer.init_workers()
            break
        except Exception as e:
            print(f"Worker initialization attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in 15 seconds...")
                time.sleep(15)  # Longer wait between retries
            else:
                print("All worker initialization attempts failed")
                raise
    trainer.fit()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        import sys
        traceback.print_exc()
        sys.exit(0)
    except Exception as e:
        traceback.print_exc()
        os._exit(1)
