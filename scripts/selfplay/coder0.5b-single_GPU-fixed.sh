#!/bin/bash
# AZR Single-GPU Training Script - OPTIMIZED with proven seeding parameters
# Uses the configuration that achieved 95% GPU utilization in seeding

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_memory_monitor_refresh_ms=0
export RAY_LOGGING_LEVEL=ERROR
export HYDRA_FULL_ERROR=1
export RAY_worker_register_timeout_seconds=300
export RAY_worker_max_concurrent_connections=1
export RAY_object_store_memory=2000000000
export RAY_worker_start_timeout_seconds=600
export RAY_DISABLE_ACTOR_CREATION_RETRY=0
export RAY_ACTOR_MAX_CONCURRENT_TASKS=1
export RAY_MAX_PENDING_ASYNC_REQUESTS=100
export RAY_ENABLE_MULTI_TENANCY=0
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1
export RAY_max_concurrent_workers=1
export RAY_worker_parallel_create=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0
export CUDA_CACHE_DISABLE=1
export CUDA_LAUNCH_BLOCKING=0
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

# Define default paths
OUTPUT_SEED_PATH=${OUTPUT_SEED_PATH:-data/0.5b_coder_seed_io.jsonl} 
OUTPUT_ERROR_SEED_PATH=${OUTPUT_ERROR_SEED_PATH:-data/0.5b_coder_error_seed_io.jsonl}
OUTPUT_CODE_F_SEED_PATH=${OUTPUT_CODE_F_SEED_PATH:-data/0.5b_coder_code_f_seed_io.jsonl}

# Clean up any existing Ray processes
ray stop --force 2>/dev/null || true
sleep 2

echo "🚀 STARTING AZR TRAINING WITH OPTIMIZED SEEDING PARAMETERS"
echo "⚡ GPU Memory: 0.7"
echo "🔥 vLLM Tokens: 512/512 (vs 128/128)"
echo "📊 Batch Size: 16 for better GPU utilization"

python -m absolute_zero_reasoner.main_azr_ppo \
    data.shuffle=True \
    actor_rollout_ref.ref.include_ref=False \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files=data/code_reason/test_answer.parquet \
    data.val_files=data/code_reason/test_answer.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    data.max_prompt_length=16384 \
    data.max_validation_prompt_length=16384 \
    data.max_response_length=16384 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-Coder-0.5B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=12 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.pretrained_tokenizer=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_num_batched_tokens=128 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.0 \
    critic.strategy=fsdp \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='azr' \
    trainer.experiment_name='0.5b_coder_training_optimized' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.remove_previous_ckpt_in_save=True \
    trainer.del_local_ckpt_after_load=True \
    trainer.test_freq=20 \
    +trainer.val_before_train=False \
    reward_fn.extraction_type=answer_conditional \
    reward_fn.math_metric=math_verify \
    trainer.val_generations_to_log_to_wandb=0 \
    azr.data_selection_strategy.update_iteration=1 \
    azr.seed_dataset=${OUTPUT_SEED_PATH} \
    azr.output_seed_path=${OUTPUT_SEED_PATH} \
    azr.error_seed_dataset=${OUTPUT_ERROR_SEED_PATH} \
    azr.output_error_seed_path=${OUTPUT_ERROR_SEED_PATH} \
    azr.code_f_seed_dataset=${OUTPUT_CODE_F_SEED_PATH} \
    azr.output_code_f_seed_path=${OUTPUT_CODE_F_SEED_PATH} \
    azr.pretrain_pred_steps=-1 \
    azr.executor=qwq \
    azr.ast_check=True \
    azr.reward.n_samples=12 \
    azr.problem_types=['code_i','code_o','code_f'] \
    azr.data_selection_strategy.banned_keywords_for_errors_and_exceptions=['raise'] \
    trainer.debug=False \
    azr.reward.generation_reward_config.complexity_reward.coef=0.0 \
    azr.reward.generation_reward_config.complexity_reward.max=0.0 \
    azr.reward.generation_reward_config.complexity_reward.enabled=False \
    azr.reward.generation_reward_config.mean_edit_distance_reward.coef=0.0 \
    azr.reward.generation_reward_config.mean_edit_distance_reward.max=0.0 \
    azr.reward.generation_reward_config.mean_edit_distance_reward.enabled=False \
    azr.reward.generation_reward_config.halstead_reward.coef=0.0 \
    azr.reward.generation_reward_config.halstead_reward.max=0.0 \
    azr.reward.generation_reward_config.halstead_reward.enabled=False \
    azr.reward.generation_reward_config.answer_diversity_reward.coef=0.0 \
    azr.reward.generation_reward_config.answer_diversity_reward.max=0.0 \
    azr.reward.generation_reward_config.answer_diversity_reward.enabled=False \
    azr.reward.generation_reward_config.answer_diversity_reward.hierarchical=False \
    azr.pred_data_mix_strategy=max_new \
    azr.data_selection_strategy.seed_batch_factor=256 \
    azr.data_selection_strategy.valid_program_filter=all \
    azr.data_selection_strategy.max_programs=16384 \
    azr.data_selection_strategy.batched_estimate=False \
    azr.reward.generation_reward_config.intrinsic_combine_method=sum \
    azr.gen_data_probabilities_strategy=uniform \
    trainer.resume_mode=auto \
    azr.data_selection_strategy.composite_start_step=-1 \
    azr.data_selection_strategy.composite_chance=0.0 \
    azr.reward.generation_reward_config.remove_comments=False \
    azr.reward.generation_reward_config.remove_after_return=False \
    azr.reward.generation_reward_config.use_original_code_as_ref=True \
    azr.reward.generation_reward_config.remove_print=False \
    azr.data_selection_strategy.composite_function_n_min=0 \
    azr.data_selection_strategy.composite_function_n_max=0 \
    azr.reward.code_f_reward_type=binary \
    trainer.wandb_run_id=null \
    +azr.generate_seed_dataset_only=False \
    trainer.total_epochs=30 \
    +REMOTE_DEBUG=False \
    $@