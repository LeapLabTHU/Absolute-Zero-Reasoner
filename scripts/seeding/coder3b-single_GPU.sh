set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_memory_monitor_refresh_ms=0
export RAY_LOGGING_LEVEL=DEBUG
export HYDRA_FULL_ERROR=1

# Define default paths, allowing overrides via environment variables
OUTPUT_SEED_PATH=${OUTPUT_SEED_PATH:-data/3b_coder_seed_io.jsonl}
OUTPUT_ERROR_SEED_PATH=${OUTPUT_ERROR_SEED_PATH:-data/3b_coder_error_seed_io.jsonl}
OUTPUT_CODE_F_SEED_PATH=${OUTPUT_CODE_F_SEED_PATH:-data/3b_coder_code_f_seed_io.jsonl}

python -m absolute_zero_reasoner.main_azr_ppo \
    data.shuffle=True \
    actor_rollout_ref.ref.include_ref=False \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files=data/code_reason/test_answer.parquet \
    data.val_files=data/code_reason/test_answer.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=1312 \
    data.max_prompt_length=6144 \
    data.max_validation_prompt_length=6144 \
    data.max_response_length=8096 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.pretrained_tokenizer=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='azr' \
    trainer.experiment_name='3b_coder_seed_single_gpu' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.remove_previous_ckpt_in_save=True \
    trainer.del_local_ckpt_after_load=True \
    trainer.test_freq=10 \
    +trainer.val_before_train=False \
    reward_fn.extraction_type=answer_conditional \
    reward_fn.math_metric=math_verify \
    trainer.val_generations_to_log_to_wandb=0 \
    azr.data_selection_strategy.update_iteration=1 \
    azr.seed_dataset=null \
    azr.error_seed_dataset=null \
    azr.code_f_seed_dataset=null \
    azr.output_seed_path=${OUTPUT_SEED_PATH} \
    azr.output_error_seed_path=${OUTPUT_ERROR_SEED_PATH} \
    azr.output_code_f_seed_path=${OUTPUT_CODE_F_SEED_PATH} \
    azr.pretrain_pred_steps=-1 \
    azr.executor=qwq \
    azr.ast_check=True \
    azr.reward.n_samples=8 \
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
    azr.data_selection_strategy.seed_batch_factor=4 \
    azr.data_selection_strategy.valid_program_filter=all \
    azr.data_selection_strategy.max_programs=16384 \
    azr.data_selection_strategy.batched_estimate=False \
    azr.reward.generation_reward_config.intrinsic_combine_method=sum \
    azr.gen_data_probabilities_strategy=uniform \
    trainer.resume_mode=auto \
    azr.data_selection_strategy.composite_start_step=-1 \
    azr.data_selection_strategy.composite_chance=0.0 \
    azr.reward.generation_reward_config.remove_comments=True \
    azr.reward.generation_reward_config.remove_after_return=True \
    azr.reward.generation_reward_config.use_original_code_as_ref=True \
    azr.reward.generation_reward_config.remove_print=False \
    azr.data_selection_strategy.composite_function_n_min=0 \
    azr.data_selection_strategy.composite_function_n_max=0 \
    azr.reward.code_f_reward_type=binary \
    trainer.wandb_run_id=null \
    +azr.generate_seed_dataset_only=True \
    trainer.total_epochs=30 \
    \
    REMOTE_DEBUG=False \
    \
    ####### QLoRA Base Model Config (FSDP Model Loading / vLLM Rollout) #######
    # Applies to actor_rollout_ref.model when loaded by fsdp_workers.py
    actor_rollout_ref.model.override_config.load_in_4bit=True \
    actor_rollout_ref.model.override_config.bnb_4bit_quant_type="nf4" \
    actor_rollout_ref.model.override_config.bnb_4bit_compute_dtype="torch.bfloat16" \
    # actor_rollout_ref.model.override_config.bnb_4bit_use_double_quant=False \
    \
    # For vLLM rollouts
    actor_rollout_ref.rollout.vllm_quantization="bitsandbytes" \
    \
    $@
