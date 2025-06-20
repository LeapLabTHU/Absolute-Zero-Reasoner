#!/bin/bash
# AZR Single-GPU Training Script - V12 (Definitive & Final)
# - Applies the complete, surgically correct set of Hydra prefixes to the FSDP
#   configuration block based on all collective error logs.
# - This is the final, working version.

# Immediately exit if any command fails.
set -e

## --------------------------------------------------------------------------
## RAY, CUDA, AND SYSTEM EXPORTS
## --------------------------------------------------------------------------
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_memory_monitor_refresh_ms=0
export RAY_LOGGING_LEVEL=ERROR
export HYDRA_FULL_ERROR=1
export RAY_worker_register_timeout_seconds=600
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
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export OMP_NUM_THREADS=8

## --------------------------------------------------------------------------
## DYNAMIC PARAMETERS FROM ORCHESTRATOR
## --------------------------------------------------------------------------
MODEL_PATH="${AZR_MODEL_PATH}"
TARGET_STEPS="${AZR_TARGET_STEPS}"

OUTPUT_SEED_PATH="${OUTPUT_SEED_PATH:-data/3b_coder_seed_io.jsonl}"
OUTPUT_ERROR_SEED_PATH="${OUTPUT_ERROR_SEED_PATH:-data/3b_coder_error_seed_io.jsonl}"
OUTPUT_CODE_F_SEED_PATH="${OUTPUT_CODE_F_SEED_PATH:-data/3b_coder_code_f_seed_io.jsonl}"

ray stop --force 2>/dev/null || true
sleep 2

echo "🚀 STARTING AZR 3B TRAINING WITH MEMORY OPTIMIZATIONS"
echo "   Base Model Path: ${MODEL_PATH}"
echo "   Target Steps: ${TARGET_STEPS}"
echo "   Cycle Number: ${AZR_CYCLE_NUM}"
echo "   CPU Offload: ENABLED"


## --------------------------------------------------------------------------
## HYDRA OVERRIDE CONFIGURATION
## --------------------------------------------------------------------------
HYDRA_OVERRIDES=(
    # --- Dynamic Training Control & PATHS ---
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "trainer.total_training_steps=${TARGET_STEPS}"
    "trainer.resume_mode=auto"
    "actor_rollout_ref.model.pretrained_tokenizer=${MODEL_PATH}"
    "+model.tokenizer_path=${MODEL_PATH}"
    "+data.tokenizer_path=${MODEL_PATH}"

    # --- QLoRA & PEFT Configuration ---
    "+actor_rollout_ref.model.override_config.load_in_4bit=True"
    "+actor_rollout_ref.model.override_config.bnb_4bit_quant_type=nf4"
    "+actor_rollout_ref.model.override_config.bnb_4bit_compute_dtype=bfloat16"
    "+actor_rollout_ref.model.override_config.bnb_4bit_use_double_quant=True"
    "+actor_rollout_ref.model.override_config.bnb_4bit_quant_storage=uint8"
    "+actor_rollout_ref.model.peft_config.enable=True"
    "+actor_rollout_ref.model.peft_config.r=64"
    "+actor_rollout_ref.model.peft_config.lora_alpha=16"
    '+actor_rollout_ref.model.peft_config.target_modules=[q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj]'
    "+actor_rollout_ref.model.peft_config.lora_dropout=0.1"
    "+actor_rollout_ref.model.peft_config.bias=none"
    "+actor_rollout_ref.model.peft_config.task_type=CAUSAL_LM"

    # --- FSDP Configuration (FINAL Corrected Prefixes) ---
    "actor_rollout_ref.actor.strategy=fsdp"
    "critic.strategy=fsdp"
    # ADD '+' because these keys DO NOT exist in the default config
    "+actor_rollout_ref.actor.fsdp_config.sharding_strategy=FULL_SHARD"
    "+actor_rollout_ref.actor.fsdp_config.cpu_offload=True"
    "+actor_rollout_ref.actor.fsdp_config.forward_prefetch=True"
    "+actor_rollout_ref.actor.fsdp_config.backward_prefetch=BACKWARD_PRE"
    "+actor_rollout_ref.actor.fsdp_config.use_orig_params=True"
    # REMOVE '+' because these keys DO exist in the default config
    "actor_rollout_ref.actor.fsdp_config.param_offload=True"
    "actor_rollout_ref.actor.fsdp_config.grad_offload=True"
    "actor_rollout_ref.ref.fsdp_config.param_offload=True"

    # --- Core Model & Memory Settings ---
    "+actor_rollout_ref.model.torch_dtype=bfloat16"
    "+actor_rollout_ref.actor.use_mixed_precision=True"
    "actor_rollout_ref.model.enable_gradient_checkpointing=True"
    "+actor_rollout_ref.model.gradient_checkpointing_ratio=0.5"
    "actor_rollout_ref.model.use_remove_padding=True"
    "+torch.cuda.empty_cache=True"

    # --- Optimizer & PPO Settings (Tuned for Higher GPU Update Utilization) ---
    "actor_rollout_ref.actor.optim.lr=2e-6"
    # Increased accumulation to create a larger effective batch for the optimizer
    "+actor_rollout_ref.actor.gradient_accumulation_steps=4"
    # Set mini-batch to match the new effective batch size (6 * 4 = 24)
    "actor_rollout_ref.actor.ppo_mini_batch_size=24"
    # Kept micro-batch at 6, as VRAM is already maxed out during generation
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=6"
    "algorithm.adv_estimator=reinforce_plus_plus"
    "actor_rollout_ref.actor.use_kl_loss=False"

    # --- Data & Rollout Configuration ---
    "data.shuffle=True"
    "actor_rollout_ref.ref.include_ref=False"
    "data.train_files=data/code_reason/test_answer.parquet"
    "data.val_files=data/code_reason/test_answer.parquet"
    "data.train_batch_size=2"
    "data.val_batch_size=2"
    "data.max_prompt_length=2048"
    "data.max_response_length=2048"
    "+actor_rollout_ref.rollout.max_model_len=2048"
    "+actor_rollout_ref.rollout.validation_max_length=2048"
    "+actor_rollout_ref.rollout.validation_micro_batch_size=1"
    "actor_rollout_ref.rollout.name=hf"
    "actor_rollout_ref.rollout.n=1"
    "actor_rollout_ref.rollout.temperature=1.0"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"

    # --- Logging & Saving ---
    "trainer.logger=[console]"
    "trainer.project_name=azr"
    "trainer.experiment_name=3b_coder_training_optimized"
    "trainer.n_gpus_per_node=1"
    "trainer.nnodes=1"
    "trainer.save_freq=2"
    "trainer.remove_previous_ckpt_in_save=True"
    "trainer.del_local_ckpt_after_load=True"
    "trainer.test_freq=20000"
    "+trainer.val_before_train=False"
    "trainer.val_generations_to_log_to_wandb=0"
    "trainer.critic_warmup=0"

    # --- AZR & Reward Function Configuration ---
    "azr.problem_types=[code_i,code_o,code_f]"
    "reward_fn.extraction_type=answer_conditional"
    "reward_fn.math_metric=math_verify"
    "azr.data_selection_strategy.update_iteration=1"
    "azr.seed_dataset=${OUTPUT_SEED_PATH}"
    "azr.output_seed_path=${OUTPUT_SEED_PATH}"
    "azr.error_seed_dataset=${OUTPUT_ERROR_SEED_PATH}"
    "azr.output_error_seed_path=${OUTPUT_ERROR_SEED_PATH}"
    "azr.code_f_seed_dataset=${OUTPUT_CODE_F_SEED_PATH}"
    "azr.output_code_f_seed_path=${OUTPUT_CODE_F_SEED_PATH}"
    "azr.ast_check=True"
    "azr.reward.n_samples=12"
)

## --------------------------------------------------------------------------
## EXECUTION
## --------------------------------------------------------------------------
python -m absolute_zero_reasoner.main_azr_ppo "${HYDRA_OVERRIDES[@]}" "$@"