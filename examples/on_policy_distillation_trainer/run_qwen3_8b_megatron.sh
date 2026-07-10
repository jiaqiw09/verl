#!/usr/bin/env bash
# On-policy distillation | text | vLLM rollout | Megatron training | NVIDIA GPUs

set -euo pipefail

# ---- user-adjustable ----
STUDENT_MODEL=${STUDENT_MODEL:-/data/wjq/qwen3.5-2b}
TEACHER_MODEL=${TEACHER_MODEL:-/data/wjq/qwen3.5-2b}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-2}

distillation_loss_mode=${DISTILLATION_LOSS_MODE:-k1}
use_policy_gradient=${USE_POLICY_GRADIENT:-True}
distillation_topk=${DISTILLATION_TOPK:-64}

train_batch_size=${TRAIN_BATCH_SIZE:-16}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-1024}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

actor_lr=${ACTOR_LR:-1e-6}

actor_tp=${ACTOR_TP:-1}
actor_pp=${ACTOR_PP:-1}

rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.3}
teacher_tp=${TEACHER_TP:-1}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.3}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-200}
test_freq=${TEST_FREQ:-5}

project_name=${PROJECT_NAME:-verl_distill_gsm8k_math}
experiment_name=${EXPERIMENT_NAME:-qwen3_8b_from_qwen3_32b_vllm_megatron}
# ---- end user-adjustable ----

gsm8k_train=/data/wjq/verl/data/gsm8k/train.parquet
gsm8k_test=/data/wjq/verl/data/gsm8k/test.parquet

train_files="['$gsm8k_train']"
val_files="['$gsm8k_test']"

max_num_tokens=$(( max_prompt_length + max_response_length + 1))
########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    +model.use_remove_padding=True
)

ACTOR=(
    actor_rollout_ref.hybrid_engine=True
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False

    actor_rollout_ref.actor.megatron.use_remove_padding=True
    actor_rollout_ref.ref.megatron.use_remove_padding=True
    # context_parallel_size=2 to trigger mindspeed GDN patch (mindspeed.core.ssm.gated_delta_net)
    # GDNFeature.register_patches only registers the thd-capable GDN when cp_size>1 / eod_pack / naive_l2norm.
    # Without this, Megatron-LM's vanilla GDN is loaded, which raises NotImplementedError on packed_seq_params.
    actor_rollout_ref.actor.megatron.context_parallel_size=2
    actor_rollout_ref.ref.megatron.context_parallel_size=2
    actor_rollout_ref.actor.megatron.sequence_parallel=False
    actor_rollout_ref.ref.megatron.sequence_parallel=False
    ++actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_algo=megatron_cp_algo
    ++actor_rollout_ref.ref.megatron.override_transformer_config.context_parallel_algo=megatron_cp_algo
    ++actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size=2
    ++actor_rollout_ref.ref.megatron.override_transformer_config.context_parallel_size=2
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=flash
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}

    actor_rollout_ref.rollout.prompt_length=${max_prompt_length}
    actor_rollout_ref.rollout.response_length=${max_response_length}

    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.rollout.enforce_eager=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

EXTRA=(
    model_engine=megatron
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${NNODES}
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp}
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    distillation.teacher_models.teacher_model.inference.max_model_len=${max_num_tokens}
    distillation.teacher_models.teacher_model.inference.enforce_eager=True
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
