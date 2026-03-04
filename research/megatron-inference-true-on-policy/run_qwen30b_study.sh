#!/bin/bash
#SBATCH --job-name=qwen30b-study
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch_short
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=logs/qwen30b-%j-%x.out
#SBATCH --error=logs/qwen30b-%j-%x.err

# =============================================================================
# Qwen3-30B-A3B Training-Generation Mismatch Study
#
# MoE model (30B total, 3B active), 2 nodes, TP=4, EP=16.
# Modes:
#   vllm             — vLLM inference baseline
#   m-inf            — Megatron-Inference default
#   batch-invariant  — M-Inf + batch_invariant_mode
#
# Usage:
#   sbatch --export=MODE=m-inf run_qwen30b_study.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via --export or environment)
# ---------------------------------------------------------------------------
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"

MODE="${MODE:?ERROR: MODE must be set. Use: vllm, m-inf, or batch-invariant}"
MAX_STEPS="${MAX_STEPS:-50}"
GPUS_PER_NODE=8
NUM_NODES="${SLURM_NNODES:-2}"

# Source .env for config and secrets.
# Copy .env.template to .env and fill in your values.
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    set -a; source "${SCRIPT_DIR}/.env"; set +a
fi

export TORCH_CUDA_ARCH_LIST='9.0 10.0'
: "${RL_DIR:?Set RL_DIR in .env}"
: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE in .env}"
: "${HF_TOKEN:?Set HF_TOKEN in .env}"
: "${HF_HOME:?Set HF_HOME in .env}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in .env}"
: "${WANDB_ENTITY:?Set WANDB_ENTITY in .env}"
: "${WANDB_PROJECT:?Set WANDB_PROJECT in .env}"

if [[ ! "$MODE" =~ ^(vllm|m-inf|batch-invariant)$ ]]; then
    echo "ERROR: Invalid MODE '$MODE'. Must be one of: vllm, m-inf, batch-invariant"
    exit 1
fi

echo "=============================================="
echo "Qwen3-30B-A3B  gen_kl_error study"
echo "  Mode:          ${MODE}"
echo "  Max steps:     ${MAX_STEPS}"
echo "  Nodes:         ${NUM_NODES}"
echo "  GPUs/node:     ${GPUS_PER_NODE}"
echo "  RL dir:        ${RL_DIR}"
echo "  Container:     ${CONTAINER_IMAGE}"
echo "  Job ID:        ${SLURM_JOB_ID:-interactive}"
echo "  Time:          $(date)"
echo "=============================================="

mkdir -p "${SCRIPT_DIR}/logs"

BASE_CMD="NRL_FORCE_REBUILD_VENVS=true uv run examples/run_grpo.py \
    --config examples/configs/grpo_math_qwen30ba3b_megatron.yaml \
    grpo.max_num_steps=${MAX_STEPS} \
    cluster.num_nodes=${NUM_NODES} cluster.gpus_per_node=${GPUS_PER_NODE} \
    grpo.num_prompts_per_step=16 grpo.num_generations_per_prompt=4 \
    policy.train_global_batch_size=16 \
    policy.megatron_cfg.tensor_model_parallel_size=4 \
    policy.megatron_cfg.expert_model_parallel_size=16 \
    policy.megatron_cfg.moe_token_dispatcher_type=alltoall \
    policy.megatron_cfg.moe_router_dtype=fp32 \
    policy.logprob_batch_size=1 \
    policy.max_total_sequence_length=512 \
    logger.wandb_enabled=true \
    logger.wandb.project=${WANDB_PROJECT}"

MINF_FLAGS="\
    policy.generation.backend=megatron \
    policy.megatron_cfg.moe_pad_experts_for_cuda_graph_inference=true \
    policy.megatron_cfg.cuda_graph_impl=none \
    policy.megatron_cfg.cuda_graph_scope=full_iteration_inference \
    policy.generation.mcore_generation_config.kv_cache_management_mode=recompute \
    policy.generation.mcore_generation_config.static_kv_memory_pointers=false \
    policy.generation.mcore_generation_config.use_cuda_graphs_for_non_decode_steps=False \
    policy.generation.mcore_generation_config.num_cuda_graphs=4 \
    policy.generation.mcore_generation_config.buffer_size_gb=8"

case "$MODE" in
    vllm)
        CMD="${BASE_CMD} \
            policy.generation.backend=vllm \
            policy.generation.vllm_cfg.tensor_parallel_size=4 \
            logger.wandb.name=qwen-30b-vllm"
        ;;
    m-inf)
        CMD="${BASE_CMD} ${MINF_FLAGS} \
            logger.wandb.name=qwen-30b-m-inf"
        ;;
    batch-invariant)
        CMD="${BASE_CMD} ${MINF_FLAGS} \
            policy.megatron_cfg.batch_invariant_mode=true \
            +policy.megatron_cfg.attention_backend=flash \
            logger.wandb.name=qwen-30b-m-inf-batch-invariant"
        ;;
esac

cd "${RL_DIR}"

export CONTAINER="${CONTAINER_IMAGE}"
export MOUNTS="/lustre:/lustre,${RL_DIR}:/opt/nemo-rl"
export GPUS_PER_NODE

export COMMAND="unset NVTE_FUSED_ATTN NVTE_FLASH_ATTN NVTE_UNFUSED_ATTN && \
    export HF_HOME=${HF_HOME} && \
    export TORCH_CUDA_ARCH_LIST='${TORCH_CUDA_ARCH_LIST}' && \
    export HF_TOKEN=${HF_TOKEN} && \
    export WANDB_API_KEY=${WANDB_API_KEY} && \
    export WANDB_ENTITY=${WANDB_ENTITY} && \
    export CUDA_DEVICE_MAX_CONNECTIONS=1 && \
    cd /opt/nemo-rl && \
    ${CMD}"

echo ""
echo "COMMAND:"
echo "${CMD}"
echo ""
echo "=============================================="

source ray.sub

echo ""
echo "=============================================="
echo "Job completed at: $(date)"
echo "=============================================="
