#!/bin/bash
#SBATCH --job-name=qwen14b-study
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=logs/qwen14b-%j-%x.out
#SBATCH --error=logs/qwen14b-%j-%x.err

# =============================================================================
# Qwen2.5-14B Training-Generation Mismatch Study
#
# Dense model, single-node, TP=2.
# Modes:
#   vllm             — vLLM inference baseline
#   m-inf            — Megatron-Inference (no FA3)
#   m-inf-fa3        — Megatron-Inference + Flash Attention 3
#   batch-invariant  — M-Inf + FA3 + batch_invariant_mode
#
# Usage:
#   sbatch --export=MODE=m-inf-fa3 run_qwen14b_study.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via --export or environment)
# ---------------------------------------------------------------------------
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
RL_DIR="/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/jinzex/post-training/RL"
CONTAINER_IMAGE="/lustre/fsw/portfolios/coreai/users/jinzex/post-training/containers/nemo_rl_v0.5.0.sqsh"

MODE="${MODE:?ERROR: MODE must be set. Use: vllm, m-inf, m-inf-fa3, or batch-invariant}"
MAX_STEPS="${MAX_STEPS:-50}"
GPUS_PER_NODE=8
NUM_NODES="${SLURM_NNODES:-1}"

if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    set -a; source "${SCRIPT_DIR}/.env"; set +a
fi

export TORCH_CUDA_ARCH_LIST='9.0 10.0'
: "${HF_TOKEN:?Set HF_TOKEN in .env}"
: "${HF_HOME:?Set HF_HOME in .env}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in .env}"
: "${WANDB_ENTITY:?Set WANDB_ENTITY in .env}"
: "${WANDB_PROJECT:?Set WANDB_PROJECT in .env}"

if [[ ! "$MODE" =~ ^(vllm|m-inf|m-inf-fa3|batch-invariant)$ ]]; then
    echo "ERROR: Invalid MODE '$MODE'. Must be one of: vllm, m-inf, m-inf-fa3, batch-invariant"
    exit 1
fi

echo "=============================================="
echo "Qwen2.5-14B  gen_kl_error study"
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
    --config examples/configs/grpo_math_1B_megatron.yaml \
    policy.model_name=Qwen/Qwen2.5-14B \
    grpo.max_num_steps=${MAX_STEPS} \
    cluster.num_nodes=${NUM_NODES} cluster.gpus_per_node=${GPUS_PER_NODE} \
    grpo.num_prompts_per_step=16 grpo.num_generations_per_prompt=4 \
    policy.train_global_batch_size=16 \
    policy.megatron_cfg.tensor_model_parallel_size=2 \
    policy.train_micro_batch_size=2 policy.logprob_batch_size=2 \
    grpo.val_at_start=false grpo.val_period=9999 \
    logger.wandb_enabled=true \
    logger.wandb.project=${WANDB_PROJECT}"

MINF_FLAGS="\
    policy.generation.backend=megatron \
    policy.megatron_cfg.cuda_graph_impl=none \
    policy.megatron_cfg.cuda_graph_scope=full_iteration_inference \
    policy.megatron_cfg.use_te_rng_tracker=true \
    policy.megatron_cfg.inference_rng_tracker=true"

case "$MODE" in
    vllm)
        CMD="${BASE_CMD} \
            policy.generation.backend=vllm \
            policy.generation.vllm_cfg.tensor_parallel_size=2 \
            logger.wandb.name=qwen-14b-vllm"
        ;;
    m-inf)
        CMD="${BASE_CMD} ${MINF_FLAGS} \
            logger.wandb.name=qwen-14b-m-inf"
        ;;
    m-inf-fa3)
        export NRL_INSTALL_FA3=1
        CMD="${BASE_CMD} ${MINF_FLAGS} \
            logger.wandb.name=qwen-14b-m-inf-fa3"
        ;;
    batch-invariant)
        export NRL_INSTALL_FA3=1
        CMD="${BASE_CMD} ${MINF_FLAGS} \
            policy.megatron_cfg.batch_invariant_mode=true \
            +policy.megatron_cfg.attention_backend=flash \
            logger.wandb.name=qwen-14b-m-inf-batch-invariant"
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
    export NRL_INSTALL_FA3=${NRL_INSTALL_FA3:-0} && \
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
