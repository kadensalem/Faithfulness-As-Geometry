#!/bin/bash
#SBATCH --job-name fur-sweep30-ab
#SBATCH --account=cs6966
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-class-grn
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --array=0-29
#SBATCH --time=4:00:00
#SBATCH --mem=80GB
#SBATCH --requeue
#SBATCH -o logs/fur-sweep30-ab-%A_%a.out
#SBATCH -e logs/fur-sweep30-ab-%A_%a.err

# ─── 30-question sweep: cond1 (β=0.1, kl=2.0, ep=3) + cond4 (β=0.1, kl=1.0, ep=3)
#
# Array task i processes question i (0-indexed) for BOTH conditions sequentially.
#
# Condition   β      kl_coeff  epochs  output (per-question parts, merged later)
# sweep30_cond1  0.10   2.0       3       data/sweep30_cond1_q{i}.jsonl
# sweep30_cond4  0.10   1.0       3       data/sweep30_cond4_q{i}.jsonl
#
# Wall-time estimate per task:
#   1 question × 5 steps × ~15 min/step × 2 conditions = ~150 min = 2.5 hrs
#   Budget 4 hrs per task with --requeue for safety.
#
# After ALL array tasks complete, run merge_sweep30.sh before
# running select_best_conditions.py.

date
mkdir -p logs data

source /uufs/chpc.utah.edu/common/home/u1427573/software/pkg/miniforge3/etc/profile.d/conda.sh
conda activate fur-sm120

export HF_HOME="/scratch/general/vast/${USER}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

nvidia-smi

PROJECT_ROOT="/uufs/chpc.utah.edu/common/home/u1427573/Faithfulness-As-Geometry"
cd "$PROJECT_ROOT"

MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
FUR_FILE="data_tune_30/mcq_cots_fur.jsonl"
QIDS_FILE="data_tune_30/qids.txt"
LR="5e-5"
SEED=42

# Pick this task's question ID (0-indexed)
TASK_ID=${SLURM_ARRAY_TASK_ID}
QID=$(sed -n "$((TASK_ID + 1))p" "$QIDS_FILE")
echo "Array task ${TASK_ID}: question = ${QID}"

# ── sweep30_cond1: β=0.10, kl=2.0, epochs=3 ─────────────────────────────────
echo ""
echo "=== sweep30_cond1: beta=0.1, kl_coeff=2.0, epochs=3 ==="
python run_fur_pilot.py \
    --model_name  "$MODEL_NAME" \
    --fur_file    "$FUR_FILE" \
    --question_ids "$QID" \
    --step_ids    0 1 2 3 4 \
    --output_file "data/sweep30_cond1_q${TASK_ID}.jsonl" \
    --epochs      3 \
    --lr          "$LR" \
    --beta        0.1 \
    --kl_coeff    2.0 \
    --seed        "$SEED" \
    --ff2 \
    --pos

# ── sweep30_cond4: β=0.10, kl=1.0, epochs=3 (baseline) ──────────────────────
echo ""
echo "=== sweep30_cond4: beta=0.1, kl_coeff=1.0, epochs=3 ==="
python run_fur_pilot.py \
    --model_name  "$MODEL_NAME" \
    --fur_file    "$FUR_FILE" \
    --question_ids "$QID" \
    --step_ids    0 1 2 3 4 \
    --output_file "data/sweep30_cond4_q${TASK_ID}.jsonl" \
    --epochs      3 \
    --lr          "$LR" \
    --beta        0.1 \
    --kl_coeff    1.0 \
    --seed        "$SEED" \
    --ff2 \
    --pos

echo ""
echo "=== Task ${TASK_ID} (${QID}) cond1+cond4 complete ==="
date
