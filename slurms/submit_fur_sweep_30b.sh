#!/bin/bash
#SBATCH --job-name fur-sweep30-cd
#SBATCH --account=cs6966
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-class-grn
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --array=0-29
#SBATCH --time=7:00:00
#SBATCH --mem=80GB
#SBATCH --requeue
#SBATCH -o logs/fur-sweep30-cd-%A_%a.out
#SBATCH -e logs/fur-sweep30-cd-%A_%a.err

# ─── 30-question sweep: cond2 (β=0.05, kl=3.0, ep=5) + cond3 (β=0.02, kl=3.0, ep=8)
#
# These are the heavier conditions — split into separate job array from cond1/cond4
# due to longer wall-time requirements.
#
# Condition      β      kl_coeff  epochs  output (per-question parts, merged later)
# sweep30_cond2  0.05   3.0       5       data/sweep30_cond2_q{i}.jsonl
# sweep30_cond3  0.02   3.0       8       data/sweep30_cond3_q{i}.jsonl
#
# Wall-time estimate per task:
#   cond2: 1 question × 5 steps × ~21 min/step = ~105 min
#   cond3: 1 question × 5 steps × ~30 min/step = ~150 min
#   Total: ~255 min = 4.25 hrs per task
#   Budget 7 hrs with --requeue for safety.
#
# After ALL array tasks complete (both this job and submit_fur_sweep_30.sh),
# run merge_sweep30.sh, then select_best_conditions.py.

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

# ── sweep30_cond2: β=0.05, kl=3.0, epochs=5 ─────────────────────────────────
echo ""
echo "=== sweep30_cond2: beta=0.05, kl_coeff=3.0, epochs=5 ==="
python run_fur_pilot.py \
    --model_name  "$MODEL_NAME" \
    --fur_file    "$FUR_FILE" \
    --question_ids "$QID" \
    --step_ids    0 1 2 3 4 \
    --output_file "data/sweep30_cond2_q${TASK_ID}.jsonl" \
    --epochs      5 \
    --lr          "$LR" \
    --beta        0.05 \
    --kl_coeff    3.0 \
    --seed        "$SEED" \
    --ff2 \
    --pos

# ── sweep30_cond3: β=0.02, kl=3.0, epochs=8 ─────────────────────────────────
echo ""
echo "=== sweep30_cond3: beta=0.02, kl_coeff=3.0, epochs=8 ==="
python run_fur_pilot.py \
    --model_name  "$MODEL_NAME" \
    --fur_file    "$FUR_FILE" \
    --question_ids "$QID" \
    --step_ids    0 1 2 3 4 \
    --output_file "data/sweep30_cond3_q${TASK_ID}.jsonl" \
    --epochs      8 \
    --lr          "$LR" \
    --beta        0.02 \
    --kl_coeff    3.0 \
    --seed        "$SEED" \
    --ff2 \
    --pos

echo ""
echo "=== Task ${TASK_ID} (${QID}) cond2+cond3 complete ==="
date
