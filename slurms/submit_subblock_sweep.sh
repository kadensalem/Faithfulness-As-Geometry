#!/bin/bash
#SBATCH --job-name fur-subblock
#SBATCH --account=cs6966
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-class-grn
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --array=0-119
#SBATCH --time=1:30:00
#SBATCH --mem=12GB
#SBATCH --requeue
#SBATCH -o logs/fur-subblock-%A_%a.out
#SBATCH -e logs/fur-subblock-%A_%a.err

# ─── Sub-block unlearning mode sweep ──────────────────────────────────────────
#
# Tests 4 unlearning granularities on step_idx=0 of each question using the
# per-question best hyperparameters from data/best_conditions_30.json.
#
# Modes:
#   whole_block         existing behaviour — unlearn the entire Answer block
#   premise_only        unlearn only the * Premise: ... line
#   reasoning_only      unlearn only the * Reasoning: ... line
#   premise_and_reasoning   unlearn Premise + Reasoning together
#
# Array layout (120 tasks = 30 questions × 4 modes):
#   TASK_ID = question_index * 4 + mode_index
#   mode_index: 0=whole_block  1=premise_only  2=reasoning_only  3=premise_and_reasoning
#
# Hyperparameter fallback (for questions with condition=None in best_conditions_30.json):
#   Default to cond_A: β=0.1, kl=1.0, epochs=3
#
# Output per task:
#   data/subblock_{mode}_q{question_index}.jsonl
#
# After all tasks complete, merge with:
#   cat data/subblock_whole_block_q*.jsonl > data/subblock_cond_whole_block.jsonl
#   (etc. for each mode)
#
# Wall-time estimate per task:
#   Comparable to whole_block sweep (~30 min); budget 1.5 hrs with --requeue.

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
BEST_CONDS="data/best_conditions_30.json"
LR="5e-5"
SEED=42

# ── Decode array task ID ──────────────────────────────────────────────────────
TASK_ID=${SLURM_ARRAY_TASK_ID}
Q_IDX=$(( TASK_ID / 4 ))
MODE_IDX=$(( TASK_ID % 4 ))

MODES=("whole_block" "premise_only" "reasoning_only" "premise_and_reasoning")
UNLEARN_MODE="${MODES[$MODE_IDX]}"

QID=$(sed -n "$((Q_IDX + 1))p" "$QIDS_FILE")
echo "Array task ${TASK_ID}: question_idx=${Q_IDX}  qid=${QID}  mode=${UNLEARN_MODE}"

# ── Look up best hyperparameters from best_conditions_30.json ─────────────────
# Key format: "{qid}_step0"
# Condition → (beta, kl_coeff, epochs):
#   sweep30_cond1: β=0.10, kl=2.0, ep=3
#   sweep30_cond2: β=0.05, kl=3.0, ep=5
#   sweep30_cond3: β=0.02, kl=3.0, ep=8
#   sweep30_cond4: β=0.10, kl=1.0, ep=3  (fallback default)

COND=$(python3 - <<PYEOF
import json, sys
data = json.load(open("${BEST_CONDS}"))
key = "${QID}_step0"
entry = data.get(key, {})
cond = entry.get("condition") or "sweep30_cond4"
print(cond)
PYEOF
)

case "$COND" in
    sweep30_cond1) BETA=0.10; KL=2.0; EPOCHS=3 ;;
    sweep30_cond2) BETA=0.05; KL=3.0; EPOCHS=5 ;;
    sweep30_cond3) BETA=0.02; KL=3.0; EPOCHS=8 ;;
    *)             BETA=0.10; KL=1.0; EPOCHS=3 ;;  # sweep30_cond4 / fallback
esac

echo "Condition: ${COND}  beta=${BETA}  kl=${KL}  epochs=${EPOCHS}"

OUTPUT_FILE="data/subblock_${UNLEARN_MODE}_q${Q_IDX}.jsonl"

python run_fur_pilot.py \
    --model_name  "$MODEL_NAME" \
    --fur_file    "$FUR_FILE" \
    --question_ids "$QID" \
    --step_ids    0 \
    --output_file "$OUTPUT_FILE" \
    --epochs      "$EPOCHS" \
    --lr          "$LR" \
    --beta        "$BETA" \
    --kl_coeff    "$KL" \
    --seed        "$SEED" \
    --ff2 \
    --pos \
    --unlearn_mode "$UNLEARN_MODE"

echo ""
echo "=== Task ${TASK_ID} (${QID} / ${UNLEARN_MODE}) complete ==="
date
