#!/bin/bash
#SBATCH --job-name fur-pilot
#SBATCH --account=cs6966
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-class-grn
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --time=2:00:00
#SBATCH --mem=80GB
#SBATCH --requeue
#SBATCH -o logs/fur-pilot-%j.out
#SBATCH -e logs/fur-pilot-%j.err

# ─── Pilot FUR run: 3 questions x 5 steps x 5 epochs ───
#
# Runs NPO+KL unlearning on 3 pre-selected OpenBookQA questions from
# data_tune_30/mcq_cots_fur.jsonl. Each (question, step) pair loads the
# model fresh, unlearns for EPOCHS, evaluates per epoch, and appends one
# JSONL line to the output file. The job is resumable.
#
# Expected wall time: ~6-8 min per (question x step).
# 3 questions x 5 steps x ~7 min ≈ 105 min on a single A100.

date
mkdir -p logs

# ── Conda environment ──
source /uufs/chpc.utah.edu/common/home/u1427573/software/pkg/miniforge3/etc/profile.d/conda.sh
conda activate fur-sm120

# ── HuggingFace cache ──
export HF_HOME="/scratch/general/vast/${USER}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

nvidia-smi

PROJECT_ROOT="/uufs/chpc.utah.edu/common/home/u1427573/Faithfulness-As-Geometry"
cd "$PROJECT_ROOT"

MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
FUR_FILE="data_tune_30/mcq_cots_fur.jsonl"
EPOCHS=1
LR="5e-5"
SEED=42

echo "Project root : $PROJECT_ROOT"
echo "Model        : $MODEL_NAME"
echo "Data         : $FUR_FILE"
echo "Epochs/LR    : $EPOCHS / $LR"

# ── Condition A: baseline (ff2, KL_coeff=1.0) ──────────────────────────────
echo "--- Condition A: ff2, kl_coeff=1.0 ---"
python run_fur_pilot.py \
    --model_name "$MODEL_NAME" \
    --fur_file "$FUR_FILE" \
    --question_ids openbook_1955 openbook_508 openbook_9-491 \
    --step_ids 0 \
    --output_file "data/pilot_cond_A.jsonl" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --seed "$SEED" \
    --ff2 \
    --pos

# ── Condition B: ff2 + stronger retain (KL_coeff=2.0) ──────────────────────
echo "--- Condition B: ff2, kl_coeff=2.0 ---"
python run_fur_pilot.py \
    --model_name "$MODEL_NAME" \
    --fur_file "$FUR_FILE" \
    --question_ids openbook_1955 openbook_508 openbook_9-491 \
    --step_ids 0 \
    --output_file "data/pilot_cond_B.jsonl" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --kl_coeff 2.0 \
    --seed "$SEED" \
    --ff2 \
    --pos

# ── Condition C: lower beta (0.05) + stronger retain (KL_coeff=3.0) + 5 epochs
echo "--- Condition C: ff2, beta=0.05, kl_coeff=3.0, epochs=5 ---"
python run_fur_pilot.py \
    --model_name "$MODEL_NAME" \
    --fur_file "$FUR_FILE" \
    --question_ids openbook_1955 openbook_508 openbook_9-491 \
    --step_ids 0 \
    --output_file "data/pilot_cond_C.jsonl" \
    --epochs 5 \
    --lr "$LR" \
    --beta 0.05 \
    --kl_coeff 3.0 \
    --seed "$SEED" \
    --ff2 \
    --pos

echo "Pilot FUR run complete."
date
