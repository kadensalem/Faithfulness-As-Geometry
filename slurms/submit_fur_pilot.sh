#!/bin/bash
#SBATCH --job-name fur-pilot
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --qos=<YOUR_QOS>
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=6:00:00
#SBATCH --mem=100GB
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
source ~/software/pkg/miniforge3/etc/profile.d/conda.sh
conda activate fur

# ── HuggingFace cache ──
export HF_HOME="/scratch/general/vast/${USER}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

nvidia-smi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
FUR_FILE="data_tune_30/mcq_cots_fur.jsonl"
OUTPUT_FILE="data/pilot_fur_results.jsonl"
EPOCHS=5
LR="1e-5"
SEED=42

echo "Project root : $PROJECT_ROOT"
echo "Model        : $MODEL_NAME"
echo "Data         : $FUR_FILE"
echo "Output       : $OUTPUT_FILE"
echo "Epochs/LR    : $EPOCHS / $LR"

python run_fur_pilot.py \
    --model_name "$MODEL_NAME" \
    --fur_file "$FUR_FILE" \
    --question_ids openbook_1955 openbook_508 openbook_9-491 \
    --output_file "$OUTPUT_FILE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --seed "$SEED" \
    --ff2 \
    --pos

echo "Pilot FUR run complete."
date
