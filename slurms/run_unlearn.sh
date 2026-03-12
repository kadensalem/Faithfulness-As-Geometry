#!/bin/bash
#SBATCH --job-name bool-fur
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --qos=<YOUR_QOS>
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=6:00:00
#SBATCH --mem=100GB
#SBATCH --requeue
#SBATCH -o logs/bool-fur-%j.out
#SBATCH -e logs/bool-fur-%j.err

# ─── Phase 2: NPO Unlearning per boolean node step ───
#
# This is the heaviest job. For each instance in the dataset and each
# node step within its CoT, a fresh copy of the model is loaded,
# the step is unlearned via NPO, and the model's final answer is
# re-evaluated. The script is resumable: it skips already-processed
# (instance, step) pairs found in the output file.
#
# Time estimate: ~5-8 min per (instance x step) depending on CoT
# length. For 50 instances x ~6 steps = 300 unlearning runs.
# Budget 4-6 hours on a single A100.

date
mkdir -p logs

# ── Activate conda environment ──
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate fur

# ── HuggingFace cache ──
export HF_HOME="/scratch/general/vast/<YOUR_UID>/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# module load cuda/12.1
nvidia-smi

# ── Configuration ──
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
LR="1e-5"
EPOCHS=5
SEED=42
MAX_INSTANCES=50
METHOD="npo_KL"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_FILE="data/boolean_cots_fur.jsonl"
OUTPUT_FILE="results/boolean_faithfulness.jsonl"
mkdir -p results

echo "Project root: $PROJECT_ROOT"
echo "Model: $MODEL_NAME"
echo "Data: $DATA_FILE"
echo "LR: $LR, Epochs: $EPOCHS, Method: $METHOD"
echo "Max instances: $MAX_INSTANCES"
echo "Output: $OUTPUT_FILE"

python boolean_unlearn.py \
  --model_name "$MODEL_NAME" \
  --data_file "$DATA_FILE" \
  --method "$METHOD" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --seed "$SEED" \
  --max_instances "$MAX_INSTANCES" \
  --output_file "$OUTPUT_FILE" \
  --system_prompt_file boolean_task

echo "Phase 2 complete."
date
