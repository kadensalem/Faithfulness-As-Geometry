#!/bin/bash
#SBATCH --job-name mcq-gen
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --qos=<YOUR_QOS>
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
#SBATCH --mem=64GB
#SBATCH --requeue
#SBATCH -o logs/mcq-gen-%j.out
#SBATCH -e logs/mcq-gen-%j.err

# ─── Phase 1: Generate MCQ CoTs using LLaMA-3-8B-Instruct ───
#
# This job loads OpenBookQA, prompts the model with the structured
# Premise/Reasoning/Conclusion format, and saves CoT datasets for
# the unlearning and geometry pipelines.

date
mkdir -p logs

# ── Activate conda environment ──
# Uncomment and fill in your paths:
# source ~/miniconda3/etc/profile.d/conda.sh   # or wherever conda.sh lives
# conda activate fur

# ── HuggingFace cache (point to fast local storage on CHPC) ──
export HF_HOME="/scratch/general/vast/<YOUR_UID>/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# module load cuda/12.1
nvidia-smi

# ── Configuration ──
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
SPLIT="test"
MAX_INSTANCES=200
SEED=42

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo "Model: $MODEL_NAME"
echo "Split: $SPLIT"
echo "Max instances: $MAX_INSTANCES"
echo "Seed: $SEED"

python generate_mcq_cots.py \
  --hf_model "$MODEL_NAME" \
  --split "$SPLIT" \
  --max_instances "$MAX_INSTANCES" \
  --seed "$SEED" \
  --device cuda:0 \
  --system_prompt_file context/structured_reasoning_prompt.tx \
  --output_dir data

echo "Phase 1 complete."
date
