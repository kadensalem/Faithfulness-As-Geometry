#!/bin/bash
#SBATCH --job-name bool-gen
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --qos=<YOUR_QOS>
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1:00:00
#SBATCH --mem=64GB
#SBATCH --requeue
#SBATCH -o logs/bool-gen-%j.out
#SBATCH -e logs/bool-gen-%j.err

# ─── Phase 1: Generate boolean CoTs using LLaMA-3-8B-Instruct ───
#
# This job prompts the model to produce structured boolean logic CoTs.
# In ground_truth mode (no GPU needed) the model is not loaded; switch
# --mode to "model" to actually run inference on the GPU.

date
mkdir -p logs

# ── Activate conda environment ──
# Uncomment and fill in your paths:
# source ~/miniconda3/etc/profile.d/conda.sh   # or wherever conda.sh lives
# conda activate fur

# ── HuggingFace cache (point to fast local storage on CHPC) ──
# CHPC home dirs are small; use scratch or group space for the cache
export HF_HOME="/scratch/general/vast/<YOUR_UID>/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# module load cuda/12.1
nvidia-smi

# ── Configuration ──
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
MODE="model"           # "ground_truth" for CPU-only, "model" for GPU inference
NUM_STRUCTURES=10      # number of distinct boolean tree topologies
INSTANCES_PER=5        # CoT instances per topology
DEPTHS="2,3,4"         # tree depths to sample from
SEED=42

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo "Mode: $MODE"
echo "Model: $MODEL_NAME"
echo "Structures: $NUM_STRUCTURES, Instances/struct: $INSTANCES_PER"
echo "Depths: $DEPTHS"
echo "Seed: $SEED"

python generate_boolean_cots.py \
  --mode "$MODE" \
  --hf_model "$MODEL_NAME" \
  --num_structures "$NUM_STRUCTURES" \
  --instances_per "$INSTANCES_PER" \
  --depths "$DEPTHS" \
  --seed "$SEED" \
  --device cuda:0 \
  --system_prompt_file boolean_task \
  --output_dir data

echo "Phase 1 complete."
date
