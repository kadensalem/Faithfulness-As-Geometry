#!/bin/bash
#SBATCH --job-name fur-extract-emb
#SBATCH --account=cs6966
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-class-grn
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --time=3:00:00
#SBATCH --mem=80GB
#SBATCH --requeue
#SBATCH -o logs/fur-extract-emb-%j.out
#SBATCH -e logs/fur-extract-emb-%j.err

# ─── Phase 3: FUR anchor embedding extraction ─────────────────────────────────
#
# Prerequisites:
#   1. submit_fur_sweep.sh  AND submit_fur_sweep4.sh  have completed
#   2. select_best_conditions.py has been run to produce data/best_conditions.json
#
# For each (question, condition, epoch) in best_conditions.json this job:
#   - Re-runs NPO+KL unlearning with the same hyperparameters and seed
#   - Stops at the selected epoch and extracts 9 anchor embeddings
#   - Pairs them with the FF-SOFT score
#
# Wall-time estimate:
#   Each question requires:
#     * 1 base-model forward pass to compute probs (upfront, once)
#     * 2 model loads (model + oracle) + N epoch training loop
#     * 1 embedding forward pass
#   At epoch 1: ~5-8 min/question; at epoch 3-4: ~15-20 min/question
#   Budget 3 hrs for up to 6 questions.

date
mkdir -p logs

source /uufs/chpc.utah.edu/common/home/u1427573/software/pkg/miniforge3/etc/profile.d/conda.sh
conda activate fur-sm120

export HF_HOME="/scratch/general/vast/${USER}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

nvidia-smi

PROJECT_ROOT="/uufs/chpc.utah.edu/common/home/u1427573/Faithfulness-As-Geometry"
cd "$PROJECT_ROOT"

python extract_fur_embeddings.py \
    --best_conditions data/best_conditions.json \
    --fur_file        data_tune_30/mcq_cots_fur.jsonl \
    --hf_model        meta-llama/Meta-Llama-3-8B-Instruct \
    --middle_layer    14 \
    --output_file     data/fur_anchor_embeddings.pkl \
    --sanity_check_dir data/pca_sanity_fur \
    --dtype           bf16

echo ""
echo "Embedding extraction complete."
date
