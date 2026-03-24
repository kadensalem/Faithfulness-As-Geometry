#!/bin/bash
#SBATCH --job-name fur-freeform-emb
#SBATCH --account=cs6966
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-class-grn
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --time=4:00:00
#SBATCH --mem=80GB
#SBATCH --requeue
#SBATCH -o logs/fur-freeform-emb-%j.out
#SBATCH -e logs/fur-freeform-emb-%j.err

# ─── Free-form CoT geometry extraction ────────────────────────────────────────
#
# Loads Meta-Llama-3-8B-Instruct once, runs a single forward pass per question
# (30 questions), extracts hidden states at L8, L14, L28, and saves geometry
# summary metrics to data/freeform_embeddings.pkl.
#
# Prerequisites:
#   1. submit_fur_sweep_30.sh / submit_fur_sweep_30b.sh array jobs complete
#   2. merge_sweep30.sh has been run
#   3. select_best_conditions.py has produced data/best_conditions_30.json
#
# Wall-time estimate:
#   Model load: ~3 min
#   Per question: 1 forward pass on ~200-800 token CoT → ~10-30 s on RTX6000
#   30 questions × ~1 min (generous) = ~30 min + buffer → 4 hrs

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

python extract_freeform_embeddings.py \
    --hf_model        meta-llama/Meta-Llama-3-8B-Instruct \
    --best_conditions data/best_conditions_30.json \
    --data_dir        data \
    --output_file     data/freeform_embeddings.pkl \
    --layers          8 14 28 \
    --dtype           bf16

echo ""
echo "Free-form embedding extraction complete."
date
