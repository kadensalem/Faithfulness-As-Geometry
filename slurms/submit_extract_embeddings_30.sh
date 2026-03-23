#!/bin/bash
#SBATCH --job-name fur-extract-emb-30
#SBATCH --account=cs6966
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-class-grn
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --time=12:00:00
#SBATCH --mem=80GB
#SBATCH --requeue
#SBATCH -o logs/fur-extract-emb-30-%j.out
#SBATCH -e logs/fur-extract-emb-30-%j.err

# ─── Phase 3 (30-question): FUR anchor + conclusion embedding extraction ───────
#
# Prerequisites:
#   1. submit_fur_sweep_30.sh AND submit_fur_sweep_30b.sh array jobs complete
#   2. merge_sweep30.sh has been run
#   3. select_best_conditions.py has produced data/best_conditions_30.json
#
# Wall-time estimate:
#   1 base-model prob pass (all 30 records, upfront): ~5 min
#   Per question: 2 model loads + N epoch training + 1 embedding pass
#   ~20 min/question average (across conditions and epochs)
#   ~30 questions × 20 min = 600 min = 10 hrs + buffer → 12 hrs
#   --requeue handles any timeout.

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
    --best_conditions  data/best_conditions_30.json \
    --fur_file         data_tune_30/mcq_cots_fur.jsonl \
    --hf_model         meta-llama/Meta-Llama-3-8B-Instruct \
    --middle_layer     14 \
    --output_file      data/fur_anchor_embeddings_30.pkl \
    --sanity_check_dir data/pca_sanity_fur_30 \
    --dtype            bf16

echo ""
echo "30-question embedding extraction complete."
date
