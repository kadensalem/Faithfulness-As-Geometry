#!/bin/bash
#SBATCH --job-name fur-patch-conc
#SBATCH --account=cs6966
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-class-grn
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --time=1:00:00
#SBATCH --mem=80GB
#SBATCH --requeue
#SBATCH -o logs/fur-patch-conc-%j.out
#SBATCH -e logs/fur-patch-conc-%j.err

# ─── Patch missing conclusion token embeddings in fur_anchor_embeddings_30.pkl ─
#
# Loads LLaMA-3-8B, runs a forward pass for each record whose CoT contains an
# S/R token after '* Conclusion:', and fills in the missing conclusion_embeddings
# at all 5 extracted layers (L4, L8, L14, L20, L28).
#
# After this runs, re-run submit_geometry_analysis.sh so that:
#   Addition 3 (conclusion probe by mode) can join subblock ff_soft labels
#   with the newly populated conclusion_D embeddings.
#
# Wall-time estimate:
#   Model load: ~3 min
#   Per record: 1 forward pass on ~500-1000 token CoT → ~20-40 s on RTX6000
#   92 records → ~60 min + buffer
#
# Prerequisite:
#   data/fur_anchor_embeddings_30.pkl must exist (from submit_extract_embeddings_30.sh)

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

python patch_conclusion_embeddings.py \
    --pkl      data/fur_anchor_embeddings_30.pkl \
    --hf_model meta-llama/Meta-Llama-3-8B-Instruct \
    --dtype    bf16 \
    --dry_run  false

echo ""
echo "Conclusion embedding patch complete."
date
