#!/bin/bash
#SBATCH --job-name fur-conc-colon
#SBATCH --account=cs6966
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-class-grn
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --time=2:00:00
#SBATCH --mem=80GB
#SBATCH --requeue
#SBATCH -o logs/fur-conc-colon-%j.out
#SBATCH -e logs/fur-conc-colon-%j.err

# ─── Extract colon-token conclusion embeddings ────────────────────────────────
#
# Runs extract_conclusion_colon.py in full mode.
#
# Outputs:
#   data/fur_anchor_embeddings_30.pkl  — updated with conclusion_colon_L8/L14/L28
#   data/subblock_conclusion_colon.pkl — colon embeddings for 15 subblock questions
#
# After this runs, the notebook cells for Change 3 (Test 3 with colon proxy)
# can be executed.  No re-run of the sweep or unlearning is needed.
#
# Wall-time estimate:
#   Model load: ~3 min
#   Per record: 1 forward pass on ~500-1000 token CoT -> ~20-40 s on RTX6000
#   37 main records + 15 subblock = 52 total -> ~35-60 min + buffer
#
# Prerequisite:
#   data/fur_anchor_embeddings_30.pkl must exist
#   data/subblock_*_q*.jsonl files must exist in data/

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

python extract_conclusion_colon.py \
    --pkl          data/fur_anchor_embeddings_30.pkl \
    --subblock_dir data \
    --subblock_pkl data/subblock_conclusion_colon.pkl \
    --hf_model     meta-llama/Meta-Llama-3-8B-Instruct \
    --dtype        bf16 \
    --dry_run      false

echo ""
echo "Conclusion colon extraction complete."
date
