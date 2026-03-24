#!/bin/bash
#SBATCH --job-name geometry-analysis
#SBATCH --account=cs6966
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-class-grn
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --time=0:20:00
#SBATCH --mem=12GB
#SBATCH -o logs/geometry-analysis-%j.out
#SBATCH -e logs/geometry-analysis-%j.err

# Runs all hypothesis tests from geometry_compare.ipynb as a batch job.
# No GPU is actually used — the partition is specified only to avoid login-node
# resource limits. Output plots saved to data/geometry_analysis_plots/.
#
# Run after extract_fur_embeddings.py has produced:
#   data/fur_anchor_embeddings.pkl      (pilot 6q)
#   data/fur_anchor_embeddings_30.pkl   (30q, if available)
#   data/best_conditions_30.json        (30q, if available)

date
mkdir -p logs data/geometry_analysis_plots

source /uufs/chpc.utah.edu/common/home/u1427573/software/pkg/miniforge3/etc/profile.d/conda.sh
conda activate fur-sm120

PROJECT_ROOT="/uufs/chpc.utah.edu/common/home/u1427573/Faithfulness-As-Geometry"
cd "$PROJECT_ROOT"

python run_geometry_analysis.py \
    --pkl           data/fur_anchor_embeddings.pkl \
    --pkl30         data/fur_anchor_embeddings_30.pkl \
    --bc30          data/best_conditions_30.json \
    --freeform_pkl  data/freeform_embeddings.pkl \
    --outdir        data/geometry_analysis_plots

echo ""
echo "Geometry analysis complete."
date
