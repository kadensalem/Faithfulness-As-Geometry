#!/bin/bash
#SBATCH --job-name anchor-embed
#SBATCH --account=cs6966
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-class-grn
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --time=1:00:00
#SBATCH --mem=40GB
#SBATCH --requeue
#SBATCH -o logs/anchor-embed-%j.out
#SBATCH -e logs/anchor-embed-%j.err

# ─── Extract 9-anchor trajectory embeddings for all 30 MCQ questions ───
#
# Loads Meta-Llama-3-8B-Instruct once, runs one forward pass per question
# with output_hidden_states=True, and extracts hidden states at 9 structural
# anchor token positions (Answer_A → ... → Final_Answer) at middle layer 14.
#
# Also extracts:
#   - Conclusion probe embeddings (S/R token per answer block)
#   - Premise / Reasoning span mean-pooled embeddings
#   - Displacement vectors (premise_delta, reasoning_delta, final_answer_delta)
#   - PCA sanity-check CSVs saved to data/pca_sanity/
#
# Output: data/pilot_trajectories.pkl
# Expected wall time: ~15 min for 30 questions (model load + 30 forward passes)

date
mkdir -p logs

# ── Conda environment ──
source /uufs/chpc.utah.edu/common/home/u1427573/software/pkg/miniforge3/etc/profile.d/conda.sh
conda activate fur-sm120

# ── HuggingFace cache ──
export HF_HOME="/scratch/general/vast/${USER}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

nvidia-smi

PROJECT_ROOT="/uufs/chpc.utah.edu/common/home/u1427573/Faithfulness-As-Geometry"
cd "$PROJECT_ROOT"

MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
FUR_FILE="data_tune_30/mcq_cots_fur.jsonl"
OUTPUT_FILE="data/pilot_trajectories.pkl"
SANITY_DIR="data/pca_sanity"
MIDDLE_LAYER=14
N_QUESTIONS=30

echo "Project root  : $PROJECT_ROOT"
echo "Model         : $MODEL_NAME"
echo "FUR file      : $FUR_FILE"
echo "Output        : $OUTPUT_FILE"
echo "Middle layer  : $MIDDLE_LAYER"
echo "N questions   : $N_QUESTIONS"

python extract_anchor_embeddings.py \
    --hf_model "$MODEL_NAME" \
    --fur_file "$FUR_FILE" \
    --n_questions "$N_QUESTIONS" \
    --middle_layer "$MIDDLE_LAYER" \
    --output_file "$OUTPUT_FILE" \
    --sanity_check_dir "$SANITY_DIR" \
    --dtype bf16

echo "Anchor embedding extraction complete."
date
