#!/bin/bash
#SBATCH --job-name fur-sweep4
#SBATCH --account=cs6966
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-class-grn
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --time=4:30:00
#SBATCH --mem=80GB
#SBATCH --requeue
#SBATCH -o logs/fur-sweep4-%j.out
#SBATCH -e logs/fur-sweep4-%j.err

# ─── Hyperparameter sweep: sweep_4 only ──────────────────────────────────────
#
# sweep_4: very gentle unlearning — slow trajectory hypothesis.
# Low β means a softer forget gradient; high kl_coeff and more epochs
# allow the model to drift slowly while strongly preserving retain knowledge.
#
# Condition  β      kl_coeff  epochs  output
# sweep_4    0.02   3.0       8       data/pilot_sweep_4.jsonl
#
# Wall-time estimate:
#   ~30 min per (question x step) at 8 epochs with prefix-forced eval
#   7 questions: 7 x 30 = 210 min (~3.5 hrs) + buffer = 4.5 hrs
#
# Submit alongside submit_fur_sweep.sh. Run select_best_conditions.py
# only after BOTH jobs complete.

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

MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
FUR_FILE="data_tune_30/mcq_cots_fur.jsonl"
LR="5e-5"
SEED=42

# First 7 questions by index in mcq_cots_fur.jsonl
QUESTIONS="openbook_7-1132 openbook_7-976 openbook_9-655 openbook_1955 openbook_508 openbook_9-491 openbook_9-520"

echo "Questions : $QUESTIONS"
echo "Model     : $MODEL_NAME"

# ── sweep_4: β=0.02, kl=3.0, epochs=8 ────────────────────────────────────────
echo ""
echo "=== sweep_4: beta=0.02, kl_coeff=3.0, epochs=8 ==="
python run_fur_pilot.py \
    --model_name "$MODEL_NAME" \
    --fur_file "$FUR_FILE" \
    --question_ids $QUESTIONS \
    --step_ids 0 \
    --output_file "data/pilot_sweep_4.jsonl" \
    --epochs 8 \
    --lr "$LR" \
    --beta 0.02 \
    --kl_coeff 3.0 \
    --seed "$SEED" \
    --ff2 \
    --pos

echo ""
echo "=== sweep_4 complete. ==="
echo "=== Run select_best_conditions.py after submit_fur_sweep.sh also finishes. ==="
date
