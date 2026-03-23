#!/bin/bash
#SBATCH --job-name fur-sweep
#SBATCH --account=cs6966
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-class-grn
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --time=7:00:00
#SBATCH --mem=80GB
#SBATCH --requeue
#SBATCH -o logs/fur-sweep-%j.out
#SBATCH -e logs/fur-sweep-%j.err

# ─── Hyperparameter sweep: sweep_1, sweep_2, sweep_3 ─────────────────────────
#
# 7 questions x step_idx=0 x 3 conditions
#
# Condition  β      kl_coeff  epochs  output
# sweep_1    0.10   1.0       3       data/pilot_sweep_1.jsonl
# sweep_2    0.10   2.0       3       data/pilot_sweep_2.jsonl
# sweep_3    0.05   3.0       5       data/pilot_sweep_3.jsonl
#
# Wall-time estimate (with prefix-forced CoT eval):
#   ~15 min per (question x step) at 3 epochs
#   ~21 min per (question x step) at 5 epochs
#   sweep_1: 7 x 15 = 105 min
#   sweep_2: 7 x 15 = 105 min
#   sweep_3: 7 x 21 = 147 min
#   Total sequential: ~357 min (6 hrs) + buffer = 7 hrs
#
# sweep_4 (8 epochs, ~210 min) runs in submit_fur_sweep4.sh separately.
# Run select_best_conditions.py after BOTH jobs complete.

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

# ── sweep_1: β=0.10, kl=1.0, epochs=3 ────────────────────────────────────────
echo ""
echo "=== sweep_1: beta=0.1, kl_coeff=1.0, epochs=3 ==="
python run_fur_pilot.py \
    --model_name "$MODEL_NAME" \
    --fur_file "$FUR_FILE" \
    --question_ids $QUESTIONS \
    --step_ids 0 \
    --output_file "data/pilot_sweep_1.jsonl" \
    --epochs 3 \
    --lr "$LR" \
    --beta 0.1 \
    --kl_coeff 1.0 \
    --seed "$SEED" \
    --ff2 \
    --pos

# ── sweep_2: β=0.10, kl=2.0, epochs=3 ────────────────────────────────────────
echo ""
echo "=== sweep_2: beta=0.1, kl_coeff=2.0, epochs=3 ==="
python run_fur_pilot.py \
    --model_name "$MODEL_NAME" \
    --fur_file "$FUR_FILE" \
    --question_ids $QUESTIONS \
    --step_ids 0 \
    --output_file "data/pilot_sweep_2.jsonl" \
    --epochs 3 \
    --lr "$LR" \
    --beta 0.1 \
    --kl_coeff 2.0 \
    --seed "$SEED" \
    --ff2 \
    --pos

# ── sweep_3: β=0.05, kl=3.0, epochs=5 ────────────────────────────────────────
echo ""
echo "=== sweep_3: beta=0.05, kl_coeff=3.0, epochs=5 ==="
python run_fur_pilot.py \
    --model_name "$MODEL_NAME" \
    --fur_file "$FUR_FILE" \
    --question_ids $QUESTIONS \
    --step_ids 0 \
    --output_file "data/pilot_sweep_3.jsonl" \
    --epochs 5 \
    --lr "$LR" \
    --beta 0.05 \
    --kl_coeff 3.0 \
    --seed "$SEED" \
    --ff2 \
    --pos

echo ""
echo "=== sweep_1/2/3 complete. Submit submit_fur_sweep4.sh if not already running. ==="
echo "=== Run select_best_conditions.py after sweep_4 also finishes. ==="
date
