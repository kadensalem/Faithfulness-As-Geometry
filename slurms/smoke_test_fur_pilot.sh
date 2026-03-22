#!/bin/bash
#SBATCH --job-name fur-smoke
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --qos=<YOUR_QOS>
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0:30:00
#SBATCH --mem=100GB
#SBATCH -o logs/fur-smoke-%j.out
#SBATCH -e logs/fur-smoke-%j.err

# ─── Smoke test: 1 question, 1 epoch ───
#
# Runs run_fur_pilot.py on a single question (openbook_1955) with 1 epoch
# to verify:
#   1. The model loads and computes nocot_probs / cot_probs without OOM
#   2. cot_to_otfd produces enough targets (NT > 2) for all 5 MCQ steps
#   3. unlearn_single() runs and returns a valid unlearning_results dict
#   4. Output JSONL is written with the expected fields:
#        id, question, step_idx, options, correct, initial_cot,
#        initial_cot_probs, initial_probs, prediction, cot_prediction,
#        cot_step, segmented_cot, unlearning_results
#
# Expected wall time: ~10 min (1 question x 5 steps x 1 epoch each).

date
mkdir -p logs

# ── Conda environment ──
source ~/software/pkg/miniforge3/etc/profile.d/conda.sh
conda activate fur

# ── HuggingFace cache ──
export HF_HOME="/scratch/general/vast/${USER}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

nvidia-smi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_FILE="data/smoke_fur_results.jsonl"

echo "=== SMOKE TEST: 1 question, 1 epoch ==="
echo "Model  : $MODEL_NAME"
echo "Output : $OUTPUT_FILE"

python run_fur_pilot.py \
    --model_name "$MODEL_NAME" \
    --fur_file "data_tune_30/mcq_cots_fur.jsonl" \
    --question_ids openbook_1955 \
    --output_file "$OUTPUT_FILE" \
    --epochs 1 \
    --lr 1e-5 \
    --seed 42 \
    --ff2

echo ""
echo "=== Output file contents ==="
python3 -c "
import json
lines = [json.loads(l) for l in open('$OUTPUT_FILE')]
print(f'Lines written: {len(lines)}')
for l in lines:
    ur = l['unlearning_results']
    epochs = sorted(int(k) for k in ur.keys())
    e0_pred = ur['0']['prediction']
    e1_pred = ur[str(epochs[-1])]['prediction']
    print(f'  step {l[\"step_idx\"]}: epoch-0 pred={e0_pred}  epoch-last pred={e1_pred}  flip={e0_pred != e1_pred}')
print('Output fields:', list(lines[0].keys()))
"

echo "Smoke test done."
date
