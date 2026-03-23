#!/bin/bash
#SBATCH --job-name fur-sweep-verify
#SBATCH --account=cs6966
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --qos=soc-gpu-class-grn
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --time=0:30:00
#SBATCH --mem=80GB
#SBATCH --requeue
#SBATCH -o logs/fur-sweep-verify-%j.out
#SBATCH -e logs/fur-sweep-verify-%j.err

# ─── Sweep verification run ───────────────────────────────────────────────────
# Runs sweep_1 params on openbook_1955 only, step 0, 1 epoch.
# Checks:
#   1. Output file is written and parseable
#   2. new_cot at epoch 0 is structured (all 4 answer blocks present)
#   3. probs at epoch 0 ~ [0.9692, 0.0084, 0.0045, 0.0178] (Bowman pass unchanged)
#   4. select_best_conditions.py runs without error on this single-record output

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
VERIFY_OUT="data/sweep_verify.jsonl"

echo "=== Verification run: sweep_1 params, openbook_1955, step 0, epochs=1 ==="

python run_fur_pilot.py \
    --model_name "$MODEL_NAME" \
    --fur_file "$FUR_FILE" \
    --question_ids openbook_1955 \
    --step_ids 0 \
    --output_file "$VERIFY_OUT" \
    --epochs 1 \
    --lr 5e-5 \
    --beta 0.1 \
    --kl_coeff 1.0 \
    --seed 42 \
    --ff2 \
    --pos

echo ""
echo "=== Verification checks ==="
python3 - <<'PYEOF'
import json, sys

path = "data/sweep_verify.jsonl"
try:
    with open(path) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    print(f"[OK] Output written: {len(recs)} record(s)")
except Exception as e:
    print(f"[FAIL] Could not read output: {e}")
    sys.exit(1)

rec = recs[0]
e0 = rec["unlearning_results"]["0"]

# Check 2: structured new_cot
cot = e0["new_cot"][0] if e0["new_cot"] else ""
blocks_found = sum(1 for L in ["A","B","C","D"] if f"Answer {L}:" in cot)
reasoning_found = cot.count("* Reasoning:")
conclusion_found = cot.count("* Conclusion:")
if blocks_found == 4 and reasoning_found >= 4 and conclusion_found >= 4:
    print(f"[OK] new_cot at epoch 0 is structured (4 blocks, {reasoning_found} Reasoning, {conclusion_found} Conclusion)")
else:
    print(f"[WARN] new_cot structure incomplete: {blocks_found} Answer blocks, {reasoning_found} Reasoning, {conclusion_found} Conclusion")
    print("  new_cot preview:", repr(cot[:300]))

# Check 3: probs match pilot
expected = [0.9692, 0.0084, 0.0045, 0.0178]
actual = e0["probs"]
diffs = [abs(a - b) for a, b in zip(actual, expected)]
if all(d < 0.01 for d in diffs):
    print(f"[OK] probs at epoch 0 match pilot: {[round(p,4) for p in actual]}")
else:
    print(f"[WARN] probs differ from pilot")
    print(f"  expected: {expected}")
    print(f"  actual:   {[round(p,4) for p in actual]}")
PYEOF

echo ""
echo "=== Running select_best_conditions.py on verify output ==="
python select_best_conditions.py --sweep_files data/sweep_verify.jsonl

echo ""
echo "Verification complete."
date
