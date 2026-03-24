#!/bin/bash
# Merge per-question output files from the 30-question sweep array jobs.
# Run this AFTER both submit_fur_sweep_30.sh AND submit_fur_sweep_30b.sh
# array tasks have all completed (check with: squeue -u $USER).
#
# Usage:
#   cd /uufs/chpc.utah.edu/common/home/u1427573/Faithfulness-As-Geometry
#   bash slurms/merge_sweep30.sh


PROJECT_ROOT="/uufs/chpc.utah.edu/common/home/u1427573/Faithfulness-As-Geometry"
cd "$PROJECT_ROOT"

echo "Merging sweep30 per-question files..."

for COND in 1 2 3 4; do
    PARTS=(data/sweep30_cond${COND}_q*.jsonl)
    if [ ${#PARTS[@]} -eq 0 ] || [ ! -f "${PARTS[0]}" ]; then
        echo "  [SKIP] No parts found for cond${COND}"
        continue
    fi
    N_PARTS=${#PARTS[@]}
    OUT="data/sweep30_cond${COND}.jsonl"
    cat "${PARTS[@]}" > "$OUT"
    N_LINES=$(wc -l < "$OUT")
    echo "  cond${COND}: merged ${N_PARTS} part files → ${OUT}  (${N_LINES} records)"
done

echo ""
echo "Merge complete. Verify record counts:"
for COND in 1 2 3 4; do
    OUT="data/sweep30_cond${COND}.jsonl"
    if [ -f "$OUT" ]; then
        N=$(wc -l < "$OUT")
        # Expect: 30 questions × 5 steps = 150 records per condition
        echo "  sweep30_cond${COND}.jsonl: ${N} records  (expected: 150)"
    fi
done

echo ""
echo "Next step: run select_best_conditions.py"
echo "  python select_best_conditions.py \\"
echo "    --sweep_files data/sweep30_cond1.jsonl data/sweep30_cond2.jsonl \\"
echo "                  data/sweep30_cond3.jsonl data/sweep30_cond4.jsonl \\"
echo "    --output data/best_conditions_30.json"
