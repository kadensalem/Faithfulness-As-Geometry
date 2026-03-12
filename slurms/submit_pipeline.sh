#!/bin/bash
# ─── Submit the full Faithfulness-as-Geometry pipeline ───
#
# Uses SLURM dependency chaining so each phase waits for the
# previous one to finish successfully before starting.
#
# Usage:
#   cd <project_root>
#   bash slurms/submit_pipeline.sh

set -euo pipefail
mkdir -p logs

echo "Submitting Phase 1: Generate boolean CoTs..."
JOB1=$(sbatch --parsable slurms/run_generate.sh)
echo "  Job ID: $JOB1"

echo "Submitting Phase 2: NPO unlearning (depends on $JOB1)..."
JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} slurms/run_unlearn.sh)
echo "  Job ID: $JOB2"

echo "Submitting Phases 3+4: Trajectories + comparison (depends on $JOB2)..."
JOB3=$(sbatch --parsable --dependency=afterok:${JOB2} slurms/run_trajectories.sh)
echo "  Job ID: $JOB3"

echo ""
echo "Pipeline submitted:"
echo "  Phase 1 (generate):     $JOB1"
echo "  Phase 2 (unlearn):      $JOB2  (after $JOB1)"
echo "  Phase 3+4 (trajectory): $JOB3  (after $JOB2)"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Cancel all:   scancel $JOB1 $JOB2 $JOB3"
