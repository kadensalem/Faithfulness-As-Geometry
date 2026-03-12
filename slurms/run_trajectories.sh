#!/bin/bash
#SBATCH --job-name bool-traj
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --qos=<YOUR_QOS>
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
#SBATCH --mem=80GB
#SBATCH --requeue
#SBATCH -o logs/bool-traj-%j.out
#SBATCH -e logs/bool-traj-%j.err

# ─── Phases 3 + 4: Extract trajectory embeddings & compare ───
#
# Phase 3: Load LLaMA-3-8B, run each boolean CoT through the model
#   with output_hidden_states=True, extract the hidden state at the
#   anchor-last token of each node step from a middle-to-late layer.
#
# Phase 4: Load the embeddings + faithfulness labels from Phase 2,
#   group by faithful/unfaithful, compute similarity matrices at
#   orders 0-3, PCA plots, and group-averaged statistics.

date
mkdir -p logs

# ── Activate conda environment ──
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate fur

# ── HuggingFace cache ──
export HF_HOME="/scratch/general/vast/<YOUR_UID>/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# module load cuda/12.1
nvidia-smi

# ── Configuration ──
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
LAYER_INDEX="auto"          # "auto" picks middle-to-late (layers 12-24 for 8B)
POOLING="anchor_last"       # extract final token of Result/Final Answer line
ACCUMULATION="cumulative"   # context-cumulative trajectory
SEED=42

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

GEO_DATA="data/boolean_cots.json"
FAITH_FILE="results/boolean_faithfulness.jsonl"
EMBED_DIR="results/boolean_trajectories"
COMPARE_DIR="results/comparison"
mkdir -p "$EMBED_DIR" "$COMPARE_DIR"

echo "Project root: $PROJECT_ROOT"
echo "Model: $MODEL_NAME"
echo "Layer: $LAYER_INDEX, Pooling: $POOLING, Accumulation: $ACCUMULATION"
echo "Geometry data: $GEO_DATA"
echo "Faithfulness labels: $FAITH_FILE"

# ──────────────────────────────────────────────────────────────
# Phase 3: Trajectory embedding extraction
# ──────────────────────────────────────────────────────────────
echo ""
echo "═══ Phase 3: Extracting trajectory embeddings ═══"

python geometry/cot-hidden-dynamic-v2.py \
  --hf_model "$MODEL_NAME" \
  --data_file "$GEO_DATA" \
  --boolean \
  --pooling "$POOLING" \
  --accumulation "$ACCUMULATION" \
  --layer_index "$LAYER_INDEX" \
  --device cuda:0 \
  --save_dir "$EMBED_DIR"

echo "Phase 3 complete."

# ──────────────────────────────────────────────────────────────
# Phase 4: Faithful vs unfaithful trajectory comparison
# ──────────────────────────────────────────────────────────────
echo ""
echo "═══ Phase 4: Comparing trajectories ═══"

python compare_trajectories.py \
  --embeddings_dir "$EMBED_DIR" \
  --geometry_data "$GEO_DATA" \
  --faithfulness_file "$FAITH_FILE" \
  --orders "0,1,2,3" \
  --save_dir "$COMPARE_DIR"

echo "Phase 4 complete."
date
