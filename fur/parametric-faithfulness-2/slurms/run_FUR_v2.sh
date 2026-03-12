#!/bin/bash

#SBATCH --job-name furV2
#SBATCH --account=
#SBATCH --partition=
#SBATCH --qos=
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1:00:00
#SBATCH --mem=100GB
#SBATCH --requeue
#SBATCH -o logs/furV2n%j


date
# source conda and activate environment
# source conda.sh
# conda activate path to conda env

# HF Cache
# mkdir -p "cache_path"
export HF_DATASETS_CACHE=""
export HF_HOME=""
# module load cuda/12.1
# nvidia-smi

MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct" 
DATASET="openbook"
# DATASET="arc-challenge"
LR="3e-5"
# NUM_P=0
NUM_C=5
SEED=1002

echo "Running model: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Learning rate: $LR"
echo "Num paraphrases: $NUM_P"
echo "Num consistency: $NUM_C"
echo "Seed: $SEED"

python furV2.py \
  --model_name "$MODEL_NAME" \
  --strategy sentencize \
  --dataset "$DATASET" \
  --lr "$LR" \
  --pos \
  --ff2 \
  --method npo_KL \
  --consistency_cot \
  --consistency_cot_num $NUM_C \
  --seed $SEED \
  # --new_cot \
  # --no_unlearn \
  # --paraphrase \
  # --num_p 1 \
date
