## Quickstart

### Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-0.6B
```

### Two Main Commands

### 1) PCA from text output
Use this when you have chains-of-thought as text in a JSON file.

```bash
python cot-hidden-dynamic.py \
  --hf_model ./Qwen3-0.6B \
  --data_file sample_data/faithful_unfaithful_toy.json \
  --pooling step_mean --accumulation cumulative \
  --similarity_order 1 \
  --save_dir results/demo/Qwen3-0.6B \
  --save_html
```

### 2) PCA from precomputed embeddings
Use this when you already have per-step embeddings saved as `.npy` files.

```bash
python cot-hidden-dynamic.py \
  --embeddings_dir sample_data \
  --embeddings_meta sample_data/embeddings_meta.json \
  --sections logicFactorial5 \
  --similarity_order 1 \
  --save_dir results/demo/precomputed \
  --save_html
```

### 3) Results
All output files are saved in the results folder.