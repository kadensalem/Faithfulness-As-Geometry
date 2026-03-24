# Faithfulness-As-Geometry: Pipeline Reference

This document describes every experiment flow, SLURM script, and key Python
script in the project. Use it as the entry point when picking up the project
or submitting new runs.

---

## Research Question

> Does the geometry of a language model's internal hidden-state trajectory
> while generating a chain-of-thought correlate with the *faithfulness* of
> that CoT — i.e. does the reasoning actually drive the model's final answer?

**Faithfulness operationalisation (FUR unlearning):** For each answer-block
step in a structured MCQ CoT, run NPO+KL unlearning on that step and measure
whether the model's final answer changes (FF-HARD) or the correct-answer
probability drops (FF-SOFT). Steps that flip the answer are considered
"load-bearing" / faithful.

**Geometry hypothesis:** Faithful steps should produce distinctive
hidden-state trajectories — higher displacement, more directional movement,
different curvature — compared to unfaithful steps.

---

## Overall Pipeline (Original / Template)

```
Phase 1  generate_mcq_cots.py     →  data_tune_30/mcq_cots_fur.jsonl
Phase 2  run_fur_pilot.py         →  data/sweep_cond_*.jsonl
         (+ select_best_conditions.py)
Phase 3  extract_fur_embeddings.py →  data/fur_anchor_embeddings*.pkl
Phase 4  geometry_compare.ipynb   →  analysis + plots
```

The original template scripts (`run_generate.sh`, `run_unlearn.sh`,
`run_trajectories.sh`, `submit_pipeline.sh`) capture this generic flow with
placeholder account/partition values. The active experiments below use the
project-specific scripts.

---

## Experiment Flows

### Flow A — Pilot Sweep (7 questions, 4 hyperparameter conditions)

**Goal:** Validate FUR unlearning on a small set of questions and sweep
hyperparameters to find conditions that reliably produce FF-HARD flips.

```
submit_fur_sweep.sh      (cond1/cond2/cond3, 7 Qs)
submit_fur_sweep4.sh     (cond4 only, 7 Qs)
        ↓  both complete
select_best_conditions.py  →  data/best_conditions.json  (7 Qs)
submit_extract_embeddings.sh  →  data/fur_anchor_embeddings.pkl
        ↓
geometry_compare.ipynb  (pilot section)
```

Pilot condition mapping:

| Condition | β | kl_coeff | epochs | Output file |
|---|---|---|---|---|
| sweep_1 / cond_A | 0.10 | 1.0 | 3 | `data/pilot_sweep_1.jsonl` |
| sweep_2 / cond_B | 0.10 | 2.0 | 3 | `data/pilot_sweep_2.jsonl` |
| sweep_3 / cond_C | 0.05 | 3.0 | 5 | `data/pilot_sweep_3.jsonl` |
| sweep_4 / cond_D | 0.02 | 3.0 | 8 | `data/pilot_sweep_4.jsonl` |

---

### Flow B — 30-Question Full Sweep (canonical, main experiment)

**Goal:** Scale the hyperparameter sweep to 30 questions; use the best
per-question condition for geometry extraction and analysis.

```
submit_fur_sweep_30.sh    (array 0-29, cond1+cond4 sequential per task)
submit_fur_sweep_30b.sh   (array 0-29, cond2+cond3 sequential per task)
        ↓  both arrays complete
merge_sweep30.sh          →  data/sweep30_cond{1-4}.jsonl
consolidate_sweep.py (manual)  →  data/sweep_cond_{A-D}.jsonl  (canonical)
select_best_conditions.py  →  data/best_conditions_30.json  (30 Qs)
submit_extract_embeddings_30.sh  →  data/fur_anchor_embeddings_30.pkl
        ↓
geometry_compare.ipynb  (30-question section)
```

Canonical condition mapping (same hyperparams, stable names):

| Canonical | Sweep name | β | kl_coeff | epochs |
|---|---|---|---|---|
| cond_A | sweep30_cond4 | 0.10 | 1.0 | 3 |
| cond_B | sweep30_cond1 | 0.10 | 2.0 | 3 |
| cond_C | sweep30_cond2 | 0.05 | 3.0 | 5 |
| cond_D | sweep30_cond3 | 0.02 | 3.0 | 8 |

Canonical files (`data/sweep_cond_{A-D}.jsonl`) are deduplicated by
`(id, step_idx)` and contain added fields `sweep_condition`, `beta`,
`kl_coeff`, `n_epochs`. Original files are archived to `data/sweep_archive/`.

---

### Flow C — Free-Form CoT Geometry (comparison condition)

**Goal:** Test whether the faithfulness-geometry signal holds on raw,
unstructured CoTs (no structural anchors). Serves as a negative control.

```
[prereq: Flow B complete, best_conditions_30.json exists]
submit_freeform_embeddings.sh  →  data/freeform_embeddings.pkl
        ↓
geometry_compare.ipynb  (Cells A–D, free-form section)
```

`extract_freeform_embeddings.py` loads `initial_cot[0]` (the pre-unlearning
CoT) from the FUR output files, runs a single forward pass per question at
L8/L14/L28, and computes displacement/curvature/segment metrics.

**Key finding:** Mean displacement at L14 ≈ 8.4 (same scale as structured
inter-anchor), all Spearman correlations with FF-SOFT |r| < 0.3. Structural
anchors appear necessary to isolate the faithfulness-geometry relationship.

---

### Flow D — Sub-Block Unlearning Mode Sweep (current experiment)

**Goal:** Test whether the premise or the reasoning line within each answer
block carries the faithfulness signal, by targeting sub-blocks rather than
whole Answer-X blocks.

```
[prereq: Flow B complete, best_conditions_30.json exists]
submit_subblock_sweep.sh   (array 0-119 = 30 Qs × 4 modes, step_idx=0)
        ↓  all tasks complete
# Merge per-question files per mode:
cat data/subblock_whole_block_q*.jsonl      > data/subblock_cond_whole_block.jsonl
cat data/subblock_premise_only_q*.jsonl     > data/subblock_cond_premise_only.jsonl
cat data/subblock_reasoning_only_q*.jsonl   > data/subblock_cond_reasoning_only.jsonl
cat data/subblock_premise_and_reasoning_q*.jsonl > data/subblock_cond_premise_and_reasoning.jsonl
        ↓
geometry_compare.ipynb  (Cells E–F, sub-block section)
```

Unlearning modes:

| Mode | What is targeted | `unlearn_target_type` |
|---|---|---|
| `whole_block` | Full Answer-X block (existing behaviour) | `whole_block` |
| `premise_only` | `* Premise: …` line only | `premise` |
| `reasoning_only` | `* Reasoning: …` line only | `reasoning` |
| `premise_and_reasoning` | Premise + Reasoning together | `premise_and_reasoning` |

Uses per-question best hyperparameters from `best_conditions_30.json`;
falls back to cond_A (β=0.1, kl=1.0, ep=3) when condition is `null`.

---

## SLURM Scripts Reference

### Active / Project-Specific

| Script | Type | Array | Wall Time | Purpose |
|---|---|---|---|---|
| `submit_fur_sweep_30.sh` | array job | 0–29 | 1 hr/task | 30-question sweep, cond1+cond4 (lighter conditions) |
| `submit_fur_sweep_30b.sh` | array job | 0–29 | 1.5 hr/task | 30-question sweep, cond2+cond3 (heavier conditions) |
| `merge_sweep30.sh` | bash (no SLURM) | — | <1 min | Concatenate per-question parts into `sweep30_cond{N}.jsonl` |
| `submit_extract_embeddings_30.sh` | single job | — | 12 hr | Re-run FUR to best epoch, extract 9-anchor embeddings (30 Qs) |
| `submit_freeform_embeddings.sh` | single job | — | 4 hr | Extract L8/L14/L28 geometry metrics from initial CoTs |
| `submit_subblock_sweep.sh` | array job | 0–119 | 1.5 hr/task | 4-mode sub-block sweep (30 Qs × 4 modes), step_idx=0 |
| `submit_geometry_analysis.sh` | single job | — | 20 min | Batch-run `run_geometry_analysis.py` (no GPU used) |
| `submit_anchor_embeddings.sh` | single job | — | 1 hr | Extract 9-anchor embeddings for base model (no FUR) |

### Pilot / Earlier Experiments

| Script | Purpose |
|---|---|
| `submit_fur_pilot.sh` | Initial 3-question pilot, 5 epochs, no sweep |
| `submit_fur_sweep.sh` | 7-question pilot sweep, cond1/2/3 |
| `submit_fur_sweep4.sh` | 7-question pilot sweep, cond4 only (heavier, separate job) |
| `submit_extract_embeddings.sh` | Anchor embedding extraction for 7-question pilot |
| `submit_fur_sweep_verify.sh` | Single question × 1 epoch smoke test for sweep correctness |
| `smoke_test_fur_pilot.sh` | 1 question × 1 epoch verification of `run_fur_pilot.py` output schema |

### Template Scripts (not runnable as-is)

| Script | Purpose |
|---|---|
| `run_generate.sh` | Phase 1 template: generate MCQ CoTs |
| `run_unlearn.sh` | Phase 2 template: NPO unlearning loop |
| `run_trajectories.sh` | Phase 3+4 template: extract trajectories and compare |
| `submit_pipeline.sh` | Chains Phase 1→2→3+4 with SLURM `--dependency=afterok` |

> Templates have placeholder `<YOUR_ACCOUNT>` / `<YOUR_PARTITION>` values.
> Fill in `cs6966` / `soc-gpu-class-grn` / `soc-gpu-class-grn` for CHPC.

---

## Key Python Scripts

| Script | Role |
|---|---|
| `generate_mcq_cots.py` | Phase 1: prompt LLaMA with structured Premise/Reasoning/Conclusion format, write `mcq_cots_fur.jsonl` |
| `run_fur_pilot.py` | Main FUR driver: loads CoTs, computes probs, calls `unlearn_single()` per (question, step, mode) |
| `select_best_conditions.py` | Selects best (condition, epoch) per (question, step): FF-HARD filter → structure quality → earliest clean epoch |
| `extract_fur_embeddings.py` | Re-runs FUR to selected epoch, extracts 9-anchor embeddings (pilot, 6 Qs) |
| `extract_anchor_embeddings.py` | Extracts 9-anchor embeddings from base model (no FUR) |
| `extract_freeform_embeddings.py` | Extracts L8/L14/L28 displacement/curvature metrics from initial CoTs |
| `compute_fur_labels.py` | Computes FF-HARD and FF-SOFT labels from FUR JSONL output |
| `run_geometry_analysis.py` | Batch (non-interactive) version of `geometry_compare.ipynb` |
| `join_pilot_data.py` | Joins anchor trajectory embeddings with FUR faithfulness labels on question ID |
| `mcq_dataload.py` | `MCQDataHandler`: implements the DataHandler interface that `furV2.py` expects |
| `mcq_unlearn.py` | Standalone Phase 2 script (older; prefer `run_fur_pilot.py`) |
| `compare_trajectories.py` | Phase 4: PCA plots and similarity matrices for faithful vs unfaithful CoTs |
| `patch_conclusion_embeddings.py` | Targeted re-extraction of missing conclusion token embeddings |
| `qualitative_example.py` | Pretty-print the best FF-HARD example with displacement vectors |

---

## FUR Internals (Do Not Modify)

Located in `fur/parametric-faithfulness-2/`:

| File | Role |
|---|---|
| `furV2.py` | `unlearn_single()`: loads model+oracle, runs NPO+KL, calls `evaluate()` per epoch |
| `data.py` | `cot_to_otfd()`: builds `SegmentOTFDataset` from `segmented_cot` with prefix accumulation |
| `evaluate.py` | `answer_probabilities()`, `generation_fixed_cot()`: Bowman-style answer prob computation |
| `util.py` | `set_random_seed()` and misc helpers |

**Important:** `cot_to_otfd` calls `all.remove(target)` using Python `==`
equality. For sub-block modes, `run_fur_pilot.py` patches `cots_train` with a
deep-copied target using `is`-identity replacement before calling
`unlearn_single()`, so the remove works correctly.

---

## Data Files

| File | Created by | Contents |
|---|---|---|
| `data_tune_30/mcq_cots_fur.jsonl` | `generate_mcq_cots.py` | 30 questions with `segmented_cot` (5 blocks per question) |
| `data_tune_30/qids.txt` | manual / generate step | One question ID per line; defines array-task → question mapping |
| `data/sweep30_cond{1-4}.jsonl` | `merge_sweep30.sh` | Merged per-question parts; 150 records per condition |
| `data/sweep_cond_{A-D}.jsonl` | consolidation step | Canonical, deduplicated condition files with metadata fields |
| `data/sweep_archive/` | consolidation step | All pre-consolidation JSONL files |
| `data/best_conditions_30.json` | `select_best_conditions.py` | Keys `"{qid}_step{N}"` → best (condition, epoch, ff_hard, ff_soft) |
| `data/fur_anchor_embeddings_30.pkl` | `extract_fur_embeddings.py` | Per-question dicts: 9-anchor embeddings, FF-SOFT, condition metadata |
| `data/pilot_trajectories.pkl` | `extract_anchor_embeddings.py` | Base-model 9-anchor embeddings (no FUR) |
| `data/freeform_embeddings.pkl` | `extract_freeform_embeddings.py` | Per-question L8/L14/L28 geometry metrics from initial CoTs |
| `data/subblock_{mode}_q{N}.jsonl` | `submit_subblock_sweep.sh` | Per-question sub-block unlearning results; merge before analysis |

---

## Notebook: `geometry_compare.ipynb`

| Cell group | Content |
|---|---|
| Setup (cells 0–3) | Imports, load pilot embeddings, basic stats |
| Pilot analysis (cells 4–12) | Displacement/curvature correlations, Wilcoxon test, cosine similarity |
| 30-question analysis (cells 13–21) | Full-sweep geometry tests, Spearman rank correlation |
| Free-form comparison (cells 22–26, Cells A–D) | Load `freeform_embeddings.pkl`, null baseline, layer comparison, structured vs free-form summary table |
| Sub-block analysis (cells 27–29, Cells E–F) | Load per-mode subblock results, FF-HARD/FF-SOFT comparison table, Mann-Whitney U vs whole_block, Spearman r between reasoning_only and whole_block |

Batch equivalent: `run_geometry_analysis.py` + `submit_geometry_analysis.sh`

---

## Common Commands

```bash
# Check running jobs
squeue -u $USER

# Submit the 30-question sweep (both arrays)
sbatch slurms/submit_fur_sweep_30.sh
sbatch slurms/submit_fur_sweep_30b.sh

# After both arrays finish, merge and select best conditions
bash slurms/merge_sweep30.sh
python select_best_conditions.py \
    --input data/sweep30_cond1.jsonl data/sweep30_cond2.jsonl \
            data/sweep30_cond3.jsonl data/sweep30_cond4.jsonl \
    --output data/best_conditions_30.json

# Submit free-form embedding extraction
sbatch slurms/submit_freeform_embeddings.sh

# Submit sub-block sweep
sbatch slurms/submit_subblock_sweep.sh

# After sub-block sweep finishes, merge per mode
for MODE in whole_block premise_only reasoning_only premise_and_reasoning; do
    cat data/subblock_${MODE}_q*.jsonl > data/subblock_cond_${MODE}.jsonl
done
```

---

## Environment

- **Conda env:** `fur-sm120`
- **Partition / QOS:** `soc-gpu-class-grn` / `soc-gpu-class-grn`
- **Account:** `cs6966`
- **GPU:** `rtxpr6000bl` (RTX 6000)
- **HF cache:** `/scratch/general/vast/$USER/hf_cache`
- **Model:** `meta-llama/Meta-Llama-3-8B-Instruct`
