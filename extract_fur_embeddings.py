#!/usr/bin/env python3
"""
extract_fur_embeddings.py

Phase 3: Re-run FUR unlearning to the selected epoch, then extract anchor
         embeddings at the model state reached at that epoch.

For each (question_id, condition, epoch) triple in best_conditions.json:
  1. Maps condition name → hyperparameters (beta, kl_coeff, lr, seed, ...)
  2. Retrieves the new_cot text from the condition's sweep JSONL
  3. Re-runs NPO+KL unlearning with identical seed, stopping at target epoch
  4. Prefix-forced forward pass: prompt + new_cot, output_hidden_states=True
  5. Extracts 9 structural anchor embeddings (same positions as Phase 2)
  6. Attaches FF-SOFT score and condition metadata from best_conditions.json

Output: data/fur_anchor_embeddings.pkl — list of dicts, one per question.
        data/pca_sanity_fur/          — per-question 2D PCA CSVs

Usage:
    python extract_fur_embeddings.py \\
        --best_conditions data/best_conditions.json \\
        --fur_file data_tune_30/mcq_cots_fur.jsonl \\
        --hf_model meta-llama/Meta-Llama-3-8B-Instruct \\
        --middle_layer 14 \\
        --output_file data/fur_anchor_embeddings.pkl
"""

import argparse
import gc
import json
import os
import pickle
import random
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ─── FUR library imports ─────────────────────────────────────────────────────
FUR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fur", "parametric-faithfulness-2")
sys.path.insert(0, FUR_DIR)

from data import FRCollator, cot_to_otfd                              # noqa: E402
from evaluate import answer_probabilities, generation_fixed_cot       # noqa: E402
from furV2 import compute_loss, get_linear_schedule_with_warmup       # noqa: E402
from util import set_random_seed                                       # noqa: E402

# mcq_dataload lives in the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from mcq_dataload import MCQDataHandler                                # noqa: E402

# Anchor extraction helpers live in extract_anchor_embeddings.py (project root)
from extract_anchor_embeddings import (                                # noqa: E402
    find_anchor_token_positions,
    span_mean,
    parse_cot_blocks,
    pca_sanity_check,
)


# ─── Condition → hyperparameter table ────────────────────────────────────────
# Must match the SLURM sweep scripts exactly.
# Condition names are those written by select_best_conditions.py (= file stem).

CONDITION_PARAMS: Dict[str, dict] = {
    # ── Original 7-question pilot sweeps ──────────────────────────────────────
    "sweep_verify":   dict(beta=0.10, kl_coeff=1.0, lr=5e-5, seed=42,
                           ff2=True, pos=True, method="npo_KL"),
    "sweep_1":        dict(beta=0.10, kl_coeff=1.0, lr=5e-5, seed=42,
                           ff2=True, pos=True, method="npo_KL"),
    "sweep_2":        dict(beta=0.10, kl_coeff=2.0, lr=5e-5, seed=42,
                           ff2=True, pos=True, method="npo_KL"),
    "sweep_3":        dict(beta=0.05, kl_coeff=3.0, lr=5e-5, seed=42,
                           ff2=True, pos=True, method="npo_KL"),
    "sweep_4":        dict(beta=0.02, kl_coeff=3.0, lr=5e-5, seed=42,
                           ff2=True, pos=True, method="npo_KL"),
    # ── 30-question 5-step sweep conditions ───────────────────────────────────
    # pilot_sweep_N equivalents: cond1=sweep_2, cond2=sweep_3, cond3=sweep_4, cond4=sweep_1
    "sweep30_cond1":  dict(beta=0.10, kl_coeff=2.0, lr=5e-5, seed=42,
                           ff2=True, pos=True, method="npo_KL"),
    "sweep30_cond2":  dict(beta=0.05, kl_coeff=3.0, lr=5e-5, seed=42,
                           ff2=True, pos=True, method="npo_KL"),
    "sweep30_cond3":  dict(beta=0.02, kl_coeff=3.0, lr=5e-5, seed=42,
                           ff2=True, pos=True, method="npo_KL"),
    "sweep30_cond4":  dict(beta=0.10, kl_coeff=1.0, lr=5e-5, seed=42,
                           ff2=True, pos=True, method="npo_KL"),
    # backward-compat aliases for per-question part files (pre-merge)
    "pilot_sweep_1":  dict(beta=0.10, kl_coeff=1.0, lr=5e-5, seed=42,
                           ff2=True, pos=True, method="npo_KL"),
    "pilot_sweep_2":  dict(beta=0.10, kl_coeff=2.0, lr=5e-5, seed=42,
                           ff2=True, pos=True, method="npo_KL"),
    "pilot_sweep_3":  dict(beta=0.05, kl_coeff=3.0, lr=5e-5, seed=42,
                           ff2=True, pos=True, method="npo_KL"),
    "pilot_sweep_4":  dict(beta=0.02, kl_coeff=3.0, lr=5e-5, seed=42,
                           ff2=True, pos=True, method="npo_KL"),
}

CONDITION_SWEEP_FILE: Dict[str, str] = {
    # ── Original 7-question pilot sweeps ──────────────────────────────────────
    "sweep_verify":   "data/sweep_verify.jsonl",
    "sweep_1":        "data/pilot_sweep_1.jsonl",
    "sweep_2":        "data/pilot_sweep_2.jsonl",
    "sweep_3":        "data/pilot_sweep_3.jsonl",
    "sweep_4":        "data/pilot_sweep_4.jsonl",
    # ── 30-question 5-step sweep conditions ───────────────────────────────────
    "sweep30_cond1":  "data/sweep30_cond1.jsonl",
    "sweep30_cond2":  "data/sweep30_cond2.jsonl",
    "sweep30_cond3":  "data/sweep30_cond3.jsonl",
    "sweep30_cond4":  "data/sweep30_cond4.jsonl",
    # backward-compat aliases
    "pilot_sweep_1":  "data/pilot_sweep_1.jsonl",
    "pilot_sweep_2":  "data/pilot_sweep_2.jsonl",
    "pilot_sweep_3":  "data/pilot_sweep_3.jsonl",
    "pilot_sweep_4":  "data/pilot_sweep_4.jsonl",
}

TRAJ_KEYS = [
    "Answer_A", "Reasoning_A", "Answer_B", "Reasoning_B",
    "Answer_C", "Reasoning_C", "Answer_D", "Reasoning_D", "Final_Answer",
]


# ─── Data helpers ─────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def adapt_record(rec: dict, model, tokenizer, DH) -> dict:
    """Compute nocot_probs / cot_probs and return a furV2-compatible target dict."""
    raw = rec["raw_instance"]
    cot_text = rec["cot"]
    _, nocot_probs, _ = answer_probabilities(model, tokenizer, DH, raw)
    cot_probs, _ = generation_fixed_cot(model, tokenizer, DH, raw, cot_text)
    return {
        "id":            rec["id"],
        "question":      rec["question"],
        "correct_letter": rec["correct_letter"],
        "cot_prompt":    rec["cot_prompt"],
        "cot":           [cot_text],
        "options":       rec["options"],
        "nocot_probs":   nocot_probs.tolist(),
        "cot_probs":     [cot_probs.tolist()],
        "segmented_cot": rec["segmented_cot"],
        "raw_instance":  rec["raw_instance"],
    }


def load_new_cot(condition: str, qid: str, epoch: int) -> Optional[str]:
    """
    Retrieve the new_cot text stored at `epoch` for `qid` in the condition's
    sweep JSONL. Returns None if not found.
    """
    rel = CONDITION_SWEEP_FILE.get(condition)
    if rel is None:
        print(f"  [WARN] No sweep file mapping for condition {condition!r}")
        return None
    sweep_path = os.path.join(PROJECT_ROOT, rel)
    if not os.path.exists(sweep_path):
        print(f"  [WARN] Sweep file not found: {sweep_path}")
        return None
    with open(sweep_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("id") != qid:
                continue
            ur = rec.get("unlearning_results", {})
            epoch_data = ur.get(str(epoch)) or ur.get(epoch)
            if epoch_data is None:
                print(f"  [WARN] Epoch {epoch} not in sweep record for {qid}")
                return None
            cot_list = epoch_data.get("new_cot", [])
            return cot_list[0] if cot_list else None
    print(f"  [WARN] Question {qid!r} not found in {sweep_path}")
    return None


# ─── Embedding extraction (single forward pass) ───────────────────────────────

@torch.no_grad()
def extract_embeddings(
    model,
    tokenizer,
    cot_prompt: str,
    new_cot_text: str,
    middle_layer: int,
) -> dict:
    """
    Prefix-forced forward pass on (prompt + new_cot_text).
    Returns anchor_embeddings, conclusion_embeddings, expression spans,
    and diagnostics.
    """
    # Apply chat template to the user-turn prompt
    formatted_prompt = cot_prompt
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": cot_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    full_text = formatted_prompt + new_cot_text
    cot_start_char = len(formatted_prompt)

    encoding = tokenizer(
        full_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    device = next(model.parameters()).device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding.get(
        "attention_mask", torch.ones_like(input_ids)
    ).to(device)
    offsets: List[Tuple[int, int]] = encoding["offset_mapping"][0].tolist()
    seq_len = input_ids.shape[1]

    model.eval()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    hs = outputs.hidden_states          # tuple: (n_layers+1) x [1, seq, dim]
    n_layers = len(hs) - 1
    last_layer = n_layers

    anchor_pos = find_anchor_token_positions(full_text, offsets, cot_start_char)
    missing = [k for k in TRAJ_KEYS if anchor_pos.get(k) is None]
    if missing:
        print(f"    WARNING: missing anchor positions {missing}", flush=True)

    def get_hidden(layer_idx: int, tok: Optional[int]) -> Optional[np.ndarray]:
        if tok is None or tok < 0 or tok >= seq_len:
            return None
        return hs[layer_idx][0, tok, :].detach().float().cpu().numpy()

    # 9-point trajectory anchors (middle layer)
    anchor_embeddings = {k: get_hidden(middle_layer, anchor_pos.get(k))
                         for k in TRAJ_KEYS}

    # Conclusion probe embeddings (middle layer)
    conclusion_embeddings = {
        l: get_hidden(middle_layer, anchor_pos.get(f"Conclusion_{l}"))
        for l in "ABCD"
    }

    # Premise / Reasoning span embeddings (mean-pooled, middle + last layer)
    expression_mid: dict = {}
    expression_last: dict = {}
    for letter in "ABCD":
        ans_pos  = anchor_pos.get(f"Answer_{letter}")
        reas_pos = anchor_pos.get(f"Reasoning_{letter}")
        conc_pos = anchor_pos.get(f"Conclusion_{letter}")
        for key, s, e in [
            (f"premise_{letter}",   ans_pos,  reas_pos),
            (f"reasoning_{letter}", reas_pos, conc_pos),
        ]:
            if s is not None and e is not None and e > s + 1:
                expression_mid[key]  = span_mean(hs[middle_layer], s + 1, e)
                expression_last[key] = span_mean(hs[last_layer],   s + 1, e)
            else:
                expression_mid[key]  = None
                expression_last[key] = None

    del outputs, hs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "anchor_embeddings":    anchor_embeddings,
        "conclusion_embeddings": conclusion_embeddings,
        "expression_mid":       expression_mid,
        "expression_last":      expression_last,
        "cot_text":             new_cot_text,
        "cot_steps":            parse_cot_blocks(new_cot_text),
        "anchor_positions":     anchor_pos,
        "seq_len":              seq_len,
        "middle_layer":         middle_layer,
        "last_layer":           last_layer,
        "n_model_layers":       n_layers,
    }


# ─── Unlearning loop with in-place embedding extraction ───────────────────────

def unlearn_and_extract(
    qid: str,
    hf_model: str,
    tokenizer,
    adapted_records: List[dict],
    params: dict,
    target_epoch: int,
    cots_train: List[dict],
    new_cot_text: str,
    middle_layer: int,
    DH,
    step_idx: int = 0,
) -> dict:
    """
    Re-run NPO+KL unlearning for `qid` up to `target_epoch` epochs using
    `params` hyperparameters, then extract anchor embeddings from the
    resulting model state.

    Mirrors furV2.unlearn_single exactly, but:
      - stops after `target_epoch` training epochs (not the full sweep count)
      - extracts embeddings in-place before unloading
      - uses a seeded DataLoader generator for reproducibility
      - step_idx selects which answer block to unlearn (0=A, 1=B, 2=C, 3=D, 4=Final)
    """
    target = next((r for r in adapted_records if r["id"] == qid), None)
    if target is None:
        raise ValueError(f"Question {qid!r} not found in fur records")

    fur_args = types.SimpleNamespace(
        strategy="sentencize",
        stepwise=True,
        method=params["method"],
        lr=params["lr"],
        pos=params["pos"],
        ff2=params["ff2"],
        kl_coeff=params["kl_coeff"],
        npo_coeff=1.0,
        beta=params["beta"],
        mmlu=0,
        gsm=0,
        num_p=None,
    )

    print(f"  step_idx={step_idx}  (unlearning answer block "
          f"{['A','B','C','D','Final'][step_idx] if step_idx < 5 else step_idx})",
          flush=True)
    dataset = cot_to_otfd(
        target, cots_train, tokenizer,
        strategy=fur_args.strategy,
        stepwise=fur_args.stepwise,
        step_idx=step_idx,
        pos=fur_args.pos,
        num_p=fur_args.num_p,
    )

    NT = dataset.num_targets()
    print(f"  Unlearning targets in dataset: {NT}", flush=True)
    if NT <= 2:
        raise RuntimeError(f"Too few unlearning targets ({NT}) for {qid}")

    # Load fresh model + frozen oracle
    model = AutoModelForCausalLM.from_pretrained(
        hf_model, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map="auto",
    )
    oracle_model = AutoModelForCausalLM.from_pretrained(
        hf_model, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map="auto",
    )
    device = str(next(model.parameters()).device)
    collator = FRCollator(tokenizer, device=device)

    if fur_args.ff2:
        for name, param in model.named_parameters():
            param.requires_grad = ("mlp.down_proj.weight" in name)

    steps_per_epoch = len(dataset)
    max_steps = target_epoch * steps_per_epoch
    optimizer = torch.optim.AdamW(model.parameters(), lr=fur_args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=max_steps,
    )

    # Seeded generator for reproducible DataLoader shuffle order
    g = torch.Generator()
    g.manual_seed(params["seed"])
    train_dataloader = DataLoader(
        dataset, batch_size=1, collate_fn=collator,
        shuffle=True, generator=g,
    )

    print(f"  Training {target_epoch} epoch(s) "
          f"({steps_per_epoch} steps/epoch)...", flush=True)
    for epoch in range(target_epoch):
        model.train()
        optimizer.zero_grad()
        for batch in train_dataloader:
            loss = compute_loss(
                model, oracle_model, batch,
                loss_type=fur_args.method,
                beta=fur_args.beta,
                npo_coeff=fur_args.npo_coeff,
                KL_coeff=fur_args.kl_coeff,
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        print(f"  epoch {epoch + 1}/{target_epoch} done", flush=True)

    # Extract embeddings while model is still in memory
    print(f"  Extracting embeddings at epoch {target_epoch}...", flush=True)
    cot_prompt = target.get("cot_prompt", "")
    emb = extract_embeddings(
        model=model,
        tokenizer=tokenizer,
        cot_prompt=cot_prompt,
        new_cot_text=new_cot_text,
        middle_layer=middle_layer,
    )

    del model, oracle_model, collator, train_dataloader, dataset
    del optimizer, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return emb


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--best_conditions", default="data/best_conditions.json",
                    help="Path to best_conditions.json")
    ap.add_argument("--fur_file", default="data_tune_30/mcq_cots_fur.jsonl",
                    help="Path to the pilot MCQ CoT JSONL")
    ap.add_argument("--hf_model", default="meta-llama/Meta-Llama-3-8B-Instruct",
                    help="HuggingFace model ID")
    ap.add_argument("--middle_layer", type=int, default=14,
                    help="Transformer block index for anchor embeddings (1-indexed)")
    ap.add_argument("--output_file", default="data/fur_anchor_embeddings.pkl",
                    help="Output pickle path")
    ap.add_argument("--sanity_check_dir", default="data/pca_sanity_fur",
                    help="Directory for PCA sanity CSVs")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    def abspath(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)

    # ── Load best_conditions.json ──────────────────────────────────────────────
    best: dict = json.loads(Path(abspath(args.best_conditions)).read_text())
    print(f"[INFO] {len(best)} total entries in best_conditions file")

    # Keys can be bare question IDs ("openbook_1955") from the old format,
    # or "{qid}_step{N}" from the new per-(question, step) format.
    # Normalise to (qid, step_idx) tuples.
    def _parse_key(key: str, sel: dict):
        """Return (qid, step_idx) from a best_conditions entry."""
        if "step_idx" in sel:
            # New format: step_idx stored explicitly
            # Key is like "openbook_1955_step0" — strip suffix to recover qid
            step_idx = int(sel["step_idx"])
            suffix = f"_step{step_idx}"
            qid = key[:-len(suffix)] if key.endswith(suffix) else key
        else:
            # Old format: bare question ID, step 0 assumed
            qid      = key
            step_idx = 0
        return qid, step_idx

    valid = {}
    for key, sel in best.items():
        if not (sel.get("condition") and sel.get("epoch") is not None and sel.get("ff_hard")):
            continue
        qid, step_idx = _parse_key(key, sel)
        valid[key] = {**sel, "_qid": qid, "_step_idx": step_idx}

    print(f"[INFO] {len(valid)} (question, step) pair(s) with FF-HARD flip and valid condition")

    if not valid:
        raise SystemExit("No valid (condition, epoch) pairs found.")

    for qid, sel in valid.items():
        cond = sel["condition"]
        if cond not in CONDITION_PARAMS:
            raise ValueError(
                f"Unknown condition {cond!r} for {qid}. "
                f"Known conditions: {list(CONDITION_PARAMS)}"
            )

    # ── Load fur records and compute nocot/cot probs (base model, one pass) ──
    print(f"\n[INFO] Loading base model for prob computation: {args.hf_model}")
    fur_records = load_jsonl(abspath(args.fur_file))
    print(f"[INFO] {len(fur_records)} records in fur file")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    DH = MCQDataHandler()
    set_random_seed(42)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, torch_dtype=dtype_map[args.dtype],
        trust_remote_code=True, device_map="auto",
    )
    adapted = [
        adapt_record(r, base_model, tokenizer, DH)
        for r in tqdm(fur_records, desc="Computing nocot/cot probs")
    ]
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[INFO] Base model unloaded.\n")

    # ── Build cots_train / cots_verify (must match original sweep exactly) ────
    # Original run_fur_pilot.py uses random.seed(args.seed) then random.shuffle.
    random.seed(42)
    shuffled = list(adapted)
    random.shuffle(shuffled)
    N_verify = min(20, len(shuffled) - len(valid) - 1)
    cots_train = shuffled
    cots_verify = shuffled[-N_verify:]   # kept for reference, not used in extraction
    print(f"[INFO] cots_train={len(cots_train)}  cots_verify={len(cots_verify)}\n")

    id_to_rec = {r["id"]: r for r in adapted}

    # ── Process each (question, step, condition, epoch) ───────────────────────
    results = []
    for i, (key, sel) in enumerate(valid.items(), start=1):
        qid        = sel["_qid"]
        step_idx   = sel["_step_idx"]
        condition  = sel["condition"]
        epoch      = sel["epoch"]
        ff_soft    = sel["ff_soft"]
        ff_hard    = sel.get("ff_hard", False)
        params     = CONDITION_PARAMS[condition]

        print(f"\n[{i}/{len(valid)}] {qid}  step={step_idx}"
              f"  condition={condition}  epoch={epoch}  ff_soft={ff_soft:.4f}",
              flush=True)

        # Retrieve the stored new_cot text for this (question, step, epoch)
        new_cot_text = load_new_cot(condition, qid, epoch)
        if new_cot_text is None:
            print(f"  [SKIP] Could not load new_cot for {qid} step={step_idx} at epoch {epoch}")
            continue

        fur_rec = id_to_rec.get(qid)
        if fur_rec is None:
            print(f"  [SKIP] {qid} not found in fur file")
            continue

        try:
            emb = unlearn_and_extract(
                qid=qid,
                hf_model=args.hf_model,
                tokenizer=tokenizer,
                adapted_records=adapted,
                params=params,
                target_epoch=epoch,
                cots_train=cots_train,
                new_cot_text=new_cot_text,
                middle_layer=args.middle_layer,
                DH=DH,
                step_idx=step_idx,
            )
        except Exception as exc:
            import traceback
            print(f"  [ERROR] {exc}", flush=True)
            traceback.print_exc()
            continue

        found = sum(1 for k in TRAJ_KEYS
                    if emb["anchor_embeddings"].get(k) is not None)
        print(f"  anchor coverage: {found}/9", flush=True)

        options = fur_rec.get("options", [])
        results.append({
            "question_id":    qid,
            "question_text":  fur_rec.get("question", ""),
            "correct_letter": fur_rec.get("correct_letter", ""),
            "options":        options,
            "condition":      condition,
            "epoch":          epoch,
            "step_idx":       step_idx,
            "ff_soft":        ff_soft,
            "ff_hard":        ff_hard,
            **emb,
        })

    # ── Save output ────────────────────────────────────────────────────────────
    out_path = abspath(args.output_file)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(results, fh)
    print(f"\n[DONE] Saved {len(results)} record(s) to {out_path}", flush=True)

    # ── PCA sanity check ──────────────────────────────────────────────────────
    sanity_dir = abspath(args.sanity_check_dir)
    pca_sanity_check(results, sanity_dir)
    print(f"[DONE] PCA CSVs saved to {sanity_dir}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n[SUMMARY]")
    total_anc = sum(
        sum(1 for k in TRAJ_KEYS if r["anchor_embeddings"].get(k) is not None)
        for r in results
    )
    print(f"  Questions processed:            {len(results)}")
    print(f"  Total anchor embeddings:        {total_anc} / {len(results) * 9}")
    conc_ok = sum(
        1 for r in results for l in "ABCD"
        if r["conclusion_embeddings"].get(l) is not None
    )
    print(f"  Conclusion probe embeddings:    {conc_ok} / {len(results) * 4}")
    ff_softs = [r["ff_soft"] for r in results]
    if ff_softs:
        print(f"  FF-SOFT range:                  "
              f"{min(ff_softs):.4f} – {max(ff_softs):.4f}  "
              f"(mean {sum(ff_softs)/len(ff_softs):.4f})")


if __name__ == "__main__":
    main()
