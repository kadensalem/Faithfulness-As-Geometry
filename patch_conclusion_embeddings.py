#!/usr/bin/env python3
"""
patch_conclusion_embeddings.py

Targeted re-extraction of conclusion token embeddings for records in
fur_anchor_embeddings_30.pkl that have missing conclusion_embeddings.

For each record with missing data:
  1. Checks whether the stored cot_text contains an S or R token after
     '* Conclusion:' in each answer block (A/B/C/D).
  2. If found, re-runs a forward pass on (cot_prompt + cot_text) and
     extracts the conclusion token hidden state at all 5 layers
     (L4, L8, L14, L20, L28).
  3. Updates conclusion_embeddings_L{N} in-place for each layer.
     Never overwrites existing non-None values or anchor_embeddings.

Dry-run mode (--dry_run, default True):
  Processes first 3 eligible records, prints before/after coverage,
  does NOT write to disk.

Full run (--dry_run false):
  Processes all eligible records and overwrites the pkl.

Usage:
    # Dry-run (safe — no writes)
    python patch_conclusion_embeddings.py --pkl data/fur_anchor_embeddings_30.pkl

    # Full run (writes to pkl)
    python patch_conclusion_embeddings.py --pkl data/fur_anchor_embeddings_30.pkl --dry_run false
"""

import argparse
import gc
import json
import os
import pickle
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── Project path setup ───────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FUR_DIR = os.path.join(PROJECT_ROOT, "fur", "parametric-faithfulness-2")
sys.path.insert(0, FUR_DIR)
sys.path.insert(0, PROJECT_ROOT)

from extract_anchor_embeddings import find_anchor_token_positions  # noqa: E402

TRAJ_KEYS = [
    "Answer_A", "Reasoning_A", "Answer_B", "Reasoning_B",
    "Answer_C", "Reasoning_C", "Answer_D", "Reasoning_D", "Final_Answer",
]
CONCLUSION_RE = re.compile(r"\*?\s*Conclusion\s*:\s*([SR])\b", re.IGNORECASE)


def check_conclusion_in_cot(cot_text: str) -> Dict[str, bool]:
    """Return which blocks (A/B/C/D) have an S/R conclusion token in the CoT."""
    # Parse block boundaries (very lightweight — just look for * Answer X: headers)
    block_re = re.compile(r"\*\s*Answer\s*([ABCD])\s*:", re.IGNORECASE)
    positions = {}
    for m in block_re.finditer(cot_text):
        positions[m.group(1).upper()] = m.start()

    results = {}
    for lt in "ABCD":
        start = positions.get(lt)
        if start is None:
            results[lt] = False
            continue
        # End = start of next block or end of text
        next_starts = [v for k, v in positions.items() if v > start]
        end = min(next_starts) if next_starts else len(cot_text)
        block_text = cot_text[start:end]
        results[lt] = bool(CONCLUSION_RE.search(block_text))
    return results


@torch.no_grad()
def extract_conclusion_embeddings(
    model,
    tokenizer,
    cot_prompt: str,
    cot_text: str,
    layers: List[int],
) -> Optional[Dict[int, Dict[str, Optional[np.ndarray]]]]:
    """
    Forward pass on (cot_prompt + cot_text).
    Returns {layer_idx: {block_letter: hidden_state_array or None}}.
    Returns None if tokenisation fails.
    """
    formatted = cot_prompt
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": cot_prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            pass

    full_text = formatted + cot_text
    cot_start_char = len(formatted)

    enc = tokenizer(
        full_text, return_tensors="pt",
        return_offsets_mapping=True, add_special_tokens=True,
    )
    device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
    offsets: List[Tuple[int, int]] = enc["offset_mapping"][0].tolist()
    seq_len = input_ids.shape[1]

    model.eval()
    outputs = model(input_ids=input_ids, attention_mask=attn_mask,
                    output_hidden_states=True)
    hs = outputs.hidden_states   # (n_layers+1) × [1, seq, dim]

    anchor_pos = find_anchor_token_positions(full_text, offsets, cot_start_char)

    def get_h(layer_idx, tok):
        if tok is None or tok < 0 or tok >= seq_len:
            return None
        return hs[layer_idx][0, tok, :].detach().float().cpu().numpy()

    result = {}
    for ln in layers:
        if ln < 0 or ln >= len(hs):
            continue
        result[ln] = {
            lt: get_h(ln, anchor_pos.get(f"Conclusion_{lt}"))
            for lt in "ABCD"
        }

    del outputs, hs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def coverage_report(records: list, label: str = "") -> None:
    layers_present = sorted(set(
        int(k.split("_L")[1])
        for r in records for k in r if k.startswith("conclusion_embeddings_L")
    ))
    tag = f" [{label}]" if label else ""
    print(f"\nConclusion coverage{tag}:")
    for ln in layers_present:
        ck = f"conclusion_embeddings_L{ln}"
        cov = {lt: sum(1 for r in records if r.get(ck, {}).get(lt) is not None)
               for lt in "ABCD"}
        total = sum(cov.values())
        print(f"  L{ln}: A={cov['A']} B={cov['B']} C={cov['C']} D={cov['D']}  "
              f"total={total}/{len(records)*4}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl",      default="data/fur_anchor_embeddings_30.pkl")
    ap.add_argument("--hf_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--dtype",    default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--dry_run",  default="true",
                    help="true = process first 3 records, no writes (default); false = full run")
    args = ap.parse_args()

    dry = args.dry_run.lower() not in ("false", "0", "no")

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    # ── Load pkl ───────────────────────────────────────────────────────────────
    pkl_path = os.path.join(PROJECT_ROOT, args.pkl)
    with open(pkl_path, "rb") as f:
        records = pickle.load(f)
    print(f"Loaded {len(records)} records from {pkl_path}")

    # ── Coverage before ────────────────────────────────────────────────────────
    coverage_report(records, "BEFORE")

    layers_in_pkl = sorted(set(
        int(k.split("_L")[1])
        for r in records for k in r if k.startswith("anchor_embeddings_L")
    ))
    print(f"\nLayer keys present: {layers_in_pkl}")

    # ── Identify records with any missing conclusion embeddings ────────────────
    def has_missing(rec):
        for ln in layers_in_pkl:
            ck = f"conclusion_embeddings_L{ln}"
            if any(rec.get(ck, {}).get(lt) is None for lt in "ABCD"):
                return True
        return False

    missing_recs = [r for r in records if has_missing(r)]
    print(f"\nRecords with any missing conclusion embeddings: {len(missing_recs)}/{len(records)}")

    # ── Diagnose: which missing records have S/R in their cot_text? ────────────
    recoverable = []
    for rec in missing_recs:
        cot = rec.get("cot_text", "")
        sr_blocks = check_conclusion_in_cot(cot)
        if any(sr_blocks.values()):
            recoverable.append((rec, sr_blocks))

    print(f"Records with S/R token in cot_text (recoverable): {len(recoverable)}/{len(missing_recs)}")

    if not recoverable:
        print("\nDIAGNOSIS: All missing records have free-text conclusions (no S/R token).")
        print("  The model generated narrative conclusions rather than structured S/R tokens.")
        print("  Examples of what follows '* Conclusion:' in missing records:")
        word_re = re.compile(r"\*?\s*Conclusion\s*:\s*(\S+)")
        word_counts: Dict[str, int] = {}
        for r in missing_recs[:30]:
            for m in word_re.findall(r.get("cot_text", "")):
                word_counts[m] = word_counts.get(m, 0) + 1
        for w, c in sorted(word_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    {w!r:25s}: {c}")
        print("\nConclusion: re-extraction will not increase coverage beyond "
              f"{len(records) - len(missing_recs)}/{len(records)}.")
        print("No changes made to pkl.")
        return

    # ── Limit to first 3 if dry run ────────────────────────────────────────────
    if dry:
        recoverable = recoverable[:3]
        print(f"\n[DRY RUN] Processing first {len(recoverable)} recoverable record(s)...")
    else:
        print(f"\n[FULL RUN] Processing {len(recoverable)} recoverable record(s)...")

    # ── Load model ─────────────────────────────────────────────────────────────
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    print(f"\nLoading model: {args.hf_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, torch_dtype=dtype_map[args.dtype],
        trust_remote_code=True, device_map="auto",
    )

    # ── Process recoverable records ────────────────────────────────────────────
    id_to_rec = {r["question_id"]: r for r in records}
    n_filled = 0

    for rec, sr_blocks in recoverable:
        qid = rec["question_id"]
        cot_prompt = rec.get("anchor_positions", {})   # may not have cot_prompt stored
        # cot_prompt is not always stored in the record; try to reconstruct
        cot_prompt_text = rec.get("cot_prompt", "")
        if not cot_prompt_text:
            print(f"  [WARN] {qid}: no cot_prompt in record — using empty prompt")

        cot_text = rec.get("cot_text", "")
        if not cot_text:
            print(f"  [SKIP] {qid}: no cot_text stored")
            continue

        print(f"\n  {qid}: extracting conclusion embeddings for blocks "
              f"{[lt for lt, ok in sr_blocks.items() if ok]}")

        emb_by_layer = extract_conclusion_embeddings(
            model=model, tokenizer=tokenizer,
            cot_prompt=cot_prompt_text, cot_text=cot_text,
            layers=layers_in_pkl,
        )
        if emb_by_layer is None:
            print(f"  [SKIP] {qid}: extraction failed")
            continue

        target_rec = id_to_rec[qid]
        for ln, conc_dict in emb_by_layer.items():
            ck = f"conclusion_embeddings_L{ln}"
            if ck not in target_rec:
                target_rec[ck] = {lt: None for lt in "ABCD"}
            for lt in "ABCD":
                if sr_blocks.get(lt) and conc_dict.get(lt) is not None:
                    # Only fill if we expected an S/R token AND got a vector
                    if target_rec[ck].get(lt) is None:
                        target_rec[ck][lt] = conc_dict[lt]
                        n_filled += 1
                        print(f"    Filled: L{ln} block {lt}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Coverage after ─────────────────────────────────────────────────────────
    coverage_report(records, "AFTER")
    print(f"\nNew conclusion embedding slots filled: {n_filled}")

    # ── Write pkl (full run only) ──────────────────────────────────────────────
    if dry:
        print("\n[DRY RUN] No changes written to disk.")
        print("Re-run with --dry_run false to apply.")
    else:
        with open(pkl_path, "wb") as f:
            pickle.dump(records, f)
        print(f"\n[DONE] Updated pkl written to {pkl_path}")


if __name__ == "__main__":
    main()
