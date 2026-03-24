#!/usr/bin/env python3
"""
extract_conclusion_colon.py

Extracts hidden-state embeddings at the ':' token of each '* Conclusion:'
header in the structured CoT — one per answer block (A/B/C/D).

Unlike patch_conclusion_embeddings.py which requires an S/R token after
the header, the colon position is always present in every record, giving
full coverage regardless of what the model generated after '* Conclusion:'.

The extracted vectors are stored under new keys that do not overwrite the
existing S/R conclusion_embeddings:
    record['conclusion_colon_L8']  = {'A': array, 'B': array, 'C': array, 'D': array}
    record['conclusion_colon_L14'] = same
    record['conclusion_colon_L28'] = same

Two outputs:
    1. data/fur_anchor_embeddings_30.pkl  — updated in-place (new keys added)
    2. data/subblock_conclusion_colon.pkl — colon embeddings for the 15
       subblock questions (not in the main pkl)

Usage:
    # Dry run: 2 records each, verify colon token, no writes
    python extract_conclusion_colon.py

    # Full run
    python extract_conclusion_colon.py --dry_run false
"""

import argparse
import gc
import glob
import json
import os
import pickle
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FUR_DIR = os.path.join(PROJECT_ROOT, "fur", "parametric-faithfulness-2")
sys.path.insert(0, FUR_DIR)
sys.path.insert(0, PROJECT_ROOT)

from extract_anchor_embeddings import char_pos_to_token_idx  # noqa: E402

ANSWER_HEADER_RE    = re.compile(r"\bAnswer\s+([A-D])\s*:", re.IGNORECASE)
CONCLUSION_COLON_RE = re.compile(r"\*?\s*Conclusion\s*:", re.IGNORECASE)
EXTRACT_LAYERS      = [8, 14, 28]


# ── Anchor-finding ─────────────────────────────────────────────────────────────

def _block_ranges(full_text: str, cot_start_char: int) -> Dict[str, Tuple[int, int]]:
    """Return {letter: (start_char, end_char)} for each answer block A-D."""
    starts: List[Tuple[str, int]] = []
    for m in ANSWER_HEADER_RE.finditer(full_text, cot_start_char):
        lt = m.group(1).upper()
        if not any(l == lt for l, _ in starts):
            starts.append((lt, m.start()))
    starts.sort(key=lambda x: x[1])
    ranges = {}
    for i, (lt, s) in enumerate(starts):
        e = starts[i + 1][1] if i + 1 < len(starts) else len(full_text)
        ranges[lt] = (s, e)
    return ranges


def find_conclusion_colon_positions(
    full_text: str,
    offsets: List[Tuple[int, int]],
    cot_start_char: int,
) -> Dict[str, Optional[int]]:
    """
    Return token index of the ':' at the end of '* Conclusion:' for each
    block A/B/C/D.  Returns None for blocks where the header is absent.
    """
    blk_ranges = _block_ranges(full_text, cot_start_char)
    positions: Dict[str, Optional[int]] = {}
    for lt in "ABCD":
        rng = blk_ranges.get(lt)
        if rng is None:
            positions[lt] = None
            continue
        blk_start, blk_end = rng
        m = CONCLUSION_COLON_RE.search(full_text, blk_start, blk_end)
        if m:
            colon_char = m.end() - 1   # ':' is the last character of the match
            positions[lt] = char_pos_to_token_idx(offsets, colon_char)
        else:
            positions[lt] = None
    return positions


# ── Forward pass ───────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_colon_embeddings(
    model,
    tokenizer,
    prompt_text: str,
    cot_text: str,
    layers: List[int],
) -> Optional[Dict[int, Dict[str, Optional[np.ndarray]]]]:
    """
    Single forward pass on (formatted_prompt + cot_text).
    Returns {layer_idx: {block_letter: ndarray or None}}.
    """
    try:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        formatted = prompt_text

    full_text      = formatted + cot_text
    cot_start_char = len(formatted)

    enc = tokenizer(
        full_text, return_tensors="pt",
        return_offsets_mapping=True, add_special_tokens=True,
        truncation=True, max_length=4096,
    )
    device    = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
    offsets: List[Tuple[int, int]] = enc["offset_mapping"][0].tolist()
    seq_len   = input_ids.shape[1]

    model.eval()
    out = model(input_ids=input_ids, attention_mask=attn_mask,
                output_hidden_states=True)
    hs  = out.hidden_states   # tuple: (n_layers+1) × [1, seq, dim]

    conc_pos = find_conclusion_colon_positions(full_text, offsets, cot_start_char)

    result: Dict[int, Dict[str, Optional[np.ndarray]]] = {}
    for ln in layers:
        if ln < 0 or ln >= len(hs):
            continue
        result[ln] = {}
        for lt in "ABCD":
            tok = conc_pos.get(lt)
            if tok is not None and 0 <= tok < seq_len:
                result[ln][lt] = hs[ln][0, tok, :].detach().float().cpu().numpy()
            else:
                result[ln][lt] = None

    del out, hs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# ── Reporting ──────────────────────────────────────────────────────────────────

def coverage_report(records: list, label: str = "") -> None:
    tag = f" [{label}]" if label else ""
    print(f"\nColon-conclusion coverage{tag}:")
    for ln in EXTRACT_LAYERS:
        ck = f"conclusion_colon_L{ln}"
        cov = {lt: sum(1 for r in records if (r.get(ck) or {}).get(lt) is not None)
               for lt in "ABCD"}
        total = sum(cov.values())
        print(f"  L{ln}: A={cov['A']} B={cov['B']} C={cov['C']} D={cov['D']}  "
              f"total={total}/{len(records) * 4}")


def verify_token_identity(model, tokenizer, rec: dict, label: str) -> None:
    """Print the decoded token at each conclusion-colon position."""
    cot_text   = rec.get("cot_text", "")
    cot_prompt = rec.get("cot_prompt", "")
    if not cot_text:
        return
    try:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": cot_prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        formatted = cot_prompt
    full_text = formatted + cot_text
    enc     = tokenizer(full_text, return_tensors="pt",
                        return_offsets_mapping=True, add_special_tokens=True,
                        truncation=True, max_length=4096)
    offsets = enc["offset_mapping"][0].tolist()
    conc_pos = find_conclusion_colon_positions(full_text, offsets, len(formatted))
    print(f"  Token identity [{label}]:")
    for lt in "ABCD":
        tok = conc_pos.get(lt)
        if tok is not None:
            tid     = enc["input_ids"][0, tok].item()
            decoded = tokenizer.decode([tid])
            print(f"    Block {lt}: tok_idx={tok}  token_id={tid}  decoded={decoded!r}")
        else:
            print(f"    Block {lt}: * Conclusion: header not found in block")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl",          default="data/fur_anchor_embeddings_30.pkl",
                    help="Main 30-question pkl to update in-place")
    ap.add_argument("--subblock_dir", default="data",
                    help="Directory with subblock_*_q*.jsonl files")
    ap.add_argument("--subblock_pkl", default="data/subblock_conclusion_colon.pkl",
                    help="Output pkl for subblock-question colon embeddings")
    ap.add_argument("--hf_model",     default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--dtype",        default="bf16",
                    choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--dry_run",      default="true",
                    help="true = 2 records only, no writes (default); false = full run")
    args = ap.parse_args()

    dry = args.dry_run.lower() not in ("false", "0", "no")

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    # ── Load main pkl ───────────────────────────────────────────────────────────
    pkl_path = os.path.join(PROJECT_ROOT, args.pkl)
    with open(pkl_path, "rb") as f:
        records = pickle.load(f)
    print(f"Loaded {len(records)} records from {pkl_path}")
    coverage_report(records, "BEFORE")

    def needs_extraction(rec):
        for ln in EXTRACT_LAYERS:
            ck = f"conclusion_colon_L{ln}"
            if any((rec.get(ck) or {}).get(lt) is None for lt in "ABCD"):
                return True
        return False

    todo_main = [r for r in records if needs_extraction(r)]
    print(f"\nMain pkl records needing extraction: {len(todo_main)}/{len(records)}")

    # ── Collect subblock CoTs ───────────────────────────────────────────────────
    subblock_cots: Dict[str, dict] = {}
    for mode in ["whole_block", "premise_only", "reasoning_only", "premise_and_reasoning"]:
        pattern = os.path.join(args.subblock_dir, f"subblock_{mode}_q*.jsonl")
        for fpath in sorted(glob.glob(pattern)):
            with open(fpath) as f:
                for line in f:
                    rec = json.loads(line.strip())
                    qid = rec.get("id")
                    if qid and qid not in subblock_cots:
                        init_cot = rec.get("initial_cot", [])
                        cot_text = init_cot[0] if isinstance(init_cot, list) else str(init_cot)
                        subblock_cots[qid] = {
                            "cot_text":   cot_text,
                            "cot_prompt": rec.get("cot_prompt", ""),
                        }
    print(f"Subblock questions found: {len(subblock_cots)}")

    if dry:
        todo_main    = todo_main[:3]
        subblock_qids = sorted(subblock_cots.keys())[:2]
        print(f"\n[DRY RUN] main: {len(todo_main)} records | subblock: {len(subblock_qids)} questions")
    else:
        subblock_qids = sorted(subblock_cots.keys())
        print(f"\n[FULL RUN] main: {len(todo_main)} records | subblock: {len(subblock_qids)} questions")

    if not todo_main and not subblock_qids:
        print("Nothing to do. Exiting.")
        return

    # ── Load model ──────────────────────────────────────────────────────────────
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    print(f"\nLoading {args.hf_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, torch_dtype=dtype_map[args.dtype],
        trust_remote_code=True, device_map="auto",
    )
    print("Model loaded.\n")

    # ── Process main pkl ────────────────────────────────────────────────────────
    # Group ALL records by question_id so we can write embeddings to every
    # step_idx variant after a single forward pass per unique qid.
    recs_by_qid: Dict[str, list] = defaultdict(list)
    for r in records:
        recs_by_qid[r["question_id"]].append(r)

    # Deduplicate todo_main to one representative record per unique question_id.
    seen_qids: set = set()
    todo_unique = []
    for rec in todo_main:
        qid = rec["question_id"]
        if qid not in seen_qids:
            seen_qids.add(qid)
            todo_unique.append(rec)
    print(f"Unique question_ids needing forward pass: {len(todo_unique)}")

    n_filled   = 0
    verified   = False

    for i, rec in enumerate(todo_unique):
        qid        = rec["question_id"]
        cot_text   = rec.get("cot_text", "")
        cot_prompt = rec.get("cot_prompt", "")
        if not cot_text:
            print(f"  [SKIP {qid}] no cot_text in record")
            continue

        if i > 0 and i % 5 == 0:
            print(f"  --- progress: {i}/{len(todo_unique)} unique qids done ---")

        print(f"  {qid}: extracting ...")
        emb = extract_colon_embeddings(model, tokenizer, cot_prompt, cot_text, EXTRACT_LAYERS)
        if emb is None:
            print(f"  [FAIL {qid}]")
            continue

        if not verified:
            verify_token_identity(model, tokenizer, rec, qid)
            # Also print a cosine similarity sanity check at L14
            ae = rec.get("anchor_embeddings", {})
            v_reas_A = ae.get("Reasoning_A")
            v_conc_A = (emb.get(14) or {}).get("A")
            if v_reas_A is not None and v_conc_A is not None:
                v_r = v_reas_A.astype(np.float32)
                v_c = v_conc_A.astype(np.float32)
                cos = float(np.dot(v_r, v_c) / (np.linalg.norm(v_r) * np.linalg.norm(v_c) + 1e-12))
                print(f"  Sanity check — cosine(Reasoning_A, conclusion_colon_A) at L14: {cos:.4f}")
            verified = True

        # Write to ALL records sharing this question_id (covers multiple step_idx variants).
        for target in recs_by_qid[qid]:
            for ln, blk_dict in emb.items():
                ck = f"conclusion_colon_L{ln}"
                if ck not in target:
                    target[ck] = {lt: None for lt in "ABCD"}
                for lt in "ABCD":
                    if target[ck].get(lt) is None and (blk_dict.get(lt) is not None):
                        target[ck][lt] = blk_dict[lt]
                        n_filled += 1

    coverage_report(records, "AFTER main")
    print(f"New slots filled in main pkl: {n_filled}")

    # ── Process subblock questions ──────────────────────────────────────────────
    subblock_out: Dict[str, dict] = {}
    sb_verified  = False

    for qid in subblock_qids:
        d          = subblock_cots[qid]
        cot_text   = d["cot_text"]
        cot_prompt = d["cot_prompt"]
        print(f"  Subblock {qid}: extracting ...")
        emb = extract_colon_embeddings(model, tokenizer, cot_prompt, cot_text, EXTRACT_LAYERS)
        if emb is None:
            print(f"  [FAIL {qid}]")
            continue

        if not sb_verified:
            # Build a temporary fake rec for verification
            verify_token_identity(model, tokenizer,
                                  {"cot_text": cot_text, "cot_prompt": cot_prompt}, qid)
            sb_verified = True

        subblock_out[qid] = {f"conclusion_colon_L{ln}": emb[ln] for ln in emb}
        cov = {ln: sum(1 for lt in "ABCD" if (emb.get(ln) or {}).get(lt) is not None)
               for ln in EXTRACT_LAYERS}
        print(f"    coverage: {cov}")

    print(f"\nSubblock questions processed: {len(subblock_out)}")

    # ── Write files (full run only) ─────────────────────────────────────────────
    if dry:
        print("\n[DRY RUN] No files written.")
        print("Re-run with --dry_run false to apply.")
    else:
        with open(pkl_path, "wb") as f:
            pickle.dump(records, f)
        print(f"\n[DONE] Main pkl updated: {pkl_path}  ({n_filled} new slots)")

        sb_pkl_path = os.path.join(PROJECT_ROOT, args.subblock_pkl)
        with open(sb_pkl_path, "wb") as f:
            pickle.dump(subblock_out, f)
        print(f"[DONE] Subblock pkl saved: {sb_pkl_path}  ({len(subblock_out)} questions)")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
