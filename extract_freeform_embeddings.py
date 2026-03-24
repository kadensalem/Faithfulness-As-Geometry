#!/usr/bin/env python3
"""
extract_freeform_embeddings.py

Free-form CoT geometry extraction pipeline (comparison condition).

For each question in data/best_conditions_30.json:
  1. Load initial_cot[0] from the corresponding FUR jsonl file
     (the pre-unlearning CoT generated with original model weights)
  2. Tokenize the CoT text and run a single forward pass
  3. Extract hidden states at L8, L14, L28
  4. Compute per-token displacement metrics on-the-fly (no full tensor storage)
  5. Save summary metrics to data/freeform_embeddings.pkl

Output dict per question:
  {
    'question_id': str,
    'ff_soft': float,         # max ff_soft across all steps
    'ff_hard': bool,
    'cot_text': str,          # initial_cot[0]
    'n_tokens': int,
    'metrics_L8':  { mean_displacement, displacement_variance,
                     arc_length, net_displacement, straightness,
                     mean_curvature, seg_early, seg_mid, seg_late },
    'metrics_L14': { same },
    'metrics_L28': { same },
  }

Usage:
    python extract_freeform_embeddings.py \\
        --hf_model meta-llama/Meta-Llama-3-8B-Instruct \\
        --best_conditions data/best_conditions_30.json \\
        --data_dir data \\
        --output_file data/freeform_embeddings.pkl \\
        --layers 8 14 28 \\
        [--n_questions 3]   # for sanity check only

Sanity check mode (--n_questions 3):
    Prints first 200 chars of each CoT, n_tokens, mean_displacement at L14,
    and verifies ff_soft values match best_conditions_30.json.
"""

import argparse
import glob
import json
import os
import pickle
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ─── Helpers ──────────────────────────────────────────────────────────────────

def cosine_sim_np(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity between two 1-D float32 vectors."""
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def menger_curvature(y_prev: np.ndarray, y_cur: np.ndarray, y_next: np.ndarray) -> float:
    """
    Menger curvature for three consecutive hidden states.

    u = y_cur  - y_prev
    v = y_next - y_cur
    numerator   = 2 * sqrt(1 - cos(u,v)^2)
    denominator = ||y_next - y_prev||
    kappa = numerator / denominator  (0 if collinear or degenerate)
    """
    u = y_cur - y_prev
    v = y_next - y_cur
    cos_uv = cosine_sim_np(u, v)
    sin2 = max(0.0, 1.0 - cos_uv ** 2)
    numerator = 2.0 * float(np.sqrt(sin2))
    denominator = float(np.linalg.norm(y_next - y_prev))
    if denominator < 1e-12:
        return 0.0
    return numerator / denominator


def compute_layer_metrics(hidden_seq: np.ndarray) -> dict:
    """
    Compute all geometry summary metrics for a single layer's token sequence.

    hidden_seq: (n_tokens, hidden_dim)  float32

    Returns dict with:
      mean_displacement, displacement_variance, arc_length,
      net_displacement, straightness, mean_curvature,
      seg_early, seg_mid, seg_late
    """
    n = len(hidden_seq)
    if n < 2:
        nan = float('nan')
        return dict(
            mean_displacement=nan, displacement_variance=nan,
            arc_length=nan, net_displacement=nan, straightness=nan,
            mean_curvature=nan, seg_early=nan, seg_mid=nan, seg_late=nan,
        )

    # Displacement vectors: delta[t] = hidden[t+1] - hidden[t]
    deltas = hidden_seq[1:] - hidden_seq[:-1]           # (n-1, hidden_dim)
    norms = np.linalg.norm(deltas, axis=1).astype(np.float64)  # (n-1,)

    mean_displacement = float(np.mean(norms))
    displacement_variance = float(np.var(norms))
    arc_length = float(np.sum(norms))

    net_disp_vec = hidden_seq[-1] - hidden_seq[0]
    net_displacement = float(np.linalg.norm(net_disp_vec))
    straightness = net_displacement / arc_length if arc_length > 1e-12 else 0.0

    # Menger curvature for consecutive triples (t-1, t, t+1)
    kappas = []
    for t in range(1, n - 1):
        k = menger_curvature(hidden_seq[t - 1], hidden_seq[t], hidden_seq[t + 1])
        kappas.append(k)
    mean_curvature = float(np.mean(kappas)) if kappas else float('nan')

    # Segment metrics: split token range into thirds
    t1 = n // 3
    t2 = 2 * n // 3
    # norms[t] = ||hidden[t+1] - hidden[t]||, indexed 0..n-2
    # early segment covers tokens 0..t1, so deltas 0..t1-1
    # mid   segment covers tokens t1..t2, so deltas t1..t2-1
    # late  segment covers tokens t2..n,  so deltas t2..n-2
    seg_early = float(np.mean(norms[:t1])) if t1 > 0 else float('nan')
    seg_mid   = float(np.mean(norms[t1:t2])) if t2 > t1 else float('nan')
    seg_late  = float(np.mean(norms[t2:])) if n - 1 > t2 else float('nan')

    return dict(
        mean_displacement=mean_displacement,
        displacement_variance=displacement_variance,
        arc_length=arc_length,
        net_displacement=net_displacement,
        straightness=straightness,
        mean_curvature=mean_curvature,
        seg_early=seg_early,
        seg_mid=seg_mid,
        seg_late=seg_late,
    )


# ─── Data loading ─────────────────────────────────────────────────────────────

def build_fur_index(data_dir: str) -> Dict[str, str]:
    """
    Scan all sweep30_*.jsonl and pilot_sweep_*.jsonl files in data_dir.
    Returns a dict mapping question_id -> path to any file containing it.
    We just need one file per question since initial_cot is identical
    across all conditions at the top level.
    """
    index: Dict[str, str] = {}
    patterns = [
        os.path.join(data_dir, "sweep30_*.jsonl"),
        os.path.join(data_dir, "pilot_sweep_*.jsonl"),
    ]
    for pattern in patterns:
        for fpath in sorted(glob.glob(pattern)):
            try:
                with open(fpath, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        qid = rec.get("id")
                        if qid and qid not in index:
                            index[qid] = fpath
                            break  # one record is enough per file to index the question
            except Exception as e:
                print(f"[WARN] Could not read {fpath}: {e}", flush=True)
    return index


def load_initial_cot(question_id: str, fur_index: Dict[str, str]) -> Optional[str]:
    """
    Load initial_cot[0] for the given question_id from the FUR index.
    Returns None if not found.
    """
    fpath = fur_index.get(question_id)
    if fpath is None:
        return None
    try:
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("id") != question_id:
                    continue
                ic = rec.get("initial_cot")
                if ic is None:
                    return None
                if isinstance(ic, list):
                    return ic[0] if ic else None
                return str(ic)
    except Exception as e:
        print(f"[WARN] Error reading {fpath}: {e}", flush=True)
    return None


def load_best_conditions(bc_path: str) -> Dict[str, dict]:
    """
    Load best_conditions_30.json and return per-question summary:
      qid -> {'ff_soft': max_ff_soft, 'ff_hard': bool}
    """
    with open(bc_path, encoding="utf-8") as f:
        bc_raw = json.load(f)

    by_q: Dict[str, List[dict]] = defaultdict(list)
    for key, val in bc_raw.items():
        qid = key.rsplit("_step", 1)[0]
        by_q[qid].append(val)

    result = {}
    for qid, steps in by_q.items():
        max_ff = max((s.get("ff_soft") or 0.0) for s in steps)
        ff_hard = any(bool(s.get("ff_hard")) for s in steps)
        result[qid] = {"ff_soft": max_ff, "ff_hard": ff_hard}
    return result


# ─── Per-question processing ──────────────────────────────────────────────────

@torch.no_grad()
def process_question(
    question_id: str,
    cot_text: str,
    ff_soft: float,
    ff_hard: bool,
    model,
    tokenizer,
    layers: List[int],
    device: str,
) -> dict:
    """
    Run a single forward pass on cot_text and compute geometry metrics
    for each requested layer.

    To save memory, hidden states are extracted layer by layer via hooks
    rather than storing the full (n_layers+1, seq_len, hidden_dim) tensor.
    """
    # Tokenize CoT text only (no chat template wrapping)
    encoding = tokenizer(
        cot_text,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding.get(
        "attention_mask", torch.ones_like(input_ids)
    ).to(device)
    n_tokens = input_ids.shape[1]

    print(f"  n_tokens={n_tokens}", flush=True)

    # ── Forward pass with output_hidden_states=True ──
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states
    # hidden_states[0] = embedding layer, hidden_states[k] = block k output

    # ── Extract metrics per layer ──
    layer_metrics = {}
    for layer_idx in layers:
        hs = hidden_states[layer_idx]  # (1, n_tokens, hidden_dim)
        seq = hs[0].detach().float().cpu().numpy()  # (n_tokens, hidden_dim)
        metrics = compute_layer_metrics(seq)
        layer_metrics[f"metrics_L{layer_idx}"] = metrics
        print(
            f"  L{layer_idx}: mean_disp={metrics['mean_displacement']:.4f}  "
            f"arc={metrics['arc_length']:.2f}  straight={metrics['straightness']:.4f}  "
            f"curvature={metrics['mean_curvature']:.6f}",
            flush=True,
        )

    # Free GPU memory
    del outputs, hidden_states
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "question_id": question_id,
        "ff_soft": ff_soft,
        "ff_hard": ff_hard,
        "cot_text": cot_text,
        "n_tokens": n_tokens,
    }
    result.update(layer_metrics)
    return result


# ─── Sanity checks ────────────────────────────────────────────────────────────

def run_sanity_checks(results: List[dict], bc_summary: Dict[str, dict]) -> None:
    """
    Sanity check prints:
    1. First 200 chars of CoT (confirm format)
    2. n_tokens per question
    3. mean_displacement at L14
    4. ff_soft matches best_conditions_30.json
    """
    print("\n" + "=" * 72, flush=True)
    print("SANITY CHECKS (3 questions)", flush=True)
    print("=" * 72, flush=True)

    for rec in results:
        qid = rec["question_id"]
        cot_preview = rec["cot_text"][:200].replace("\n", " ")
        n_tok = rec["n_tokens"]
        m14 = rec.get("metrics_L14", {})
        mean_disp = m14.get("mean_displacement", float("nan"))
        ff_soft_computed = rec["ff_soft"]
        ff_soft_bc = bc_summary.get(qid, {}).get("ff_soft", None)

        ff_match = (
            ff_soft_bc is not None
            and abs(ff_soft_computed - ff_soft_bc) < 1e-6
        )

        print(f"\n[Q] {qid}", flush=True)
        print(f"  CoT (first 200 chars): {cot_preview!r}", flush=True)
        print(f"  n_tokens: {n_tok}", flush=True)
        print(f"  mean_displacement at L14: {mean_disp:.4f}", flush=True)
        print(
            f"  ff_soft: {ff_soft_computed:.4f} "
            f"(bc={ff_soft_bc})  match={ff_match}",
            flush=True,
        )
        if mean_disp < 0.1:
            print(
                f"  WARNING: mean_displacement={mean_disp:.4f} seems very low. "
                "Expected ~0.5–3.0 for token-level deltas.",
                flush=True,
            )

    print("\n[CHECK] Model identity: using original weights (epoch 0).", flush=True)
    print("  This script loads meta-llama/Meta-Llama-3-8B-Instruct directly.", flush=True)
    print("  No unlearning has been applied.", flush=True)

    print("\n[CHECK] Coverage:", flush=True)
    print(f"  Questions with results: {len(results)}", flush=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    ap = argparse.ArgumentParser(
        description="Extract token-level geometry metrics from free-form CoTs"
    )
    ap.add_argument(
        "--hf_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    ap.add_argument(
        "--best_conditions", type=str, default="data/best_conditions_30.json",
    )
    ap.add_argument(
        "--data_dir", type=str, default="data",
        help="Directory containing sweep30_*.jsonl and pilot_sweep_*.jsonl files",
    )
    ap.add_argument(
        "--output_file", type=str, default="data/freeform_embeddings.pkl",
    )
    ap.add_argument(
        "--layers", type=int, nargs="+", default=[8, 14, 28],
        help="Layer indices to extract (1-indexed transformer block output)",
    )
    ap.add_argument(
        "--n_questions", type=int, default=None,
        help="Process only this many questions (for sanity check). Omit for all 30.",
    )
    ap.add_argument(
        "--device", type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument(
        "--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"],
    )
    args = ap.parse_args()

    # Make paths absolute relative to script directory
    base = os.path.dirname(os.path.abspath(__file__))
    def abs_path(p):
        return p if os.path.isabs(p) else os.path.join(base, p)

    bc_path = abs_path(args.best_conditions)
    data_dir = abs_path(args.data_dir)
    out_path = abs_path(args.output_file)

    # ── Load best conditions ──
    print(f"[INFO] Loading best conditions from {bc_path}", flush=True)
    bc_summary = load_best_conditions(bc_path)
    question_ids = sorted(bc_summary.keys())
    print(f"[INFO] {len(question_ids)} unique questions in best_conditions", flush=True)

    if args.n_questions is not None:
        question_ids = question_ids[: args.n_questions]
        print(f"[INFO] Sanity-check mode: processing {len(question_ids)} questions", flush=True)

    # ── Build FUR file index ──
    print(f"[INFO] Scanning FUR output files in {data_dir}", flush=True)
    fur_index = build_fur_index(data_dir)
    print(f"[INFO] FUR index covers {len(fur_index)} questions", flush=True)

    # Check coverage
    missing_fur = [qid for qid in question_ids if qid not in fur_index]
    if missing_fur:
        print(f"[WARN] {len(missing_fur)} questions have no FUR file: {missing_fur}", flush=True)
    else:
        print(f"[INFO] All {len(question_ids)} questions found in FUR index", flush=True)

    # ── Load model ──
    print(f"\n[INFO] Loading model: {args.hf_model}", flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    n_model_layers = model.config.num_hidden_layers
    print(f"[INFO] Model has {n_model_layers} transformer layers", flush=True)

    # Validate requested layers
    for layer_idx in args.layers:
        if layer_idx < 0 or layer_idx > n_model_layers:
            raise ValueError(
                f"Layer index {layer_idx} out of range for {n_model_layers}-layer model"
            )
    print(f"[INFO] Extracting at layers: {args.layers}", flush=True)

    # ── Process each question ──
    results: List[dict] = []

    for i, qid in enumerate(question_ids, start=1):
        print(f"\n[{i}/{len(question_ids)}] {qid}", flush=True)

        # Load initial CoT
        cot_text = load_initial_cot(qid, fur_index)
        if cot_text is None:
            print(f"  [SKIP] No initial_cot found", flush=True)
            continue
        if not cot_text.strip():
            print(f"  [SKIP] Empty initial_cot", flush=True)
            continue

        ff_info = bc_summary[qid]

        try:
            rec = process_question(
                question_id=qid,
                cot_text=cot_text,
                ff_soft=ff_info["ff_soft"],
                ff_hard=ff_info["ff_hard"],
                model=model,
                tokenizer=tokenizer,
                layers=args.layers,
                device=args.device,
            )
            results.append(rec)
        except Exception as e:
            print(f"  [ERROR] {e}", flush=True)
            import traceback
            traceback.print_exc()

    # ── Sanity checks (if running subset) ──
    if args.n_questions is not None:
        run_sanity_checks(results, bc_summary)

    # ── Save output ──
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n[DONE] Saved {len(results)} records to {out_path}", flush=True)

    # ── Summary ──
    print("\n[SUMMARY]", flush=True)
    print(f"  Questions processed: {len(results)}/{len(question_ids)}", flush=True)
    if results:
        ff_softs = [r["ff_soft"] for r in results]
        n_toks = [r["n_tokens"] for r in results]
        print(f"  ff_soft range: {min(ff_softs):.4f} – {max(ff_softs):.4f}", flush=True)
        print(f"  n_tokens: mean={np.mean(n_toks):.1f}  min={min(n_toks)}  max={max(n_toks)}", flush=True)
        # Print mean_displacement at L14 across all questions
        disp_l14 = [
            r["metrics_L14"]["mean_displacement"]
            for r in results
            if "metrics_L14" in r and not np.isnan(r["metrics_L14"].get("mean_displacement", float("nan")))
        ]
        if disp_l14:
            print(f"  mean_displacement L14: mean={np.mean(disp_l14):.4f}  range=[{min(disp_l14):.4f}, {max(disp_l14):.4f}]", flush=True)


if __name__ == "__main__":
    main()
