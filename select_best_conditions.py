"""
select_best_conditions.py

Reads all 4 sweep output files and selects the best (condition, epoch) pair
per question for anchor embedding extraction.

Selection criterion (applied in order):
  Step 1 — FF-HARD filter: prediction must flip from epoch 0.
  Step 2 — Structure quality filter: new_cot must pass >= 2 of 3 checks
            in >= 2 of 4 answer blocks.
              Check A: TTR (unique tokens / total) in * Reasoning: slot >= 0.4
              Check B: * Reasoning: slot >= 8 whitespace-split tokens
              Check C: * Conclusion: slot first non-whitespace token is S or R
  Step 3 — Earliest clean epoch preferred.
  Step 4 — Tiebreak on lowest specificity change (fewest of 20 held-out
            predictions changed from epoch 0).

Output: data/best_conditions.json + stdout summary table.

Usage:
  python select_best_conditions.py
  python select_best_conditions.py --sweep_files data/sweep_verify.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Default sweep files
# ---------------------------------------------------------------------------

DEFAULT_SWEEP_FILES = {
    "sweep_1": "data/pilot_sweep_1.jsonl",
    "sweep_2": "data/pilot_sweep_2.jsonl",
    "sweep_3": "data/pilot_sweep_3.jsonl",
    "sweep_4": "data/pilot_sweep_4.jsonl",
}

LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}
IDX_TO_LETTER = {v: k for k, v in LETTER_TO_IDX.items()}


# ---------------------------------------------------------------------------
# CoT structure parsing
# ---------------------------------------------------------------------------

def _extract_slot(cot: str, slot_marker: str, end_markers: list) -> str:
    """Extract text following slot_marker up to the first end_marker found."""
    idx = cot.find(slot_marker)
    if idx == -1:
        return ""
    start = idx + len(slot_marker)
    end = len(cot)
    for em in end_markers:
        pos = cot.find(em, start)
        if pos != -1 and pos < end:
            end = pos
    return cot[start:end].strip()


def parse_answer_blocks(cot: str):
    """
    Return a list of dicts, one per answer block found (A–D).
    Each dict has keys: letter, premise, reasoning, conclusion.
    """
    blocks = []
    # Split on Answer X: boundaries
    parts = re.split(r"(?=\bAnswer\s+[A-D]\s*:)", cot)
    for part in parts:
        m = re.match(r"Answer\s+([A-D])\s*:", part)
        if not m:
            continue
        letter = m.group(1)
        # Reasoning slot: from "* Reasoning:" to "* Conclusion:" or next "Answer"
        reasoning = _extract_slot(
            part, "* Reasoning:",
            ["* Conclusion:", "Answer A:", "Answer B:", "Answer C:", "Answer D:", "**Final Answer**"]
        )
        # Conclusion slot: from "* Conclusion:" to next "Answer" or end
        conclusion = _extract_slot(
            part, "* Conclusion:",
            ["Answer A:", "Answer B:", "Answer C:", "Answer D:", "**Final Answer**"]
        )
        blocks.append({"letter": letter, "reasoning": reasoning, "conclusion": conclusion})
    return blocks


# ---------------------------------------------------------------------------
# Structure quality checks
# ---------------------------------------------------------------------------

def check_ttr(reasoning: str, threshold: float = 0.4) -> bool:
    """Check A: type-token ratio of reasoning slot >= threshold."""
    tokens = reasoning.split()
    if not tokens:
        return False
    return len(set(tokens)) / len(tokens) >= threshold


def check_min_length(reasoning: str, min_tokens: int = 8) -> bool:
    """Check B: reasoning slot has >= min_tokens whitespace-split tokens."""
    return len(reasoning.split()) >= min_tokens


def check_conclusion(conclusion: str) -> bool:
    """Check C: first non-whitespace token in conclusion is S or R."""
    tokens = conclusion.strip().split()
    if not tokens:
        return False
    first = tokens[0].rstrip(".,;:")
    return first in ("S", "R")


def structure_quality(cot: str):
    """
    Return (passes: bool, blocks_passing: int, checks_summary: str).
    A CoT passes if >= 2 of 3 checks pass in >= 2 of 4 answer blocks.
    """
    blocks = parse_answer_blocks(cot)
    if not blocks:
        return False, 0, "no blocks parsed"

    passing_blocks = 0
    check_details = []
    for b in blocks:
        a = check_ttr(b["reasoning"])
        bl = check_min_length(b["reasoning"])
        c = check_conclusion(b["conclusion"])
        n_pass = sum([a, bl, c])
        passes = n_pass >= 2
        if passes:
            passing_blocks += 1
        check_details.append(f"{b['letter']}:{'✓' if a else '✗'}{'✓' if bl else '✗'}{'✓' if c else '✗'}")

    overall = passing_blocks >= 2
    return overall, passing_blocks, " ".join(check_details)


# ---------------------------------------------------------------------------
# Specificity change
# ---------------------------------------------------------------------------

def specificity_delta(epoch0_preds: list, epochN_preds: list) -> int:
    """Count how many of the held-out predictions changed from epoch 0."""
    return sum(1 for a, b in zip(epoch0_preds, epochN_preds) if a != b)


# ---------------------------------------------------------------------------
# Load sweep data
# ---------------------------------------------------------------------------

def load_sweep_file(path: str, condition_name: str) -> dict:
    """Load a sweep JSONL and index by question id. Returns {qid: record}."""
    records = {}
    p = Path(path)
    if not p.exists():
        print(f"  [missing] {path} — skipping {condition_name}")
        return records
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["_condition"] = condition_name
            records[rec["id"]] = rec
    print(f"  [loaded] {path}: {len(records)} record(s)")
    return records


# ---------------------------------------------------------------------------
# Per-question selection
# ---------------------------------------------------------------------------

def select_for_question(qid: str, candidates: list) -> dict:
    """
    candidates: list of dicts with keys:
      condition, epoch (int), probs, prediction, new_cot, specificity_preds,
      correct_idx, epoch0_prediction, epoch0_specificity_preds, initial_probs

    Returns selection dict.
    """
    correct_idx = candidates[0]["correct_idx"]
    epoch0_pred = candidates[0]["epoch0_prediction"]
    initial_probs = candidates[0]["initial_probs"]
    ff_soft_initial = initial_probs[correct_idx]

    # Step 1: FF-HARD filter
    flipped = [c for c in candidates if c["prediction"] != epoch0_pred]

    if not flipped:
        return {
            "condition": None,
            "epoch": None,
            "ff_hard": False,
            "ff_soft": round(ff_soft_initial, 4),
            "structure_checks_passed": None,
            "specificity_delta": None,
            "new_cot_preview": "",
            "note": "no FF-HARD flip in any condition — pre-unlearning only",
        }

    # Step 2: Structure quality filter
    structured = []
    for c in flipped:
        cot_text = c["new_cot"][0] if c["new_cot"] else ""
        passes, blocks_passing, detail = structure_quality(cot_text)
        c["_struct_passes"] = passes
        c["_blocks_passing"] = blocks_passing
        c["_struct_detail"] = detail
        if passes:
            structured.append(c)

    pool = structured if structured else flipped  # fall back if none pass structure
    fallback_note = "" if structured else "structure filter: no pair passed — using earliest flip"

    # Step 3: Earliest clean epoch
    min_epoch = min(c["epoch"] for c in pool)
    earliest = [c for c in pool if c["epoch"] == min_epoch]

    # Step 4: Tiebreak — lowest specificity change
    for c in earliest:
        c["_spec_delta"] = specificity_delta(
            c["epoch0_specificity_preds"], c["specificity_preds"]
        )
    best = min(earliest, key=lambda c: c["_spec_delta"])

    cot_text = best["new_cot"][0] if best["new_cot"] else ""
    preview = cot_text[:200].replace("\n", " / ")

    return {
        "condition": best["condition"],
        "epoch": best["epoch"],
        "ff_hard": True,
        "ff_soft": round(ff_soft_initial, 4),
        "structure_checks_passed": best.get("_blocks_passing", 0),
        "specificity_delta": best["_spec_delta"],
        "new_cot_preview": preview,
        "note": fallback_note,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--sweep_files", nargs="*", default=None,
        help="Override sweep files. Provide paths; condition names are inferred "
             "from filenames. If omitted, uses default pilot_sweep_1..4.jsonl paths."
    )
    parser.add_argument(
        "--output", default="data/best_conditions.json",
        help="Output JSON path (default: data/best_conditions.json)"
    )
    args = parser.parse_args()

    # Build condition→file mapping
    if args.sweep_files:
        sweep_map = {}
        for path in args.sweep_files:
            name = Path(path).stem  # e.g. "pilot_sweep_1" or "sweep_verify"
            sweep_map[name] = path
    else:
        sweep_map = DEFAULT_SWEEP_FILES

    print("Loading sweep files:")
    all_records = {}  # condition → {qid: record}
    for cond, path in sweep_map.items():
        all_records[cond] = load_sweep_file(path, cond)

    if not any(all_records.values()):
        print("No records loaded — nothing to do.")
        sys.exit(0)

    # Collect all question IDs across all conditions
    all_qids = sorted({qid for recs in all_records.values() for qid in recs})
    print(f"\n{len(all_qids)} unique question(s) found across all conditions.\n")

    # Build per-question candidate list
    results = {}
    for qid in all_qids:
        candidates = []
        for cond, recs in all_records.items():
            if qid not in recs:
                continue
            rec = recs[qid]
            ur = rec["unlearning_results"]

            # Epoch 0 baseline (pre-unlearning)
            e0 = ur.get("0") or ur.get(0)
            if e0 is None:
                continue
            epoch0_pred = e0["prediction"]
            epoch0_spec = e0.get("specificity_preds", [])
            initial_probs = rec.get("initial_probs", e0["probs"])
            correct_letter = rec.get("correct", "A")
            correct_idx = LETTER_TO_IDX.get(correct_letter, 0)

            # Epoch N candidates (skip epoch 0 — no unlearning yet)
            for epoch_key, epoch_data in ur.items():
                epoch_num = int(epoch_key)
                if epoch_num == 0:
                    continue
                candidates.append({
                    "condition": cond,
                    "epoch": epoch_num,
                    "probs": epoch_data["probs"],
                    "prediction": epoch_data["prediction"],
                    "new_cot": epoch_data.get("new_cot", []),
                    "specificity_preds": epoch_data.get("specificity_preds", []),
                    "correct_idx": correct_idx,
                    "epoch0_prediction": epoch0_pred,
                    "epoch0_specificity_preds": epoch0_spec,
                    "initial_probs": initial_probs,
                })

        if not candidates:
            results[qid] = {
                "condition": None, "epoch": None, "ff_hard": False,
                "ff_soft": None, "structure_checks_passed": None,
                "specificity_delta": None, "new_cot_preview": "",
                "note": "no data found for this question",
            }
        else:
            results[qid] = select_for_question(qid, candidates)

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}\n")

    # Print summary table
    col_w = [20, 10, 6, 8, 9, 10, 12]
    header = (
        f"{'Question':<{col_w[0]}}  "
        f"{'Condition':<{col_w[1]}}  "
        f"{'Epoch':<{col_w[2]}}  "
        f"{'FF-HARD':<{col_w[3]}}  "
        f"{'FF-SOFT':<{col_w[4]}}  "
        f"{'StructBlks':<{col_w[5]}}  "
        f"{'SpecDelta':<{col_w[6]}}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for qid, sel in results.items():
        cond_str = sel["condition"] or "—"
        epoch_str = str(sel["epoch"]) if sel["epoch"] is not None else "—"
        ff_hard_str = "YES" if sel["ff_hard"] else "NO"
        ff_soft_str = f"{sel['ff_soft']:.4f}" if sel["ff_soft"] is not None else "—"
        struct_str = str(sel["structure_checks_passed"]) if sel["structure_checks_passed"] is not None else "—"
        spec_str = str(sel["specificity_delta"]) if sel["specificity_delta"] is not None else "—"
        note = f"  ← {sel['note']}" if sel["note"] else ""
        print(
            f"{qid:<{col_w[0]}}  "
            f"{cond_str:<{col_w[1]}}  "
            f"{epoch_str:<{col_w[2]}}  "
            f"{ff_hard_str:<{col_w[3]}}  "
            f"{ff_soft_str:<{col_w[4]}}  "
            f"{struct_str:<{col_w[5]}}  "
            f"{spec_str:<{col_w[6]}}"
            f"{note}"
        )
    print(sep)
    total = len(results)
    flipped = sum(1 for s in results.values() if s["ff_hard"])
    print(f"\n{flipped}/{total} questions achieved FF-HARD flip.")
    if total - flipped > 0:
        print(f"{total - flipped} question(s) flagged as pre-unlearning only.")


if __name__ == "__main__":
    main()
