"""
select_best_conditions.py

Reads sweep output JSONL files and selects the best (condition, epoch) pair
per (question_id, step_idx) combination.

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

ff_soft definition (per-step):
  The maximum probability mass removed from the epoch-0 predicted answer
  across all tested epochs:
    ff_soft = epoch0_probs[epoch0_pred] - min(epochN_probs[epoch0_pred])
  This is always >= 0 and varies per (question, step), so argmax across
  steps identifies the step whose unlearning most disrupts the model.

Output: data/best_conditions_30.json (default) + stdout summary table.

Keys in output JSON: "{question_id}_step{step_idx}" → selection dict

Usage:
  # 30-question 5-step dataset (default):
  python select_best_conditions.py \\
    --sweep_files data/sweep30_cond1.jsonl data/sweep30_cond2.jsonl \\
                  data/sweep30_cond3.jsonl data/sweep30_cond4.jsonl \\
    --output data/best_conditions_30.json

  # Existing 7-question step-0-only pilot (backward compat):
  python select_best_conditions.py \\
    --sweep_files data/pilot_sweep_1.jsonl data/pilot_sweep_2.jsonl \\
                  data/pilot_sweep_3.jsonl data/pilot_sweep_4.jsonl \\
    --output data/best_conditions_verify.json
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Default sweep files
# ---------------------------------------------------------------------------

DEFAULT_SWEEP_FILES = {
    "sweep30_cond1": "data/sweep30_cond1.jsonl",
    "sweep30_cond2": "data/sweep30_cond2.jsonl",
    "sweep30_cond3": "data/sweep30_cond3.jsonl",
    "sweep30_cond4": "data/sweep30_cond4.jsonl",
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
    parts = re.split(r"(?=\bAnswer\s+[A-D]\s*:)", cot)
    for part in parts:
        m = re.match(r"Answer\s+([A-D])\s*:", part)
        if not m:
            continue
        letter = m.group(1)
        reasoning = _extract_slot(
            part, "* Reasoning:",
            ["* Conclusion:", "Answer A:", "Answer B:", "Answer C:", "Answer D:", "**Final Answer**"]
        )
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
    tokens = reasoning.split()
    if not tokens:
        return False
    return len(set(tokens)) / len(tokens) >= threshold


def check_min_length(reasoning: str, min_tokens: int = 8) -> bool:
    return len(reasoning.split()) >= min_tokens


def check_conclusion(conclusion: str) -> bool:
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
        a  = check_ttr(b["reasoning"])
        bl = check_min_length(b["reasoning"])
        c  = check_conclusion(b["conclusion"])
        n_pass = sum([a, bl, c])
        if n_pass >= 2:
            passing_blocks += 1
        check_details.append(f"{b['letter']}:{'✓' if a else '✗'}{'✓' if bl else '✗'}{'✓' if c else '✗'}")

    overall = passing_blocks >= 2
    return overall, passing_blocks, " ".join(check_details)


# ---------------------------------------------------------------------------
# Specificity change
# ---------------------------------------------------------------------------

def specificity_delta(epoch0_preds: list, epochN_preds: list) -> int:
    return sum(1 for a, b in zip(epoch0_preds, epochN_preds) if a != b)


# ---------------------------------------------------------------------------
# Load sweep data
# ---------------------------------------------------------------------------

def load_sweep_file(path: str, condition_name: str) -> dict:
    """
    Load a sweep JSONL and index by (question_id, step_idx).
    Returns {(qid, step_idx): record}.
    """
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
            qid      = rec["id"]
            step_idx = rec.get("step_idx", 0)
            key = (qid, step_idx)
            if key in records:
                print(f"  [warn] duplicate key {key} in {path} — keeping first")
            else:
                records[key] = rec
    print(f"  [loaded] {path}: {len(records)} record(s)")
    return records


# ---------------------------------------------------------------------------
# Per-(question, step) selection
# ---------------------------------------------------------------------------

def select_for_pair(qid: str, step_idx: int, candidates: list) -> dict:
    """
    candidates: list of dicts with keys:
      condition, epoch (int), probs, prediction, new_cot, specificity_preds,
      correct_idx, epoch0_prediction, epoch0_specificity_preds, initial_probs

    ff_soft = max drop in epoch-0-prediction probability across all epochs tested.
    This varies per (question, step), enabling argmax across steps.
    """
    epoch0_pred      = candidates[0]["epoch0_prediction"]
    initial_probs    = candidates[0]["initial_probs"]
    correct_idx      = candidates[0]["correct_idx"]
    epoch0_pred_prob = initial_probs[epoch0_pred]

    # ff_soft: max disruption to epoch-0 prediction probability
    all_pred_probs = [c["probs"][epoch0_pred] for c in candidates if c["epoch"] >= 1]
    min_pred_prob  = min(all_pred_probs) if all_pred_probs else epoch0_pred_prob
    ff_soft        = round(max(0.0, epoch0_pred_prob - min_pred_prob), 4)

    step_text = candidates[0].get("step_text", "")

    # Step 1: FF-HARD filter
    flipped = [c for c in candidates if c["prediction"] != epoch0_pred]

    if not flipped:
        return {
            "condition":              None,
            "epoch":                  None,
            "ff_hard":                False,
            "ff_soft":                ff_soft,
            "step_idx":               step_idx,
            "step_text":              step_text,
            "structure_checks_passed": None,
            "specificity_delta":      None,
            "new_cot_preview":        "",
            "note":                   "no FF-HARD flip in any condition",
        }

    # Step 2: Structure quality filter
    structured = []
    for c in flipped:
        cot_text = c["new_cot"][0] if c["new_cot"] else ""
        passes, blocks_passing, detail = structure_quality(cot_text)
        c["_struct_passes"]  = passes
        c["_blocks_passing"] = blocks_passing
        c["_struct_detail"]  = detail
        if passes:
            structured.append(c)

    pool         = structured if structured else flipped
    fallback_note = "" if structured else "structure filter: no pair passed — using earliest flip"

    # Step 3: Earliest clean epoch
    min_epoch = min(c["epoch"] for c in pool)
    earliest  = [c for c in pool if c["epoch"] == min_epoch]

    # Step 4: Tiebreak — lowest specificity change
    for c in earliest:
        c["_spec_delta"] = specificity_delta(
            c["epoch0_specificity_preds"], c["specificity_preds"]
        )
    best = min(earliest, key=lambda c: c["_spec_delta"])

    cot_text = best["new_cot"][0] if best["new_cot"] else ""
    preview  = cot_text[:200].replace("\n", " / ")

    return {
        "condition":               best["condition"],
        "epoch":                   best["epoch"],
        "ff_hard":                 True,
        "ff_soft":                 ff_soft,
        "step_idx":                step_idx,
        "step_text":               step_text,
        "structure_checks_passed": best.get("_blocks_passing", 0),
        "specificity_delta":       best["_spec_delta"],
        "new_cot_preview":         preview,
        "note":                    fallback_note,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--sweep_files", nargs="*", default=None,
        help="Override sweep files. Condition names are inferred from file stems. "
             "If omitted, uses default sweep30_cond1..4 paths."
    )
    parser.add_argument(
        "--output", default="data/best_conditions_30.json",
        help="Output JSON path (default: data/best_conditions_30.json)"
    )
    args = parser.parse_args()

    # Build condition→file mapping
    if args.sweep_files:
        sweep_map = {Path(p).stem: p for p in args.sweep_files}
    else:
        sweep_map = DEFAULT_SWEEP_FILES

    print("Loading sweep files:")
    all_records = {}  # condition → {(qid, step_idx): record}
    for cond, path in sweep_map.items():
        all_records[cond] = load_sweep_file(path, cond)

    if not any(all_records.values()):
        print("No records loaded — nothing to do.")
        sys.exit(0)

    # Collect all (qid, step_idx) keys
    all_keys = sorted({key for recs in all_records.values() for key in recs})
    print(f"\n{len(all_keys)} unique (question, step) pair(s) found across all conditions.\n")

    # Build per-(question, step) candidate list
    results = {}
    for (qid, step_idx) in all_keys:
        candidates = []
        for cond, recs in all_records.items():
            if (qid, step_idx) not in recs:
                continue
            rec = recs[(qid, step_idx)]
            ur  = rec["unlearning_results"]

            e0 = ur.get("0") or ur.get(0)
            if e0 is None:
                continue
            epoch0_pred       = e0["prediction"]
            epoch0_spec       = e0.get("specificity_preds", [])
            initial_probs     = rec.get("initial_probs", e0["probs"])
            correct_letter    = rec.get("correct", "A")
            correct_idx       = LETTER_TO_IDX.get(correct_letter, 0)
            step_text         = rec.get("cot_step", "")

            for epoch_key, epoch_data in ur.items():
                epoch_num = int(epoch_key)
                if epoch_num == 0:
                    continue
                candidates.append({
                    "condition":              cond,
                    "epoch":                  epoch_num,
                    "probs":                  epoch_data["probs"],
                    "prediction":             epoch_data["prediction"],
                    "new_cot":                epoch_data.get("new_cot", []),
                    "specificity_preds":      epoch_data.get("specificity_preds", []),
                    "correct_idx":            correct_idx,
                    "epoch0_prediction":      epoch0_pred,
                    "epoch0_specificity_preds": epoch0_spec,
                    "initial_probs":          initial_probs,
                    "step_text":              step_text,
                })

        out_key = f"{qid}_step{step_idx}"
        if not candidates:
            results[out_key] = {
                "condition": None, "epoch": None, "ff_hard": False,
                "ff_soft": None, "step_idx": step_idx, "step_text": "",
                "structure_checks_passed": None, "specificity_delta": None,
                "new_cot_preview": "", "note": "no data found",
            }
        else:
            results[out_key] = select_for_pair(qid, step_idx, candidates)

    # Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}\n")

    # ---------------------------------------------------------------------------
    # Per-(question, step) summary table
    # ---------------------------------------------------------------------------
    col_w = [20, 10, 6, 8, 9, 10, 12]
    header = (
        f"{'Question+Step':<28}  "
        f"{'Condition':<14}  "
        f"{'Epoch':<6}  "
        f"{'FF-HARD':<8}  "
        f"{'FF-SOFT':<9}  "
        f"{'StructBlks':<10}  "
        f"{'SpecDelta':<10}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for out_key, sel in results.items():
        cond_str   = sel["condition"] or "—"
        epoch_str  = str(sel["epoch"]) if sel["epoch"] is not None else "—"
        ff_hard_str = "YES" if sel["ff_hard"] else "NO"
        ff_soft_str = f"{sel['ff_soft']:.4f}" if sel["ff_soft"] is not None else "—"
        struct_str  = str(sel["structure_checks_passed"]) if sel["structure_checks_passed"] is not None else "—"
        spec_str    = str(sel["specificity_delta"]) if sel["specificity_delta"] is not None else "—"
        note        = f"  ← {sel['note']}" if sel.get("note") else ""
        print(
            f"{out_key:<28}  "
            f"{cond_str:<14}  "
            f"{epoch_str:<6}  "
            f"{ff_hard_str:<8}  "
            f"{ff_soft_str:<9}  "
            f"{struct_str:<10}  "
            f"{spec_str:<10}"
            f"{note}"
        )
    print(sep)

    total   = len(results)
    flipped = sum(1 for s in results.values() if s["ff_hard"])
    print(f"\n{flipped}/{total} (question, step) pairs achieved FF-HARD flip.")

    # ---------------------------------------------------------------------------
    # Per-question summary: steps flipped, most salient step
    # ---------------------------------------------------------------------------
    by_question = defaultdict(list)
    for out_key, sel in results.items():
        # Extract qid: everything before _stepN
        qid = "_".join(out_key.split("_")[:-1])  # handles multi-underscore IDs
        # More robustly: use the step_idx field
        by_question[qid].append(sel)

    print("\n── Per-question summary ──────────────────────────────────────────────────────")
    print(f"{'Question':<25}  {'Steps FF-HARD':<14}  {'Most salient step':<20}  "
          f"{'ff_soft':<8}  {'Condition'}")
    print("-" * 90)
    for qid in sorted(by_question):
        steps = by_question[qid]
        n_flipped = sum(1 for s in steps if s["ff_hard"])
        # Most salient = highest ff_soft among all steps
        valid = [s for s in steps if s["ff_soft"] is not None]
        if valid:
            best_step = max(valid, key=lambda s: s["ff_soft"])
            step_label = f"step_{best_step['step_idx']}"
            salient_ff = best_step["ff_soft"]
            salient_cond = best_step["condition"] or "—"
        else:
            step_label, salient_ff, salient_cond = "—", "—", "—"
        print(f"{qid:<25}  {n_flipped}/{len(steps)} steps flipped  "
              f"{step_label:<20}  {salient_ff!s:<8}  {salient_cond}")


if __name__ == "__main__":
    main()
