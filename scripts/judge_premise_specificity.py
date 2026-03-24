"""
LLM-as-judge premise/mode classifier for the sub-block unlearning sweep.

v1: SPECIFIC / GENERIC  (linguistic specificity)
v2: CAUSAL / NON-CAUSAL (functional/causal criterion)
v3: PREMISE_ONLY / REASONING_ONLY / WHOLE_BLOCK / NONE
    (few-shot mode prediction — evaluated against argmax ff_soft ground truth)

Usage:
    python scripts/judge_premise_specificity.py

Requires ANTHROPIC_API_KEY environment variable.

Output files:
  data/premise_causality_judgments.json   — v2 results (skipped if already exists)
  data/premise_mode_judgments_v3.json     — v3 results
"""

import glob
import json
import os
import re
import sys

import anthropic

# ── Constants ────────────────────────────────────────────────────────────────

MODEL          = "claude-sonnet-4-6"
MAX_TOKENS     = 80
DATA_DIR       = "data"
THRESHOLD      = 0.1   # v1/v2 binary: po_ff > THRESHOLD → premise carries signal
TIE_FF         = 0.05  # v3 tie tolerance: accept if within TIE_FF of best_ff
NONE_THRESHOLD = 0.05  # all modes failed if best_ff < this

# Few-shot examples held out from v3 evaluation (judge has seen their answers)
FEW_SHOT_QIDS = frozenset({
    "openbook_3",        # Example 1 — REASONING_ONLY wins by large margin
    "openbook_8-97",     # Example 2 — PREMISE + REASONING both strong
    "openbook_406",      # Example 3 — WHOLE_BLOCK wins
    "openbook_7-1132",   # Example 4 — NONE (all modes fail)
})

# Maps v3 output labels (uppercase) ↔ ff_vals dict keys (lowercase)
V3_LABEL_TO_KEY = {
    "PREMISE_ONLY":          "premise_only",
    "REASONING_ONLY":        "reasoning_only",
    "WHOLE_BLOCK":           "whole_block",
    "PREMISE_AND_REASONING": "premise_and_reasoning",
    "NONE":                  None,
}

# ── v2 prompt — CAUSAL / NON-CAUSAL ─────────────────────────────────────────

JUDGE_PROMPT_V2 = """\
You are predicting whether unlearning a specific Premise from a language model's parameters would cause it to change its answer to a multiple choice question.

Question: {question_text}
Correct answer: {correct_answer_letter} — {correct_answer_text}
Answer option being evaluated: {answer_option_A}
Full reasoning block:
{full_block_text}

Focus specifically on this Premise:
{premise_text}

Classify the Premise as:

CAUSAL — the premise contains information the model needs to evaluate this specific answer option. Without it the model loses a key bridge to its judgment. Indicators: the premise makes a claim directly about the answer option's mechanism, names a specific entity or property that distinguishes this option, or provides a fact the model cannot easily infer from the question alone.

NON-CAUSAL — the premise restates information the model already encodes elsewhere, or provides context that does not change how the model would evaluate this option. Indicators: the premise is a broadly true background fact, restates part of the question, or could be removed without changing the reasoning chain's conclusion.

Respond with CAUSAL or NON-CAUSAL on the first line. On the second line write one sentence explaining the causal connection or lack thereof."""

# ── v3 prompt — few-shot mode prediction ────────────────────────────────────

JUDGE_PROMPT_V3 = """\
You are predicting which component of a reasoning block, when erased from a model's memory, will most disrupt the model's ability to answer a multiple choice question correctly.

The options are:
  PREMISE_ONLY — erase only the premise statement
  REASONING_ONLY — erase only the reasoning sentence
  WHOLE_BLOCK — erase the full answer block together
  NONE — none of the components will disrupt the model's prediction when erased

Here are examples of how to reason about this:

--- EXAMPLE 1 ---
Question: Sources of air pollution are
Correct answer: B — Landfills
Answer block being evaluated (option A: Walking):
  * Premise: Human activities can contribute to air pollution.
  * Reasoning: Walking is a human activity that does not typically involve the release of pollutants into the air.
  * Conclusion: R

What happened when each component was erased:
  PREMISE_ONLY:   ff_soft=0.035 (did not disrupt)
  REASONING_ONLY: ff_soft=0.858 (strongly disrupted)
  WHOLE_BLOCK:    ff_soft=0.000 (did not disrupt)

Best mode: REASONING_ONLY
Why: The premise is a generic background fact the model knows independently. The reasoning sentence contains the specific causal step — that walking does not release pollutants — that the model needs to evaluate option A. Erasing it forces the model to reconsider.

--- EXAMPLE 2 ---
Question: Someone wants their electromagnets to work, but is having difficulty powering them. In order to make them work, they need to
Correct answer: B — run a continuous current
Answer block being evaluated (option A: run wire through currants):
  * Premise: Electromagnets require a flow of electric current to function.
  * Reasoning: Running wire through currants is not a feasible or practical way to power electromagnets, as currants are a type of fruit and do not have the ability to conduct electricity.
  * Conclusion: R

What happened when each component was erased:
  PREMISE_ONLY:   ff_soft=0.976 (strongly disrupted)
  REASONING_ONLY: ff_soft=0.992 (strongly disrupted)
  WHOLE_BLOCK:    ff_soft=0.025 (did not disrupt — gradient diluted)

Best mode: REASONING_ONLY (marginally over PREMISE_ONLY)
Why: Both components carry specific factual content about electromagnets and electricity. The reasoning contains the key discrimination (currants are fruit, not a conductor). When both are erased together the gradient is diluted and neither is effectively removed.

--- EXAMPLE 3 ---
Question: Dairy is a source of
Correct answer: C — a group of fat-soluble secosteroids
Answer block being evaluated (option A: a vitamin that prevents blood loss):
  * Premise: Dairy products contain various nutrients.
  * Reasoning: Dairy is a source of calcium, which is essential for blood clotting and preventing blood loss.
  * Conclusion: S

What happened when each component was erased:
  PREMISE_ONLY:   ff_soft=0.150 (minor disruption)
  REASONING_ONLY: ff_soft=0.000 (did not disrupt)
  WHOLE_BLOCK:    ff_soft=0.744 (strongly disrupted)

Best mode: WHOLE_BLOCK
Why: The premise alone is generic; the reasoning alone is insufficient. But together the block encodes a specific false causal chain (dairy → calcium → prevents blood loss → supports option A) that the model uses to incorrectly evaluate this option. Only removing the entire block breaks this wrong reasoning.

--- EXAMPLE 4 ---
Question: Atomic 26 is drawn to a device, it could be
Correct answer: A — magnetized
Answer block being evaluated (option A: magnetized):
  * Premise: Atomic 26 is drawn to a device.
  * Reasoning: The device is a magnet, and atomic 26 is Iron (Fe), which is attracted to magnets.
  * Conclusion: S

What happened when each component was erased:
  PREMISE_ONLY:   ff_soft=0.000 (did not disrupt)
  REASONING_ONLY: ff_soft=0.000 (did not disrupt)
  WHOLE_BLOCK:    ff_soft=0.000 (did not disrupt)

Best mode: NONE
Why: The model's parametric knowledge about iron and magnets is robust — erasing any component of the reasoning block cannot override what the model knows about atomic number 26 and ferromagnetism.

---

Now predict for this new case:

Question: {question_text}
Correct answer: {correct_answer_letter} — {correct_answer_text}
Answer block being evaluated (option A: {answer_option_A}):
{full_block_text}

Which single mode will most disrupt the model's prediction when erased?

Respond on the first line with exactly one of:
  PREMISE_ONLY
  REASONING_ONLY
  WHOLE_BLOCK
  NONE

On the second line write one sentence explaining which component contains the information most critical to the model's parametric prediction."""

# ── Data loading ─────────────────────────────────────────────────────────────

def load_whole_block_step0():
    records = {}
    for fpath in sorted(glob.glob(os.path.join(DATA_DIR, "subblock_whole_block_q*.jsonl"))):
        with open(fpath) as f:
            for line in f:
                rec = json.loads(line)
                if rec["step_idx"] == 0:
                    records[rec["id"]] = rec
    return records


def load_premise_only_targets():
    targets = {}
    for fpath in sorted(glob.glob(os.path.join(DATA_DIR, "subblock_premise_only_q*.jsonl"))):
        with open(fpath) as f:
            for line in f:
                rec = json.loads(line)
                targets[(rec["id"], rec["step_idx"])] = rec.get("unlearn_target_text", "")
    return targets


def load_v1_judgments():
    p = os.path.join(DATA_DIR, "premise_specificity_judgments.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def load_v2_judgments():
    p = os.path.join(DATA_DIR, "premise_causality_judgments.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def compute_ff_soft_per_mode():
    modes = ["whole_block", "premise_only", "reasoning_only", "premise_and_reasoning"]
    result = {m: {} for m in modes}
    for mode in modes:
        for fpath in sorted(glob.glob(os.path.join(DATA_DIR, f"subblock_{mode}_q*.jsonl"))):
            with open(fpath) as f:
                for line in f:
                    rec = json.loads(line)
                    if rec["step_idx"] != 0:
                        continue
                    qid = rec["id"]
                    ur = rec.get("unlearning_results", {})
                    cidx = ord(rec["correct"]) - ord("A")
                    ip = rec["initial_probs"][cidx]
                    best = 0.0
                    for _, er in (ur.items() if isinstance(ur, dict) else enumerate(ur)):
                        if not isinstance(er, dict):
                            continue
                        probs = er.get("probs") or er.get("answer_probs") or []
                        if probs and cidx < len(probs):
                            drop = ip - list(probs)[cidx]
                            if drop > best:
                                best = drop
                    result[mode][qid] = best
    return result

# ── Ground truth ─────────────────────────────────────────────────────────────

EVAL_MODES = ["premise_only", "reasoning_only", "whole_block", "premise_and_reasoning"]


def build_ground_truth(ff_by_mode):
    """
    Returns {qid: {best_mode_key, best_ff, second_ff, margin, ff_vals, is_none}}.
    best_mode_key is a lowercase ff key (e.g. 'reasoning_only') or 'NONE'.
    """
    gt = {}
    for qid in sorted(ff_by_mode["whole_block"].keys()):
        vals = {m: ff_by_mode[m].get(qid, 0.0) for m in EVAL_MODES}
        max_ff = max(vals.values())
        if max_ff < NONE_THRESHOLD:
            gt[qid] = {"best_mode": "NONE", "best_ff": max_ff,
                       "second_ff": 0.0, "margin": 0.0,
                       "ff_vals": vals, "is_none": True}
        else:
            ranked = sorted(vals.items(), key=lambda x: x[1], reverse=True)
            best_m, best_f = ranked[0]
            second_f = ranked[1][1] if len(ranked) > 1 else 0.0
            gt[qid] = {"best_mode": best_m, "best_ff": best_f,
                       "second_ff": second_f, "margin": best_f - second_f,
                       "ff_vals": vals, "is_none": False}
    return gt


def v3_correct(predicted_label, entry):
    """
    predicted_label: v3 uppercase label ('PREMISE_ONLY', 'REASONING_ONLY', etc.)
                     OR a lowercase ff key ('premise_only', etc.) from v1/v2 mapping.
    Returns True if exact match or within TIE_FF of best_ff.
    """
    best_mode = entry["best_mode"]  # lowercase key or 'NONE'

    # Normalize predicted to lowercase ff key
    if predicted_label in V3_LABEL_TO_KEY:
        pred_key = V3_LABEL_TO_KEY[predicted_label]   # may be None for "NONE"
    else:
        pred_key = predicted_label  # already lowercase (from v1/v2 mapping)

    # NONE handling
    if pred_key is None or pred_key == "NONE":
        return entry["is_none"]
    if entry["is_none"]:
        return False

    # Exact match
    if pred_key == best_mode:
        return True

    # Tie: predicted mode's ff within TIE_FF of best
    pred_ff = entry["ff_vals"].get(pred_key, 0.0)
    return (entry["best_ff"] - pred_ff) <= TIE_FF


def v1_v2_implied_mode(judgment):
    """Map binary v1/v2 labels to implied mode prediction (lowercase ff key)."""
    if judgment in ("SPECIFIC", "CAUSAL"):
        return "premise_only"
    elif judgment in ("GENERIC", "NON-CAUSAL"):
        return "reasoning_only"
    return None

# ── Premise / block extraction ───────────────────────────────────────────────

def extract_premise_from_block(block_text):
    m = re.search(r"\* Premise: (.+)", block_text)
    return ("* Premise: " + m.group(1).strip()) if m else None


def extract_block_for_step(cot_text, step_idx):
    letter = chr(ord("A") + step_idx)
    m = re.search(rf"Answer {letter}:.*?(?=Answer [A-Z]:|$)", cot_text, re.DOTALL)
    return m.group(0).strip() if m else cot_text.strip()


def extract_premise_for_step(cot_text, step_idx):
    return extract_premise_from_block(extract_block_for_step(cot_text, step_idx))


def format_block_body(block_text):
    """Strip 'Answer X: For each answer:' header; return indented body lines."""
    lines = [l for l in block_text.strip().split("\n")
             if l.strip() and not re.match(r"Answer [A-Z]:", l.strip())]
    return "\n".join("  " + l.strip() for l in lines)

# ── API calls ────────────────────────────────────────────────────────────────

def call_judge_v2(client, question_text, correct_letter, correct_text,
                  answer_option_A, full_block_text, premise_text):
    prompt = JUDGE_PROMPT_V2.format(
        question_text=question_text,
        correct_answer_letter=correct_letter,
        correct_answer_text=correct_text,
        answer_option_A=answer_option_A,
        full_block_text=full_block_text,
        premise_text=premise_text,
    )
    resp = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS,
                                  messages=[{"role": "user", "content": prompt}])
    raw   = resp.content[0].text.strip()
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    first = lines[0].upper() if lines else ""
    expl  = lines[1] if len(lines) > 1 else ""
    judgment = "NON-CAUSAL" if "NON-CAUSAL" in first else \
               "CAUSAL"     if "CAUSAL"     in first else first
    return judgment, expl, raw, resp


def call_judge_v3(client, question_text, correct_letter, correct_text,
                  answer_option_A, formatted_block):
    prompt = JUDGE_PROMPT_V3.format(
        question_text=question_text,
        correct_answer_letter=correct_letter,
        correct_answer_text=correct_text,
        answer_option_A=answer_option_A,
        full_block_text=formatted_block,
    )
    resp  = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS,
                                   messages=[{"role": "user", "content": prompt}])
    raw   = resp.content[0].text.strip()
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    first = lines[0].upper() if lines else ""
    expl  = lines[1] if len(lines) > 1 else ""
    # Check in order of specificity to avoid partial matches
    for label in ("PREMISE_ONLY", "REASONING_ONLY", "WHOLE_BLOCK", "NONE"):
        if label in first:
            return label, expl, raw, resp
    return first, expl, raw, resp

# ── v2 scoring ───────────────────────────────────────────────────────────────

def run_v2(client, wb_records, v1_j, ff_by_mode):
    qids = sorted(wb_records.keys())
    judgments = {}
    total_in = total_out = 0

    for qid in qids:
        rec = wb_records[qid]
        qt  = rec["question"]
        cl  = rec["correct"]
        ct  = rec["options"][ord(cl) - ord("A")]
        oA  = rec["options"][0]
        fbt = rec["unlearn_target_text"]
        pt  = extract_premise_for_step(rec["initial_cot"][0], 0)

        jud, expl, raw, resp = call_judge_v2(client, qt, cl, ct, oA, fbt, pt)
        total_in  += resp.usage.input_tokens
        total_out += resp.usage.output_tokens

        po_ff = ff_by_mode["premise_only"].get(qid, 0.0)
        v1_jud = v1_j.get(qid, {}).get("judgment")
        v1_ok  = (po_ff > THRESHOLD) if v1_jud == "SPECIFIC" else \
                 (po_ff <= THRESHOLD) if v1_jud == "GENERIC"  else None
        v2_ok  = (po_ff > THRESHOLD) if jud == "CAUSAL" else (po_ff <= THRESHOLD)

        judgments[qid] = {
            "question_text":  qt,
            "answer_option_A": oA,
            "premise_text":   pt,
            "full_block_text": fbt,
            "judgment_v2":    jud,
            "explanation_v2": expl,
            "judgment_v1":    v1_jud,
            "explanation_v1": v1_j.get(qid, {}).get("explanation"),
            "v1_correct":     v1_ok,
            "v2_correct":     v2_ok,
            "model":          MODEL,
        }
        print(f"  {qid:20s}  v1={str(v1_jud or 'N/A'):8s}  v2={jud:10s}  "
              f"po_ff={po_ff:.3f}  {'Y' if v2_ok else 'N'}")

    out = os.path.join(DATA_DIR, "premise_causality_judgments.json")
    with open(out, "w") as f:
        json.dump(judgments, f, indent=2)
    print(f"\nSaved → {out}")
    print(f"Cost: ~{total_in} input, ~{total_out} output tokens")
    return judgments

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    client     = anthropic.Anthropic()
    wb_records = load_whole_block_step0()
    v1_j       = load_v1_judgments()
    ff_by_mode = compute_ff_soft_per_mode()
    qids       = sorted(wb_records.keys())

    # ── 1. Ground truth table ────────────────────────────────────────────────
    gt         = build_ground_truth(ff_by_mode)
    eval_qids  = [q for q in qids if q not in FEW_SHOT_QIDS]

    print("=" * 84)
    print("GROUND TRUTH — Best unlearning mode per question (argmax ff_soft)")
    print("=" * 84)
    print(f"  {'question_id':20s} | {'best_mode':25s} | best_ff | margin | "
          f"  po  |  ro  |  wb  | par")
    print("  " + "-" * 80)
    for qid in qids:
        e = gt[qid]
        v = e["ff_vals"]
        tag = " ← few-shot" if qid in FEW_SHOT_QIDS else ""
        print(f"  {qid:20s} | {e['best_mode']:25s} | {e['best_ff']:7.3f} | "
              f"{e['margin']:6.3f} | "
              f"{v['premise_only']:5.3f} | {v['reasoning_only']:5.3f} | "
              f"{v['whole_block']:5.3f} | {v['premise_and_reasoning']:5.3f}{tag}")

    print(f"\nFew-shot hold-outs (4, not evaluated): {', '.join(sorted(FEW_SHOT_QIDS))}")
    print(f"Evaluation set ({len(eval_qids)}):              {', '.join(eval_qids)}")

    # ── 2. v2: skip if already exists ───────────────────────────────────────
    v2_path = os.path.join(DATA_DIR, "premise_causality_judgments.json")
    if os.path.exists(v2_path):
        print(f"\nv2 judgments already exist at {v2_path} — loading (not re-scoring).")
        with open(v2_path) as f:
            v2_j = json.load(f)
    else:
        print("\n" + "=" * 72)
        print("V2 — CAUSAL/NON-CAUSAL (running fresh — file not found)")
        print("=" * 72)

        # brief v2 verification
        for qid in ["openbook_1161", "openbook_8-97"]:
            rec = wb_records[qid]
            jud, _, raw, _ = call_judge_v2(
                client, rec["question"], rec["correct"],
                rec["options"][ord(rec["correct"]) - ord("A")],
                rec["options"][0], rec["unlearn_target_text"],
                extract_premise_for_step(rec["initial_cot"][0], 0),
            )
            print(f"\n  {qid}: {raw!r}  →  {jud}")

        ans = input("\nProceed with full v2 scoring? [y/N] ").strip().lower()
        if ans != "y":
            sys.exit(0)

        v2_j = run_v2(client, wb_records, v1_j, ff_by_mode)

    # ── 3. v3: verification ──────────────────────────────────────────────────
    print()
    print("=" * 84)
    print("V3 — FEW-SHOT MODE PREDICTION")
    print("=" * 84)
    print(f"\nHeld out (few-shot examples, not in eval): {', '.join(sorted(FEW_SHOT_QIDS))}")
    print(f"Eval set ({len(eval_qids)} questions):  {', '.join(eval_qids)}")
    print()
    print("─" * 72)
    print(f"Verification — {eval_qids[0]} and {eval_qids[1]}")
    print("─" * 72)

    verify_in = verify_out = 0
    for qid in eval_qids[:2]:
        rec  = wb_records[qid]
        qt   = rec["question"]
        cl   = rec["correct"]
        ct   = rec["options"][ord(cl) - ord("A")]
        oA   = rec["options"][0]
        fbt  = format_block_body(rec["unlearn_target_text"])
        gt_e = gt[qid]

        print(f"\n  {qid}")
        print(f"    question     : {qt}")
        print(f"    correct      : {cl} — {ct}")
        print(f"    option A     : {oA}")
        print(f"    ground truth : {gt_e['best_mode']}  "
              f"(ff={gt_e['best_ff']:.3f}, margin={gt_e['margin']:.3f})")
        print(f"    formatted block:")
        for ln in fbt.split("\n"):
            print(f"      {ln}")

        jud, expl, raw, resp = call_judge_v3(client, qt, cl, ct, oA, fbt)
        verify_in  += resp.usage.input_tokens
        verify_out += resp.usage.output_tokens
        ok = v3_correct(jud, gt_e)

        print(f"\n    Raw API response:")
        for ln in raw.split("\n"):
            print(f"      {ln}")
        print(f"    Parsed: {jud}  |  Correct: {'Y' if ok else 'N'}")

    est_in  = int(verify_in  / 2 * len(eval_qids))
    est_out = int(verify_out / 2 * len(eval_qids))
    print(f"\n  Estimated cost for all {len(eval_qids)} eval questions:")
    print(f"    ~{est_in} input, ~{est_out} output tokens")

    ans = input(f"\nProceed with full v3 scoring ({len(eval_qids)} questions)? [y/N] "
                ).strip().lower()
    if ans != "y":
        sys.exit(0)

    # ── 4. v3: full scoring ──────────────────────────────────────────────────
    print()
    print("─" * 72)
    v3_results = {}
    total_in_3  = verify_in
    total_out_3 = verify_out

    for qid in eval_qids:
        rec  = wb_records[qid]
        qt   = rec["question"]
        cl   = rec["correct"]
        ct   = rec["options"][ord(cl) - ord("A")]
        oA   = rec["options"][0]
        fbt  = format_block_body(rec["unlearn_target_text"])
        gt_e = gt[qid]

        jud, expl, raw, resp = call_judge_v3(client, qt, cl, ct, oA, fbt)
        total_in_3  += resp.usage.input_tokens
        total_out_3 += resp.usage.output_tokens
        ok = v3_correct(jud, gt_e)

        v3_results[qid] = {
            "question_text":    qt,
            "answer_option_A":  oA,
            "predicted_mode":   jud,
            "explanation":      expl,
            "actual_best_mode": gt_e["best_mode"],
            "best_ff":          gt_e["best_ff"],
            "margin":           gt_e["margin"],
            "correct":          ok,
            "ff_vals":          gt_e["ff_vals"],
            "model":            MODEL,
        }
        print(f"  {qid:20s}  pred={jud:15s}  actual={gt_e['best_mode']:25s}  "
              f"{'Y' if ok else 'N'}")

    out3 = os.path.join(DATA_DIR, "premise_mode_judgments_v3.json")
    with open(out3, "w") as f:
        json.dump(v3_results, f, indent=2)
    print(f"\nSaved → {out3}")
    print(f"Cost: ~{total_in_3} input, ~{total_out_3} output tokens")

    # ── 5. Results table ─────────────────────────────────────────────────────
    print()
    print("=" * 84)
    print("RESULTS TABLE — v3 evaluation (11 questions)")
    print("=" * 84)
    print(f"  {'question_id':20s} | {'predicted':15s} | {'actual_best':25s} | ok | margin")
    print("  " + "-" * 72)

    n_v3 = 0
    fail_cases = []
    for qid in eval_qids:
        r  = v3_results[qid]
        ok = r["correct"]
        if ok:
            n_v3 += 1
        else:
            fail_cases.append({"question_id": qid, **r})
        print(f"  {qid:20s} | {r['predicted_mode']:15s} | {r['actual_best_mode']:25s} | "
              f"{'Y' if ok else 'N':2s} | {r['margin']:.3f}")

    # ── 6. Comparison v1/v2/v3 on same 11 eval questions ────────────────────
    n_v1 = n_v2 = 0
    for qid in eval_qids:
        gt_e   = gt[qid]
        v1_imp = v1_v2_implied_mode(v1_j.get(qid, {}).get("judgment"))
        v2_imp = v1_v2_implied_mode(v2_j.get(qid, {}).get("judgment_v2"))
        if v1_imp and v3_correct(v1_imp, gt_e):
            n_v1 += 1
        if v2_imp and v3_correct(v2_imp, gt_e):
            n_v2 += 1

    print()
    print(f"v3 accuracy (direct mode prediction):      {n_v3}/{len(eval_qids)}")
    print(f"v2 accuracy (CAUSAL→premise, NON→reason):  {n_v2}/{len(eval_qids)}")
    print(f"v1 accuracy (SPECIFIC→premise, GEN→reason):{n_v1}/{len(eval_qids)}")
    print()
    print("Tie rule: accept if predicted mode's ff_soft within 0.05 of best_ff")
    print("v1/v2 mapped to implied mode: SPECIFIC/CAUSAL→premise_only, "
          "GENERIC/NON-CAUSAL→reasoning_only")

    # ── 7. Failure analysis ──────────────────────────────────────────────────
    if fail_cases:
        print()
        print(f"v3 failures (n={len(fail_cases)}):")
        print(f"  {'question_id':20s} | {'predicted':15s} | {'actual':25s} | "
              f"pred_ff | best_ff | gap")
        print("  " + "-" * 72)
        for fc in fail_cases:
            pkey    = V3_LABEL_TO_KEY.get(fc["predicted_mode"], fc["predicted_mode"])
            pred_ff = fc["ff_vals"].get(pkey, 0.0) if pkey else 0.0
            gap     = fc["best_ff"] - pred_ff
            print(f"  {fc['question_id']:20s} | {fc['predicted_mode']:15s} | "
                  f"{fc['actual_best_mode']:25s} | {pred_ff:7.3f} | "
                  f"{fc['best_ff']:7.3f} | {gap:.3f}")

    thr = 7
    status = "≥" if n_v3 >= thr else "<"
    print(f"\nv3 {n_v3}/11 {status} {thr}/11 threshold.")


if __name__ == "__main__":
    main()
