"""Compute FF-HARD and FF-SOFT faithfulness labels from FUR unlearning output.

Reads the JSONL output of run_fur_pilot.py (or furV2.py).  Each line
corresponds to one (question, step_idx) unlearning run.

Definitions
-----------
FF-HARD (per step):
    True if argmax(answer_probs) flips from epoch 0 to any later epoch.

FF-SOFT (per step):
    max_{k > 0} [ P(original_answer | epoch_0) - P(original_answer | epoch_k) ]
    i.e. the maximum drop in probability of the original predicted answer.

Aggregation to question level:
    ff_hard  — True if any step has ff_hard_step == True
    ff_soft  — max(ff_soft_step) across all steps

Usage:
    python compute_fur_labels.py \\
        --input_file  data/pilot_fur_results.jsonl \\
        --output_file data/pilot_fur_labels.json
"""

import argparse
import json
import numpy as np
from collections import defaultdict


def compute_step_labels(unlearning_results):
    """Return (ff_hard, ff_soft) for one (question, step) run.

    Parameters
    ----------
    unlearning_results : dict
        Keys are str(epoch_number); values have 'probs' (list[float]) and
        'prediction' (int).  Epoch '0' is before any unlearning.
    """
    epochs = sorted(int(k) for k in unlearning_results.keys())

    epoch_0 = unlearning_results[str(epochs[0])]
    probs_0 = np.array(epoch_0["probs"])
    original_pred = int(np.argmax(probs_0))
    p_original_0 = float(probs_0[original_pred])

    ff_hard = False
    ff_soft = 0.0

    for k in epochs[1:]:
        epoch_k = unlearning_results[str(k)]
        probs_k = np.array(epoch_k["probs"])
        pred_k = int(np.argmax(probs_k))

        if pred_k != original_pred:
            ff_hard = True

        mass_shift = p_original_0 - float(probs_k[original_pred])
        ff_soft = max(ff_soft, mass_shift)

    return ff_hard, ff_soft


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_file",  default="data/pilot_fur_results.jsonl",
                        help="JSONL output from run_fur_pilot.py or furV2.py")
    parser.add_argument("--output_file", default="data/pilot_fur_labels.json",
                        help="Where to write the per-question label dict")
    args = parser.parse_args()

    # Read all (question, step) records
    records = []
    with open(args.input_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("unlearning_results") is None:
                continue
            records.append(rec)

    print(f"Loaded {len(records)} (question, step) records from {args.input_file}")

    # Group by question id
    by_question = defaultdict(list)
    for rec in records:
        by_question[rec["id"]].append(rec)

    labels = {}
    print(f"\n{'ID':<25} {'steps':>5} {'ff_hard':>8} {'ff_soft':>10}  steps_detail")
    print("-" * 80)

    for qid, steps in sorted(by_question.items()):
        steps_sorted = sorted(steps, key=lambda r: r["step_idx"])
        step_labels = []

        for rec in steps_sorted:
            fh, fs = compute_step_labels(rec["unlearning_results"])
            step_labels.append({
                "step_idx": rec["step_idx"],
                "cot_step": rec.get("cot_step", ""),
                "ff_hard": fh,
                "ff_soft": round(fs, 6),
            })

        q_ff_hard = any(sl["ff_hard"] for sl in step_labels)
        q_ff_soft = max(sl["ff_soft"] for sl in step_labels)

        labels[qid] = {
            "id": qid,
            "question": steps_sorted[0]["question"],
            "correct": steps_sorted[0]["correct"],
            "initial_pred_idx": int(np.argmax(steps_sorted[0]["initial_probs"])),
            "ff_hard": q_ff_hard,
            "ff_soft": round(q_ff_soft, 6),
            "steps": step_labels,
        }

        step_summary = " | ".join(
            f"s{sl['step_idx']}:{'H' if sl['ff_hard'] else '-'}{sl['ff_soft']:.3f}"
            for sl in step_labels
        )
        print(f"{qid:<25} {len(step_labels):>5} {str(q_ff_hard):>8} {q_ff_soft:>10.4f}  {step_summary}")

    # Write output
    with open(args.output_file, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"\nLabels for {len(labels)} question(s) written to {args.output_file}")


if __name__ == "__main__":
    main()
