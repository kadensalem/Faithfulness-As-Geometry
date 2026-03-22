"""Qualitative display of the best FF-HARD example from the pilot.

For the question with the most steps where FF-HARD=True (i.e. the
unlearning actually flipped the model's answer), prints:

  - Question text and answer choices
  - Correct answer and model's original prediction
  - For each of the 9 anchor points:
      * Anchor name
      * L2 norm of the displacement vector arriving at this anchor
      * Whether this anchor's step had FF-HARD=True (CoT step faithfulness)
  - Per-step flip summary (step text, epoch-0 pred, last-epoch pred)

Alignment check: for each of the 4 answer blocks, check whether the
displacement vector points from Answer_X toward Reasoning_X (premise-
to-reasoning transition) in the same general direction as the
Conclusion_X decision probe (if available).

Usage:
    python qualitative_example.py \\
        --joined data/pilot_joined.pkl \\
        --layer  mid
"""

import argparse
import pickle

import numpy as np


ANCHOR_NAMES = [
    "Answer_A", "Reasoning_A",
    "Answer_B", "Reasoning_B",
    "Answer_C", "Reasoning_C",
    "Answer_D", "Reasoning_D",
    "Final_Answer",
]

LETTERS = ["A", "B", "C", "D"]


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def norm(v):
    return float(np.linalg.norm(v))


def pick_best_example(joined):
    """Return the question with the most FF-HARD steps; break ties by ff_soft."""
    best = None
    best_score = (-1, -1.0)
    for qid, d in joined.items():
        steps = d.get("steps", [])
        hard_count = sum(1 for s in steps if s["ff_hard"])
        score = (hard_count, d.get("ff_soft", 0.0))
        if score > best_score:
            best_score = score
            best = (qid, d)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--joined", default="data/pilot_joined.pkl")
    parser.add_argument("--layer",  default="mid",
                        choices=["mid", "last"],
                        help="Which layer's anchor embeddings to display")
    args = parser.parse_args()

    with open(args.joined, "rb") as f:
        joined = pickle.load(f)

    result = pick_best_example(joined)
    if result is None:
        print("No data found in joined file.")
        return

    qid, d = result
    steps_with_hard = [s for s in d.get("steps", []) if s["ff_hard"]]

    print("=" * 70)
    print(f"BEST FF-HARD EXAMPLE:  {qid}")
    print("=" * 70)
    print(f"Question : {d.get('question_text', d.get('question', ''))}")
    print()

    choices = d.get("answer_choices", {})
    for letter in LETTERS:
        marker = "<-- CORRECT" if letter == d.get("correct_answer", d.get("correct", "")) else ""
        print(f"  ({letter}) {choices.get(letter, '')}  {marker}")

    correct = d.get("correct_answer", d.get("correct", "?"))
    initial_pred_idx = d.get("initial_pred_idx", -1)
    initial_pred = LETTERS[initial_pred_idx] if 0 <= initial_pred_idx < 4 else "?"
    faithful_str = "faithful" if initial_pred == correct else "UNFAITHFUL"
    print(f"\nCorrect: {correct}   Model pred (no-CoT): {initial_pred}  [{faithful_str}]")
    print(f"FF-HARD: {d['ff_hard']}   FF-SOFT: {d['ff_soft']:.4f}")
    print(f"Steps with flip: {len(steps_with_hard)}/{len(d.get('steps', []))}")

    # ------------------------------------------------------------------
    # 9-anchor trajectory
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("9-ANCHOR TRAJECTORY  (anchor → displacement L2 norm)")
    print("-" * 70)

    ae = d.get("anchor_embeddings", {})   # middle-layer anchor embeddings
    dv = d.get("displacement_vectors", {})

    # Map step_idx → ff_hard for annotation
    step_hard = {s["step_idx"]: s["ff_hard"] for s in d.get("steps", [])}

    # step_idx assignment per anchor (5-block MCQ segmentation):
    #   Answer_A / Reasoning_A → step 0
    #   Answer_B / Reasoning_B → step 1  ... etc.
    #   Final_Answer            → step 4
    anchor_to_step = {
        "Answer_A": 0, "Reasoning_A": 0,
        "Answer_B": 1, "Reasoning_B": 1,
        "Answer_C": 2, "Reasoning_C": 2,
        "Answer_D": 3, "Reasoning_D": 3,
        "Final_Answer": 4,
    }

    # Displacement vectors that "arrive" at each anchor
    # premise_delta_X = Reasoning_X − Answer_X  → arrives at Reasoning_X
    # reasoning_delta_X = Answer_{X+1} − Reasoning_X → arrives at Answer_{X+1}
    # final_answer_delta → arrives at Final_Answer
    anchor_disp_key = {
        "Answer_A":     None,             # no displacement arriving here (start)
        "Reasoning_A":  "premise_delta_A",
        "Answer_B":     "reasoning_delta_A",
        "Reasoning_B":  "premise_delta_B",
        "Answer_C":     "reasoning_delta_B",
        "Reasoning_C":  "premise_delta_C",
        "Answer_D":     "reasoning_delta_C",
        "Reasoning_D":  "premise_delta_D",
        "Final_Answer": "final_answer_delta",
    }

    for anchor in ANCHOR_NAMES:
        emb = ae.get(anchor)
        disp_key = anchor_disp_key[anchor]
        disp_vec = dv.get(disp_key) if disp_key else None
        disp_str = f"{norm(disp_vec):7.3f}" if disp_vec is not None else "    N/A"
        step_idx = anchor_to_step[anchor]
        flip_str = " [FLIP]" if step_hard.get(step_idx) else ""
        found_str = "ok" if emb is not None else "MISSING"
        print(f"  {anchor:<15}  disp_norm={disp_str}  emb={found_str}{flip_str}")

    # ------------------------------------------------------------------
    # Alignment check: does premise_delta_X align with conclusion probe?
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("ALIGNMENT CHECK  (cosine between premise_delta_X and Conclusion_X)")
    print("-" * 70)

    conclusion_embs = d.get("conclusion_embeddings", {})

    for letter in LETTERS:
        pd = dv.get(f"premise_delta_{letter}")
        ce = conclusion_embs.get(letter)
        if pd is not None and ce is not None:
            cos = cosine(pd, ce)
            direction = "aligned" if cos > 0.1 else ("anti-aligned" if cos < -0.1 else "orthogonal")
            print(f"  premise_delta_{letter} · Conclusion_{letter} = {cos:+.4f}  [{direction}]")
        else:
            missing = []
            if pd is None: missing.append("premise_delta")
            if ce is None: missing.append("conclusion_emb")
            print(f"  Answer block {letter}: MISSING {', '.join(missing)}")

    # ------------------------------------------------------------------
    # Per-step flip summary
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("PER-STEP UNLEARNING SUMMARY")
    print("-" * 70)

    for s in sorted(d.get("steps", []), key=lambda x: x["step_idx"]):
        flip = "FLIP" if s["ff_hard"] else "    "
        step_text = s["cot_step"][:80].replace("\n", " ")
        print(f"  [{flip}] step {s['step_idx']}  soft={s['ff_soft']:.4f}  \"{step_text}\"")

    print()


if __name__ == "__main__":
    main()
