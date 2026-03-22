"""Join anchor trajectory embeddings with FUR faithfulness labels.

Reads:
  - data/pilot_trajectories.pkl  (from extract_anchor_embeddings.py)
  - data/pilot_fur_labels.json   (from compute_fur_labels.py)

Joins on question ID (the 'id' field).  Prints a confirmation table and
saves the merged dict to data/pilot_joined.pkl.

Usage:
    python join_pilot_data.py \\
        --trajectories data/pilot_trajectories.pkl \\
        --labels       data/pilot_fur_labels.json  \\
        --output       data/pilot_joined.pkl
"""

import argparse
import json
import pickle

import numpy as np


ANCHOR_NAMES = [
    "Answer_A",
    "Reasoning_A",
    "Answer_B",
    "Reasoning_B",
    "Answer_C",
    "Reasoning_C",
    "Answer_D",
    "Reasoning_D",
    "Final_Answer",
]


def load_trajectories(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_labels(path):
    with open(path) as f:
        return json.load(f)


def norm(v):
    """L2 norm of a 1-D numpy array."""
    return float(np.linalg.norm(v))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trajectories", default="data/pilot_trajectories.pkl")
    parser.add_argument("--labels",       default="data/pilot_fur_labels.json")
    parser.add_argument("--output",       default="data/pilot_joined.pkl")
    args = parser.parse_args()

    trajectories = load_trajectories(args.trajectories)
    labels = load_labels(args.labels)

    # trajectories is a list of dicts; extract_anchor_embeddings.py stores
    # the question id under 'question_id' (not 'id')
    traj_by_id = {t["question_id"]: t for t in trajectories}

    print(f"Trajectories loaded : {len(traj_by_id)} questions")
    print(f"FUR labels loaded   : {len(labels)} questions")

    joined = {}
    missing_traj  = []
    missing_label = []

    for qid in set(list(traj_by_id.keys()) + list(labels.keys())):
        if qid not in traj_by_id:
            missing_traj.append(qid)
            continue
        if qid not in labels:
            missing_label.append(qid)
            continue
        joined[qid] = {**traj_by_id[qid], **labels[qid]}

    if missing_traj:
        print(f"\nWARNING — no trajectory for: {missing_traj}")
    if missing_label:
        print(f"WARNING — no FUR label for:   {missing_label}")

    # ------------------------------------------------------------------
    # Confirmation table
    # ------------------------------------------------------------------
    print(f"\nJoined {len(joined)} question(s)\n")

    header = (f"{'ID':<25}  {'ff_hard':>8}  {'ff_soft':>8}  "
              f"{'n_anchors':>9}  {'mean_disp_norm':>14}")
    print(header)
    print("-" * len(header))

    for qid, d in sorted(joined.items()):
        # anchor_embeddings: dict of {anchor_name: np.ndarray | None}
        ae = d.get("anchor_embeddings", {})
        n_anchors = sum(1 for v in ae.values() if v is not None)

        # displacement_vectors: dict of named vectors from extract_anchor_embeddings.py
        dv = d.get("displacement_vectors", {})
        valid_disps = [v for v in dv.values() if v is not None]
        mean_disp = float(np.mean([norm(v) for v in valid_disps])) if valid_disps else float("nan")

        print(f"{qid:<25}  {str(d['ff_hard']):>8}  {d['ff_soft']:>8.4f}  "
              f"{n_anchors:>9}  {mean_disp:>14.4f}")

    print()

    # ------------------------------------------------------------------
    # Print per-step flip details for each question
    # ------------------------------------------------------------------
    for qid, d in sorted(joined.items()):
        print(f"  {qid}  ff_hard={d['ff_hard']}  ff_soft={d['ff_soft']:.4f}")
        for sl in d.get("steps", []):
            marker = "FLIP" if sl["ff_hard"] else "    "
            step_preview = sl["cot_step"][:60].replace("\n", " ")
            print(f"    [{marker}] step {sl['step_idx']}  soft={sl['ff_soft']:.4f}  "
                  f"\"{step_preview}\"")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    with open(args.output, "wb") as f:
        pickle.dump(joined, f)

    print(f"\nJoined data saved to {args.output}")


if __name__ == "__main__":
    main()
