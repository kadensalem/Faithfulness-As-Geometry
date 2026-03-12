#!/usr/bin/env python3
"""
Phase 4: Compare geometric trajectories of faithful vs unfaithful CoTs.

Loads precomputed embeddings (from cot-hidden-dynamic-v2.py) and
faithfulness labels (from boolean_unlearn.py), then produces:
  - PCA trajectory plots grouped by faithful / unfaithful
  - Similarity matrices at orders 0–3
  - Group-averaged similarity statistics (within-faithful,
    within-unfaithful, cross-group)

Usage:
    python compare_trajectories.py \
        --embeddings_dir results/boolean_trajectories/<model_layer_pooling>/data/embeddings \
        --faithfulness_file data/boolean_faithfulness.jsonl \
        --geometry_data data/boolean_cots.json \
        --orders 0,1,2,3 \
        --save_dir results/comparison
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

GEOMETRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geometry")
if GEOMETRY_PATH not in sys.path:
    sys.path.insert(0, GEOMETRY_PATH)

from utils import plot_trajectories_pca
from utils_stat import (
    pairwise_similarity,
    pairwise_menger_curvature_similarity,
    plot_similarity_heatmap,
)


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def load_geometry_data(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_faithfulness_labels(path: str) -> Dict[str, Dict[str, Optional[bool]]]:
    """
    Load per-instance, per-step faithfulness from the unlearning output.
    Returns {instance_id: {step_label: bool, ...}}.
    """
    labels: Dict[str, dict] = {}
    if not os.path.exists(path):
        return labels
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            iid = rec.get("instance_id", "")
            step = rec.get("step_label", "")
            labels.setdefault(iid, {})[step] = rec.get("faithful")
    return labels


def load_embeddings(emb_dir: str) -> Dict[str, np.ndarray]:
    """Load .npy embedding files into {label: array [T, D]}."""
    embs = {}
    if not os.path.isdir(emb_dir):
        return embs
    for fname in sorted(os.listdir(emb_dir)):
        if not fname.endswith(".npy"):
            continue
        label = fname[:-4]
        embs[label] = np.load(os.path.join(emb_dir, fname))
    return embs


def load_step_metadata(steps_dir: str) -> Dict[str, dict]:
    """Load step JSON metadata."""
    meta = {}
    if not os.path.isdir(steps_dir):
        return meta
    for fname in sorted(os.listdir(steps_dir)):
        if not fname.endswith(".json"):
            continue
        label = fname[:-5]
        with open(os.path.join(steps_dir, fname), "r", encoding="utf-8") as f:
            meta[label] = json.load(f)
    return meta


# ──────────────────────────────────────────────────────────────────────
# Grouping
# ──────────────────────────────────────────────────────────────────────

def classify_embeddings_by_faithfulness(
    step_meta: Dict[str, dict],
    geometry_data: dict,
) -> Tuple[Dict[str, list], Dict[str, list]]:
    """
    Split labels into faithful / unfaithful groups based on the
    'faithful' field in geometry data records.
    """
    faithful_labels: Dict[str, list] = {}
    unfaithful_labels: Dict[str, list] = {}

    for logic_key, records in geometry_data.items():
        for rec in records:
            topic = rec.get("topic", "")
            faith = rec.get("faithful")
            label_candidates = [
                f"{logic_key}:{topic}:faithful",
                f"{logic_key}:{topic}:unfaithful",
                f"{logic_key}:{topic}",
            ]
            matched = None
            for lbl in step_meta:
                for cand in label_candidates:
                    safe_cand = _safe_label(cand)
                    if lbl == safe_cand or lbl.startswith(safe_cand):
                        matched = lbl
                        break
                if matched:
                    break

            if matched is None:
                continue

            if faith is True:
                faithful_labels.setdefault(logic_key, []).append(matched)
            elif faith is False:
                unfaithful_labels.setdefault(logic_key, []).append(matched)

    return faithful_labels, unfaithful_labels


def _safe_label(label: str) -> str:
    s = label.strip().lower()
    for ch in ["/", "\\", ":", " ", ",", "|", "*", "?", "\n", "\t", "(", ")", "[", "]"]:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("._")


# ──────────────────────────────────────────────────────────────────────
# Similarity analysis helpers
# ──────────────────────────────────────────────────────────────────────

def compute_group_similarities(
    embs: Dict[str, np.ndarray],
    faithful_labels: List[str],
    unfaithful_labels: List[str],
    order: int,
) -> dict:
    """
    Compute within-group and cross-group similarity statistics.
    """
    def _label_to_vecs(label):
        arr = embs.get(label)
        if arr is None:
            return None
        return [arr[i] for i in range(arr.shape[0])]

    all_labels = faithful_labels + unfaithful_labels
    label2vecs = {}
    for lbl in all_labels:
        vecs = _label_to_vecs(lbl)
        if vecs is not None:
            label2vecs[lbl] = vecs

    if len(label2vecs) < 2:
        return {"error": "too few sequences for similarity"}

    if order == 3:
        labels_out, sim = pairwise_menger_curvature_similarity(
            label2vecs, metric="pearson", align="truncate",
        )
    else:
        labels_out, sim = pairwise_similarity(
            label2vecs, order=order, metric="mean_cos",
        )

    label_to_idx = {l: i for i, l in enumerate(labels_out)}
    f_set = set(faithful_labels) & set(labels_out)
    u_set = set(unfaithful_labels) & set(labels_out)

    def _mean_pairs(group_a, group_b):
        vals = []
        for a in group_a:
            for b in group_b:
                if a == b:
                    continue
                ia, ib = label_to_idx.get(a), label_to_idx.get(b)
                if ia is not None and ib is not None:
                    vals.append(sim[ia, ib])
        return float(np.mean(vals)) if vals else None

    return {
        "within_faithful": _mean_pairs(f_set, f_set),
        "within_unfaithful": _mean_pairs(u_set, u_set),
        "cross_group": _mean_pairs(f_set, u_set),
        "n_faithful": len(f_set),
        "n_unfaithful": len(u_set),
        "labels": labels_out,
        "similarity_matrix": sim,
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Compare geometric trajectories: faithful vs unfaithful CoTs"
    )
    ap.add_argument("--embeddings_dir", type=str, required=True,
                     help="Directory containing .npy embedding files")
    ap.add_argument("--steps_dir", type=str, default=None,
                     help="Directory with step JSON metadata (defaults to sibling of embeddings_dir)")
    ap.add_argument("--geometry_data", type=str, default="data/boolean_cots.json")
    ap.add_argument("--faithfulness_file", type=str,
                     default="data/boolean_faithfulness.jsonl")
    ap.add_argument("--orders", type=str, default="0,1,2,3")
    ap.add_argument("--save_dir", type=str, default="results/comparison")
    ap.add_argument("--color_scale", type=str, default="RdBu_r")
    ap.add_argument("--save_html", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    orders = [int(o.strip()) for o in args.orders.split(",")]

    if args.steps_dir is None:
        parent = os.path.dirname(args.embeddings_dir)
        args.steps_dir = os.path.join(parent, "steps")

    print("[INFO] Loading data...")
    embs = load_embeddings(args.embeddings_dir)
    step_meta = load_step_metadata(args.steps_dir)
    geo_data = load_geometry_data(args.geometry_data)

    print(f"  Loaded {len(embs)} embedding sequences")
    print(f"  Loaded {len(step_meta)} step metadata files")

    faithful_map, unfaithful_map = classify_embeddings_by_faithfulness(
        step_meta, geo_data,
    )

    all_faithful = [l for labels in faithful_map.values() for l in labels]
    all_unfaithful = [l for labels in unfaithful_map.values() for l in labels]
    print(f"  Faithful sequences: {len(all_faithful)}")
    print(f"  Unfaithful sequences: {len(all_unfaithful)}")

    # ── PCA plots: faithful vs unfaithful per logic group ──

    print("\n[INFO] Generating PCA trajectory plots...")
    all_logic_keys = set(list(faithful_map.keys()) + list(unfaithful_map.keys()))

    for logic_key in sorted(all_logic_keys):
        f_labels = faithful_map.get(logic_key, [])
        u_labels = unfaithful_map.get(logic_key, [])

        trajectories = {}
        for lbl in f_labels:
            if lbl in embs:
                arr = embs[lbl]
                trajectories[f"F:{lbl}"] = [arr[i] for i in range(arr.shape[0])]
        for lbl in u_labels:
            if lbl in embs:
                arr = embs[lbl]
                trajectories[f"U:{lbl}"] = [arr[i] for i in range(arr.shape[0])]

        if len(trajectories) < 2:
            continue

        plot_trajectories_pca(
            trajectories, exclude_prompt=False,
            title=f"{logic_key} — Faithful (F) vs Unfaithful (U)",
            save_pdf_path=os.path.join(args.save_dir, f"{_safe_label(logic_key)}_pca.pdf"),
            save_html_path=(
                os.path.join(args.save_dir, f"{_safe_label(logic_key)}_pca.html")
                if args.save_html else None
            ),
            width=1000, height=700,
        )

    # ── Similarity analysis per order ──

    results_summary = {}
    for order in orders:
        print(f"\n[INFO] Similarity analysis (order={order})...")

        label2vecs = {}
        for lbl, arr in embs.items():
            label2vecs[lbl] = [arr[i] for i in range(arr.shape[0])]

        if len(label2vecs) < 2:
            print("  Skipping: too few sequences")
            continue

        if order == 3:
            labels_out, sim = pairwise_menger_curvature_similarity(
                label2vecs, metric="pearson", align="truncate",
            )
            title = f"Global Similarity (Menger curvature)"
        else:
            labels_out, sim = pairwise_similarity(
                label2vecs, order=order, metric="mean_cos",
            )
            title = f"Global Similarity (order={order})"

        base = f"similarity_order{order}"
        plot_similarity_heatmap(
            sim, labels=labels_out, title=title,
            width=1100, height=1000,
            save_pdf_path=os.path.join(args.save_dir, f"{base}.pdf"),
            color_scale=args.color_scale,
        )

        df_sim = pd.DataFrame(sim, index=labels_out, columns=labels_out)
        df_sim.to_csv(os.path.join(args.save_dir, f"{base}.csv"))

        stats = compute_group_similarities(
            embs, all_faithful, all_unfaithful, order,
        )
        results_summary[f"order_{order}"] = {
            "within_faithful": stats.get("within_faithful"),
            "within_unfaithful": stats.get("within_unfaithful"),
            "cross_group": stats.get("cross_group"),
            "n_faithful": stats.get("n_faithful"),
            "n_unfaithful": stats.get("n_unfaithful"),
        }

        print(f"  Within-faithful:   {stats.get('within_faithful')}")
        print(f"  Within-unfaithful: {stats.get('within_unfaithful')}")
        print(f"  Cross-group:       {stats.get('cross_group')}")

    # ── Save summary ──

    summary_path = os.path.join(args.save_dir, "comparison_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] Summary saved to {summary_path}")
    print(f"       Figures saved to {os.path.abspath(args.save_dir)}")


if __name__ == "__main__":
    main()
