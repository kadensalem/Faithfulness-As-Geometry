#!/usr/bin/env python3
"""
Phase 3: Modified hidden-state trajectory extraction for the
Faithfulness-as-Geometry pipeline.

Key differences from the original cot-hidden-dynamic.py:
  1. --layer_index / --layer_indices: extract from a specific transformer
     layer instead of the last hidden state.
  2. --pooling anchor_last: extract the hidden state at the final token
     of the '* Result:' line (or '**Final Answer:**') instead of
     mean-pooling over the step.
  3. --boolean: use **Node [XX]** block boundaries instead of sentence
     splitting.

Usage:
    python geometry/cot-hidden-dynamic-v2.py \
        --hf_model meta-llama/Meta-Llama-3-8B-Instruct \
        --data_file data/boolean_cots.json \
        --pooling anchor_last --accumulation cumulative \
        --layer_index 20 --boolean \
        --similarity_order 1 \
        --save_dir results/boolean_trajectories
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Literal, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from utils import split_cot_steps, split_boolean_nodes
from utils_stat import (
    pairwise_similarity,
    pairwise_menger_curvature_similarity,
    plot_similarity_heatmap,
)
from utils import plot_trajectories_pca


# ──────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────

@dataclass
class LogicItem:
    logic: str
    topic: Optional[str]
    steps: List[str]
    faithful: Optional[bool] = None


def load_dataset_any_logic(path: str, boolean: bool = False) -> List[LogicItem]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items: List[LogicItem] = []
    splitter = split_boolean_nodes if boolean else split_cot_steps
    for logic_key, seq_list in data.items():
        if not isinstance(seq_list, list):
            continue
        for rec in seq_list:
            if not isinstance(rec, dict) or "steps" not in rec:
                continue
            steps = rec.get("steps", [])
            if isinstance(steps, str):
                steps = splitter(steps)
            elif isinstance(steps, list) and len(steps) == 1 and isinstance(steps[0], str):
                steps = splitter(steps[0])
            topic_val = rec.get("topic", None)
            topic_str: Optional[str] = str(topic_val) if topic_val is not None else None
            faithful = rec.get("faithful", None)
            items.append(LogicItem(
                logic=str(logic_key), topic=topic_str,
                steps=steps, faithful=faithful,
            ))
    return items


# ──────────────────────────────────────────────────────────────────────
# Anchor-position helpers
# ──────────────────────────────────────────────────────────────────────

_RESULT_RE = re.compile(r"\*\s*Result:\s*'(True|False)'")
_ANSWER_RE = re.compile(r"\*\*Final Answer:\s*(True|False)\*\*")


def _find_anchor_suffix(step: str) -> str:
    """
    Return the substring ending at the anchor point (Result line or
    Final Answer line) within a single step.
    """
    m = _RESULT_RE.search(step)
    if m:
        return step[: m.end()]
    m = _ANSWER_RE.search(step)
    if m:
        return step[: m.end()]
    return step


# ──────────────────────────────────────────────────────────────────────
# Step representation via token hidden states
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def step_vectors_for_sequence(
    tokenizer: AutoTokenizer,
    model,
    steps: List[str],
    *,
    pooling: Literal[
        "step_mean", "context_mean", "last",
        "context_aware_mean", "anchor_last",
    ] = "step_mean",
    accumulation: Literal["cumulative", "isolated"] = "cumulative",
    context_aware_k: int = 16,
    layer_index: Optional[int] = None,
    device: str = "cpu",
) -> List[np.ndarray]:
    """
    Extract per-step embeddings from a sequence of reasoning steps.

    New parameters vs. the original:
      layer_index:  if set, use outputs.hidden_states[layer_index]
                    instead of the last hidden state.
      pooling="anchor_last":  extract the hidden state at the final
                    token of the * Result / **Final Answer** line.
    """
    vecs: List[np.ndarray] = []
    prev_input_ids: torch.Tensor | None = None
    context = ""

    for t, step in enumerate(steps):
        if accumulation == "cumulative":
            context = step if t == 0 else (context + "\n" + step)
        else:
            context = step

        enc = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
        )

        if layer_index is not None:
            hs = outputs.hidden_states[layer_index]
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hs = outputs.last_hidden_state
        else:
            hs = outputs.hidden_states[-1]

        L = input_ids.shape[1]

        if pooling == "context_mean":
            v = hs.mean(dim=1).squeeze(0)
        elif pooling == "last":
            v = hs[:, -1, :].squeeze(0)
        elif pooling == "anchor_last":
            anchor_suffix = _find_anchor_suffix(step)
            if accumulation == "cumulative" and t > 0:
                anchor_ctx = "\n".join(steps[:t]) + "\n" + anchor_suffix
            else:
                anchor_ctx = anchor_suffix
            anchor_ids = tokenizer(
                anchor_ctx, add_special_tokens=False
            )["input_ids"]
            pos = min(len(anchor_ids) - 1, L - 1)
            pos = max(pos, 0)
            v = hs[:, pos, :].squeeze(0)
        else:
            if accumulation == "cumulative" and prev_input_ids is not None:
                prev_len = prev_input_ids.shape[1]
            else:
                prev_len = 0
            start = min(prev_len, L)
            step_slice = hs[:, start:, :] if start < L else hs[:, -1:, :]

            if pooling == "step_mean":
                v = step_slice.mean(dim=1).squeeze(0)
            else:
                k = max(0, int(context_aware_k))
                ctx_start = max(0, start - k)
                ctx_slice = hs[:, ctx_start:, :]
                v = ctx_slice.mean(dim=1).squeeze(0)

        vecs.append(v.detach().float().cpu().numpy().astype(np.float32))
        prev_input_ids = input_ids

    return vecs


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def build_label(item: LogicItem) -> str:
    if item.topic is None or str(item.topic).strip() == "":
        return f"{item.logic}:abstract"
    tag = "faithful" if item.faithful else ("unfaithful" if item.faithful is False else "")
    suffix = f":{tag}" if tag else ""
    return f"{item.logic}:{item.topic}{suffix}"


def group_by_logic(items: List[LogicItem]) -> Dict[str, List[LogicItem]]:
    groups: Dict[str, List[LogicItem]] = {}
    for it in items:
        groups.setdefault(it.logic, []).append(it)
    return groups


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_label(label: str) -> str:
    s = label.strip().lower()
    for ch in ["/", "\\", ":", " ", ",", "|", "*", "?", "\n", "\t", "(", ")", "[", "]"]:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("._")


def _resolve_layer_index(model, layer_index_arg: Optional[str]) -> Optional[int]:
    """
    Resolve --layer_index to an integer.
    Accepts an int literal or 'auto' (computes int(num_layers * 0.65)).
    """
    if layer_index_arg is None:
        return None
    if layer_index_arg.lower() == "auto":
        n = _count_layers(model)
        idx = int(n * 0.65)
        print(f"[INFO] Auto layer_index = {idx} (model has {n} layers)")
        return idx
    return int(layer_index_arg)


def _count_layers(model) -> int:
    config = model.config
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(config, attr):
            return getattr(config, attr)
    return len(model.encoder.layer) if hasattr(model, "encoder") else 32


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Hidden-state trajectory analysis (v2) — layer selection, anchor pooling, boolean nodes"
    )
    ap.add_argument("--hf_model", type=str, required=False)
    ap.add_argument("--hf_models", type=str, default=None)
    ap.add_argument("--data_file", type=str, default="data/boolean_cots.json")
    ap.add_argument("--boolean", action="store_true",
                     help="Use **Node [XX]** block boundaries instead of sentence splitting")
    ap.add_argument("--pooling", type=str, default="anchor_last",
                     choices=["step_mean", "context_mean", "last",
                              "context_aware_mean", "anchor_last"])
    ap.add_argument("--accumulation", type=str, default="cumulative",
                     choices=["cumulative", "isolated"])
    ap.add_argument("--context_aware_k", type=int, default=16)
    ap.add_argument("--layer_index", type=str, default="auto",
                     help="Layer to extract hidden states from (int, 'auto', or omit for last)")
    ap.add_argument("--layer_indices", type=str, default=None,
                     help="Comma-separated layers to sweep (overrides --layer_index)")
    ap.add_argument("--device", type=str,
                     default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default=None,
                     choices=["auto", "fp16", "fp32", "bf16"])
    ap.add_argument("--attn_implementation", type=str, default=None)
    ap.add_argument("--sections", type=str, default="all")
    ap.add_argument("--similarity_order", type=int, default=1)
    ap.add_argument("--save_dir", type=str, default="results/boolean_trajectories")
    ap.add_argument("--hide_axis_text", action="store_true")
    ap.add_argument("--color_scale", type=str, default="RdBu_r")
    ap.add_argument("--save_html", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.save_dir)

    items = load_dataset_any_logic(args.data_file, boolean=args.boolean)
    if args.sections != "all":
        keep = {s.strip() for s in args.sections.split(",") if s.strip()}
        items = [it for it in items if it.logic in keep]
    if not items:
        raise SystemExit("No items loaded. Check data_file or sections filter.")

    if not args.hf_model and not args.hf_models:
        raise SystemExit("Provide --hf_model or --hf_models.")

    model_ids: List[str]
    if args.hf_models:
        model_ids = [m.strip() for m in args.hf_models.split(",") if m.strip()]
    else:
        model_ids = [args.hf_model]

    layer_indices_to_sweep: List[Optional[str]]
    if args.layer_indices:
        layer_indices_to_sweep = [
            li.strip() for li in args.layer_indices.split(",") if li.strip()
        ]
    else:
        layer_indices_to_sweep = [args.layer_index]

    def _select_dtype() -> torch.dtype:
        if args.dtype in (None, "auto"):
            return torch.float16 if torch.cuda.is_available() else torch.float32
        return {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]

    def _safe_load_model(mid: str):
        tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
        if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
            tok.pad_token = tok.eos_token
        try:
            tok.padding_side = "left"
        except Exception:
            pass

        common_kw = {"trust_remote_code": True}
        if args.attn_implementation:
            common_kw["attn_implementation"] = args.attn_implementation

        mdl = None
        if args.load_in_4bit or args.load_in_8bit:
            qkw = dict(common_kw, device_map=args.device_map)
            if args.load_in_4bit:
                qkw["load_in_4bit"] = True
            if args.load_in_8bit:
                qkw["load_in_8bit"] = True
            try:
                mdl = AutoModel.from_pretrained(mid, **qkw)
            except Exception:
                try:
                    mdl = AutoModelForCausalLM.from_pretrained(mid, **qkw)
                except Exception as e:
                    print(f"[WARN] quantized load failed: {e}")

        if mdl is None:
            rkw = dict(common_kw, torch_dtype=_select_dtype())
            try:
                mdl = AutoModel.from_pretrained(mid, **rkw)
            except Exception:
                mdl = AutoModelForCausalLM.from_pretrained(mid, **rkw)
            mdl.to(args.device)

        mdl.eval()
        return tok, mdl

    for mid in model_ids:
        print(f"[INFO] Loading model: {mid}")
        tokenizer, model = _safe_load_model(mid)

        for li_str in layer_indices_to_sweep:
            layer_idx = _resolve_layer_index(model, li_str)
            layer_tag = f"layer{layer_idx}" if layer_idx is not None else "last"
            model_tag = mid.split("/")[-1].strip() or _safe_label(mid)

            print(f"[INFO] Extracting embeddings: model={model_tag}, layer={layer_tag}, pooling={args.pooling}")

            label2steps: Dict[str, List[np.ndarray]] = {}
            label2meta: Dict[str, dict] = {}
            total_items = len(items)

            for idx, it in enumerate(items, start=1):
                if idx == 1 or idx % 10 == 0 or idx == total_items:
                    print(f"[PROGRESS] {idx}/{total_items} ({idx / total_items * 100:.1f}%)")
                vecs = step_vectors_for_sequence(
                    tokenizer, model, it.steps,
                    pooling=args.pooling,
                    accumulation=args.accumulation,
                    context_aware_k=args.context_aware_k,
                    layer_index=layer_idx,
                    device=args.device,
                )
                label = build_label(it)
                label2steps[label] = vecs
                label2meta[label] = {
                    "logic": it.logic, "topic": it.topic,
                    "num_steps": len(it.steps), "faithful": it.faithful,
                }

            save_root = os.path.join(
                args.save_dir,
                _safe_label(f"{model_tag}_{layer_tag}_{args.pooling}"),
            )
            ensure_dir(save_root)

            data_root = os.path.join(save_root, "data")
            emb_dir = os.path.join(data_root, "embeddings")
            steps_dir = os.path.join(data_root, "steps")
            ensure_dir(emb_dir)
            ensure_dir(steps_dir)

            manifest = {
                "pooling": args.pooling,
                "accumulation": args.accumulation,
                "context_aware_k": int(args.context_aware_k),
                "similarity_order": int(args.similarity_order),
                "layer_index": layer_idx,
                "layer_tag": layer_tag,
                "boolean": args.boolean,
                "sections": args.sections,
                "hf_model": mid,
                "device": args.device,
                "labels": list(label2steps.keys()),
            }

            for it in items:
                label = build_label(it)
                safe = _safe_label(label)
                vecs = label2steps[label]
                arr = np.stack(vecs).astype(np.float32)
                np.save(os.path.join(emb_dir, f"{safe}.npy"), arr)
                with open(os.path.join(steps_dir, f"{safe}.json"), "w", encoding="utf-8") as f:
                    json.dump({
                        "label": label, "logic": it.logic,
                        "topic": it.topic, "steps": it.steps,
                        "faithful": it.faithful,
                    }, f, ensure_ascii=False, indent=2)

            with open(os.path.join(data_root, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)

            # PCA trajectory plots per logic group
            groups = group_by_logic(items)
            print(f"[INFO] PCA plots for {len(groups)} logic groups...")
            from sklearn.decomposition import PCA as _PCA

            for logic_key, group_items in groups.items():
                if not group_items:
                    continue
                sub = {build_label(it): label2steps[build_label(it)] for it in group_items}

                all_points = np.vstack([vec for embs in sub.values() for vec in embs])
                pca_model = _PCA(n_components=3, random_state=42).fit(all_points)
                pca_dir = os.path.join(data_root, "pca", logic_key)
                ensure_dir(pca_dir)
                np.savez_compressed(
                    os.path.join(pca_dir, "pca_model.npz"),
                    components_=pca_model.components_.astype(np.float32),
                    explained_variance_=pca_model.explained_variance_.astype(np.float32),
                    mean_=pca_model.mean_.astype(np.float32),
                )
                for label, embs in sub.items():
                    traj = np.vstack(embs)
                    proj = pca_model.transform(traj)
                    df_proj = pd.DataFrame({
                        "t": np.arange(1, proj.shape[0] + 1, dtype=int),
                        "pc1": proj[:, 0].astype(np.float32),
                        "pc2": proj[:, 1].astype(np.float32),
                        "pc3": proj[:, 2].astype(np.float32),
                    })
                    df_proj.to_csv(
                        os.path.join(pca_dir, f"{_safe_label(label)}_pca3d.csv"),
                        index=False,
                    )

                plot_trajectories_pca(
                    sub, exclude_prompt=False,
                    title=(
                        f"{logic_key} Trajectories (PCA) "
                        f"[{args.pooling}/{args.accumulation}/{layer_tag}] — {model_tag}"
                    ),
                    save_pdf_path=os.path.join(save_root, f"{logic_key}_trajectories_pca.pdf"),
                    save_html_path=(
                        os.path.join(save_root, f"{logic_key}_trajectories_pca.html")
                        if args.save_html else None
                    ),
                    width=1000, height=700,
                )

            # Global similarity heatmap
            order_sel = int(args.similarity_order)
            if order_sel == 3:
                labels_all, sim_all = pairwise_menger_curvature_similarity(
                    label2steps, metric="pearson", align="truncate",
                )
                title = f"Global Similarity (Menger curvature) [{layer_tag}] — {model_tag}"
                base_name = f"global_similarity_order3_menger_{layer_tag}"
            else:
                labels_all, sim_all = pairwise_similarity(
                    label2steps, order=order_sel, metric="mean_cos",
                )
                title = f"Global Similarity (order={order_sel}) [{layer_tag}] — {model_tag}"
                base_name = f"global_similarity_order{order_sel}_{layer_tag}"

            plot_similarity_heatmap(
                sim_all, labels=labels_all, title=title,
                width=1100, height=1000,
                save_pdf_path=os.path.join(save_root, f"{base_name}.pdf"),
                show_axis_text=not args.hide_axis_text,
                color_scale=args.color_scale,
            )

            df_sim = pd.DataFrame(sim_all, index=labels_all, columns=labels_all)
            df_sim.to_csv(os.path.join(data_root, f"{base_name}.csv"))
            np.save(os.path.join(data_root, f"{base_name}.npy"), sim_all.astype(np.float32))

            print(f"[DONE] Figures: {os.path.abspath(save_root)}")
            print(f"       Data:    {os.path.abspath(data_root)}")


if __name__ == "__main__":
    main()
