#!/usr/bin/env python3
"""
Phase 2: NPO unlearning adapted for MCQ answer-block reasoning steps.

For each MCQ CoT instance and each answer-block step, unlearns the step
via Negative Preference Optimization (NPO) and checks whether the model's
final answer flips.  Steps where the answer flips are classified as
*faithful*; steps where it does not are *unfaithful*.

Usage:
    python mcq_unlearn.py \
        --model_name meta-llama/Meta-Llama-3-8B-Instruct \
        --data_file data/mcq_cots_fur.jsonl \
        --epochs 5 --lr 1e-5
"""

import sys
import os

FUR_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fur", "parametric-faithfulness-2",
)
if FUR_PATH not in sys.path:
    sys.path.insert(0, FUR_PATH)

import argparse
import json
import gc
import random

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from data import (
    FRCollator,
    SegmentOTFDataset,
)
from unlearn import (
    compute_loss,
    get_linear_schedule_with_warmup,
)
from evaluate import (
    answer_probabilities,
    completion_probabilities,
    complete,
    generation_fixed_cot,
)
from util import set_random_seed
from mcq_dataload import (
    MCQDataHandler,
    segment_mcq_cot,
    load_mcq_fur_data,
)


# ──────────────────────────────────────────────────────────────────────
# Segmentation strategy for MCQ answer blocks
# ──────────────────────────────────────────────────────────────────────

def mcq_make_targets(cot_dict: dict) -> list:
    """
    Build forget targets from an MCQ CoT using answer-block segmentation.
    Each answer block (Premise/Reasoning/Conclusion) becomes a
    (completion, prefix) pair.
    """
    prompt = cot_dict["cot_prompt"]
    segments = cot_dict.get("segmented_cot")
    if segments is None:
        segments = segment_mcq_cot(cot_dict["cot"])

    targets = []
    prefixes: list = []
    for seg in segments:
        targets.append({
            "prompt": prompt,
            "completion": seg,
            "prefix": "\n".join(prefixes) if prefixes else None,
        })
        prefixes.append(seg)
    return targets


def mcq_cot_to_otfd(
    target: dict,
    all_cots: list,
    tokenizer,
    n_retain: int = 4,
    stepwise: bool = True,
    step_idx: int = 0,
):
    """Build a SegmentOTFDataset for MCQ answer-step unlearning."""
    pool = [c for c in all_cots if c["id"] != target["id"]]

    forget_targets = mcq_make_targets(target)
    retain_pool = random.sample(pool, min(n_retain, len(pool)))
    retain_targets = []
    for r in retain_pool:
        retain_targets.extend(mcq_make_targets(r))

    return SegmentOTFDataset(
        forget_targets,
        retain_targets,
        tokenizer,
        stepwise=stepwise,
        step_idx=step_idx,
    )


# ──────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────

def evaluate_mcq(model, tokenizer, dh, target, step_idx):
    """
    Evaluate the model after (or before) unlearning a specific answer step.
    Returns a dict with probabilities, predictions, and new CoT.
    """
    model.eval()

    cot_text = target["cot"]
    cot_prefix = dh.make_cot_prompt(target["raw_instance"])
    cot_probability = completion_probabilities(
        model, tokenizer, cot_prefix, [cot_text]
    )

    segments = target.get("segmented_cot")
    if segments is None:
        segments = segment_mcq_cot(cot_text)

    unlearned_step = segments[step_idx] if step_idx < len(segments) else ""
    previous_steps = segments[:step_idx]

    if previous_steps:
        step_prefix = "\n".join([cot_prefix] + previous_steps)
    else:
        step_prefix = cot_prefix

    step_probability = completion_probabilities(
        model, tokenizer, step_prefix, [unlearned_step]
    )

    _, probs_after, prediction_after = answer_probabilities(
        model, tokenizer, dh, target["raw_instance"]
    )

    new_cot = complete(model, tokenizer, dh.make_cot_prompt(target["raw_instance"]))
    new_cot_probs, _ = generation_fixed_cot(
        model, tokenizer, dh, target["raw_instance"], new_cot
    )

    return {
        "probs": probs_after.tolist(),
        "prediction": prediction_after,
        "target_cot_step": unlearned_step,
        "new_cot": new_cot,
        "new_cot_probs": new_cot_probs.tolist(),
        "cot_prob": cot_probability.detach().cpu().float().numpy().tolist(),
        "cot_step_prob": step_probability.detach().cpu().float().numpy().tolist(),
    }


# ──────────────────────────────────────────────────────────────────────
# Single-instance unlearning loop
# ──────────────────────────────────────────────────────────────────────

def unlearn_single_step(
    model_id: str,
    tokenizer,
    args,
    target: dict,
    step_idx: int,
    all_cots: list,
    dh: MCQDataHandler,
):
    """
    Unlearn a single answer-block step from one MCQ CoT instance.
    Returns per-epoch evaluation results.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map="auto",
    )
    oracle_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map="auto",
    )
    device = model.device
    collator = FRCollator(tokenizer, device=device)

    dataset = mcq_cot_to_otfd(
        target, all_cots, tokenizer,
        n_retain=4, stepwise=True, step_idx=step_idx,
    )

    n_targets = dataset.num_targets()
    print(f"  Num forget targets: {n_targets}")
    if n_targets <= 2:
        print("  Too few targets, skipping")
        del model, oracle_model
        gc.collect()
        torch.cuda.empty_cache()
        return None

    epochs = args.epochs
    loader = DataLoader(dataset, batch_size=1, collate_fn=collator, shuffle=True)
    max_steps = epochs * len(dataset)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=max_steps,
    )

    results_per_epoch = {}
    results_per_epoch[0] = evaluate_mcq(model, tokenizer, dh, target, step_idx)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        for batch in loader:
            loss = compute_loss(
                model, oracle_model, batch, loss_type=args.method,
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        epoch_result = evaluate_mcq(model, tokenizer, dh, target, step_idx)
        results_per_epoch[epoch + 1] = epoch_result

    del collator, loader, dataset, scheduler, optimizer, model, oracle_model
    gc.collect()
    torch.cuda.empty_cache()

    return results_per_epoch


# ──────────────────────────────────────────────────────────────────────
# Faithfulness classification
# ──────────────────────────────────────────────────────────────────────

def classify_faithfulness(results_per_epoch: dict) -> bool:
    """
    A reasoning step is *faithful* if unlearning it causes the model's
    prediction to change relative to the pre-unlearning prediction.
    """
    if results_per_epoch is None:
        return None

    initial_pred = results_per_epoch[0]["prediction"]
    for epoch_key in sorted(results_per_epoch.keys()):
        if epoch_key == 0:
            continue
        if results_per_epoch[epoch_key]["prediction"] != initial_pred:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="NPO unlearning for MCQ CoT answer-block steps"
    )
    ap.add_argument("--model_name", type=str,
                    default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--data_file", type=str,
                    default="data/mcq_cots_fur.jsonl")
    ap.add_argument("--method", type=str, default="npo_KL")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_instances", type=int, default=50,
                    help="Max instances to process")
    ap.add_argument("--output_file", type=str,
                    default="data/mcq_faithfulness.jsonl")
    ap.add_argument("--system_prompt_file", type=str,
                    default="context/structured_reasoning_prompt.tx")
    args = ap.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_random_seed(args.seed)

    model_id = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dh = MCQDataHandler(system_prompt_path=args.system_prompt_file)
    all_cots = load_mcq_fur_data(args.data_file)
    random.shuffle(all_cots)

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    processed = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            for line in f:
                rec = json.loads(line)
                processed.add(rec.get("id", ""))

    for idx, target in enumerate(all_cots[:args.max_instances]):
        segments = target.get("segmented_cot")
        if segments is None:
            segments = segment_mcq_cot(target["cot"])
            target["segmented_cot"] = segments

        n_steps = len(segments)

        print(f"\n[{idx + 1}/{min(len(all_cots), args.max_instances)}] "
              f"Instance: {target['id']} ({n_steps} steps)")

        step_faithfulness = {}
        for step_idx in range(n_steps):
            check_id = f"{target['id']}_step{step_idx}"
            if check_id in processed:
                print(f"  Step {step_idx}: already processed, skipping")
                continue

            step_label = _step_label(segments, step_idx)
            print(f"  Unlearning step {step_idx} ({step_label})...")

            results = unlearn_single_step(
                model_id, tokenizer, args,
                target, step_idx, all_cots, dh,
            )

            is_faithful = classify_faithfulness(results)
            step_faithfulness[step_label] = is_faithful

            record = {
                "id": check_id,
                "instance_id": target["id"],
                "step_idx": step_idx,
                "step_label": step_label,
                "faithful": is_faithful,
                "correct_letter": target.get("correct_letter", ""),
                "unlearning_results": results,
            }

            with open(args.output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"  Step faithfulness: {step_faithfulness}")


def _step_label(segments: list, step_idx: int) -> str:
    """Derive a human-readable label for a segment (e.g. 'Answer_A' or 'Final')."""
    if step_idx >= len(segments):
        return "unknown"
    seg = segments[step_idx].strip()
    import re
    m = re.match(r"Answer\s+([A-Za-z])\s*:", seg)
    if m:
        return f"Answer_{m.group(1).upper()}"
    if "**Final Answer**" in seg or re.match(r"Answer\s*:", seg):
        return "Final"
    return f"step_{step_idx}"


if __name__ == "__main__":
    main()
