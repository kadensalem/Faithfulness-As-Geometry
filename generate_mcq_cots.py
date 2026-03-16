#!/usr/bin/env python3
"""
Phase 1 of the MCQ CoT Faithfulness-as-Geometry pipeline.

Loads OpenBookQA from HuggingFace, prompts a model with the structured
Premise/Reasoning/Conclusion format, parses the output into per-answer
segments, and writes datasets compatible with both the geometry trajectory
pipeline and the FUR unlearning pipeline.

Usage:
    python generate_mcq_cots.py \
        --hf_model meta-llama/Meta-Llama-3-8B-Instruct \
        --max_instances 200 --split test
"""

import argparse
import json
import os
import re
import random
from typing import List, Tuple, Optional, Dict, Any

from datasets import load_dataset

from mcq_dataload import MCQDataHandler, segment_mcq_cot


# ──────────────────────────────────────────────────────────────────────
# Model loading and generation
# ──────────────────────────────────────────────────────────────────────

def load_hf_model(model_id: str, device: str = "cuda:0"):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def prompt_model_for_cot(
    model, tokenizer, prompt_text: str, max_new_tokens: int = 2048,
) -> str:
    import torch

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


# ──────────────────────────────────────────────────────────────────────
# Response parsing
# ──────────────────────────────────────────────────────────────────────

def parse_model_cot(response: str) -> Tuple[List[str], Optional[str]]:
    """
    Parse model-generated MCQ CoT into answer-block steps and extract
    the final answer letter.
    """
    steps = segment_mcq_cot(response)

    answer_match = re.search(
        r"\*\*Final Answer\*\*\s*\n?\s*Answer\s*:\s*([A-Za-z])",
        response,
    )
    if answer_match is None:
        answer_match = re.search(r"Answer\s*:\s*([A-Za-z])\s*$", response)
    model_answer = answer_match.group(1).upper() if answer_match else None

    return steps, model_answer


# ──────────────────────────────────────────────────────────────────────
# Dataset generation
# ──────────────────────────────────────────────────────────────────────

def generate_dataset(
    dh: MCQDataHandler,
    model_id: str,
    split: str = "test",
    max_instances: int = 200,
    device: str = "cuda:0",
    seed: int = 42,
) -> Tuple[Dict, List[dict]]:
    """
    Generate the MCQ CoT dataset.

    Returns:
        geometry_data: dict keyed by dataset_key, compatible with geometry pipeline
        fur_data:      list of dicts, compatible with FUR pipeline
    """
    random.seed(seed)

    print(f"[INFO] Loading model: {model_id}")
    tokenizer, model = load_hf_model(model_id, device)

    _, validation, test = dh.get_dataset_splits()
    split_data = test if split == "test" else validation
    if split_data is None:
        raise RuntimeError(f"Split '{split}' not available for this dataset")

    indices = list(range(len(split_data)))
    random.shuffle(indices)
    indices = indices[:max_instances]

    dataset_key = "openbook"
    geometry_data: Dict[str, list] = {dataset_key: []}
    fur_data: List[dict] = []

    for count, idx in enumerate(indices):
        instance = split_data[idx]
        instance_id = instance.get("id", str(idx))
        correct_letter = dh.correct_answer_letter(instance)
        prompt_text = dh.make_cot_prompt(instance)

        print(f"[{count + 1}/{len(indices)}] id={instance_id}  correct={correct_letter}")

        response = prompt_model_for_cot(model, tokenizer, prompt_text)
        steps, model_answer = parse_model_cot(response)

        if not steps:
            print(f"  [WARN] Parsing failed for {instance_id}, skipping")
            continue

        full_cot = "\n".join(steps)
        is_faithful = (model_answer == correct_letter) if model_answer else None

        geo_record = {
            "topic": instance_id,
            "lang": "en",
            "steps": steps,
            "question": instance.get("question_stem", ""),
            "correct_letter": correct_letter,
            "model_answer": model_answer,
            "faithful": is_faithful,
        }
        geometry_data[dataset_key].append(geo_record)

        answer_choices = dh.get_answer_choices(instance)
        fur_record = {
            "id": f"{dataset_key}_{instance_id}",
            "question": instance.get("question_stem", ""),
            "correct_letter": correct_letter,
            "cot_prompt": prompt_text,
            "cot": full_cot,
            "options": answer_choices,
            "segmented_cot": steps,
            "model_answer": model_answer,
            "faithful": is_faithful,
            "raw_instance": {
                "id": instance_id,
                "question_stem": instance.get("question_stem", ""),
                "choices": {
                    "label": instance["choices"]["label"],
                    "text": instance["choices"]["text"],
                },
                "answerKey": correct_letter,
            },
        }
        fur_data.append(fur_record)

    return geometry_data, fur_data


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate MCQ CoT datasets for Faithfulness-as-Geometry"
    )
    ap.add_argument("--hf_model", type=str, required=True)
    ap.add_argument("--split", type=str, default="test",
                    choices=["test", "validation"])
    ap.add_argument("--max_instances", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--system_prompt_file", type=str,
                    default="context/structured_reasoning_prompt.tx")
    ap.add_argument("--output_dir", type=str, default="data")
    args = ap.parse_args()

    dh = MCQDataHandler(system_prompt_path=args.system_prompt_file)
    os.makedirs(args.output_dir, exist_ok=True)

    geometry_data, fur_data = generate_dataset(
        dh=dh,
        model_id=args.hf_model,
        split=args.split,
        max_instances=args.max_instances,
        device=args.device,
        seed=args.seed,
    )

    geo_path = os.path.join(args.output_dir, "mcq_cots.json")
    with open(geo_path, "w", encoding="utf-8") as f:
        json.dump(geometry_data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Geometry data saved to {geo_path}")

    fur_path = os.path.join(args.output_dir, "mcq_cots_fur.jsonl")
    with open(fur_path, "w", encoding="utf-8") as f:
        for record in fur_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[INFO] FUR data saved to {fur_path}")

    total = sum(len(v) for v in geometry_data.values())
    n_faithful = sum(
        1 for v in geometry_data.values()
        for r in v if r.get("faithful") is True
    )
    print(f"\n[SUMMARY]")
    print(f"  Total instances: {total}")
    print(f"  Faithful (correct answer): {n_faithful}")
    print(f"  Unfaithful / unknown: {total - n_faithful}")


if __name__ == "__main__":
    main()
