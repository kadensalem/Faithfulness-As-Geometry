#!/usr/bin/env python3
"""
Phase 1 of the Boolean CoT Faithfulness-as-Geometry pipeline.

Generates random boolean expression trees with labeled node IDs, produces
structured chain-of-thought solutions (ground-truth or model-generated),
and outputs datasets compatible with both the geometry trajectory pipeline
and the FUR unlearning pipeline.

Usage:
    # Ground-truth CoTs (no model needed):
    python generate_boolean_cots.py --mode ground_truth \
        --num_structures 5 --instances_per 10 --depths 2,3,4

    # Model-generated CoTs:
    python generate_boolean_cots.py --mode model \
        --hf_model meta-llama/Meta-Llama-3-8B-Instruct \
        --num_structures 5 --instances_per 10 --depths 2,3
"""

import argparse
import json
import os
import re
import random
import itertools
from typing import List, Tuple, Optional, Dict, Any


# ──────────────────────────────────────────────────────────────────────
# Tree data structures
# ──────────────────────────────────────────────────────────────────────

BINARY_OPS = ["and", "or", "xor"]


class BoolLeaf:
    __slots__ = ("value",)

    def __init__(self, value: bool):
        self.value = value


class BoolOp:
    __slots__ = ("op", "left", "right")

    def __init__(self, op: str, left, right):
        self.op = op
        self.left = left
        self.right = right


def evaluate(node) -> bool:
    if isinstance(node, BoolLeaf):
        return node.value
    l = evaluate(node.left)
    r = evaluate(node.right)
    if node.op == "and":
        return l and r
    if node.op == "or":
        return l or r
    if node.op == "xor":
        return l ^ r
    raise ValueError(f"Unknown op: {node.op}")


# ──────────────────────────────────────────────────────────────────────
# Tree topology (structure without leaf values)
# ──────────────────────────────────────────────────────────────────────

class Topology:
    """Complete binary tree topology: ops at each internal node, leaves are placeholders."""
    __slots__ = ("op", "left", "right")

    def __init__(self, op: str, left: Optional["Topology"], right: Optional["Topology"]):
        self.op = op
        self.left = left
        self.right = right

    def num_leaves(self) -> int:
        l = 1 if self.left is None else self.left.num_leaves()
        r = 1 if self.right is None else self.right.num_leaves()
        return l + r

    def num_nodes(self) -> int:
        l = 0 if self.left is None else self.left.num_nodes()
        r = 0 if self.right is None else self.right.num_nodes()
        return 1 + l + r

    def ops_postorder(self) -> List[str]:
        ops = []
        if self.left is not None:
            ops.extend(self.left.ops_postorder())
        if self.right is not None:
            ops.extend(self.right.ops_postorder())
        ops.append(self.op)
        return ops

    def logic_key(self) -> str:
        return "boolTree_" + "_".join(self.ops_postorder())

    def depth(self) -> int:
        l = 0 if self.left is None else self.left.depth()
        r = 0 if self.right is None else self.right.depth()
        return 1 + max(l, r)

    def instantiate(self, leaf_values: List[bool]) -> BoolOp:
        idx = [0]

        def _fill(topo):
            if topo is None:
                v = leaf_values[idx[0]]
                idx[0] += 1
                return BoolLeaf(v)
            return BoolOp(
                op=topo.op,
                left=_fill(topo.left),
                right=_fill(topo.right),
            )

        return _fill(self)


def random_topology(depth: int, ops: List[str] = BINARY_OPS) -> Topology:
    if depth <= 1:
        return Topology(random.choice(ops), None, None)
    return Topology(
        random.choice(ops),
        random_topology(depth - 1, ops),
        random_topology(depth - 1, ops),
    )


# ──────────────────────────────────────────────────────────────────────
# Expression formatting
# ──────────────────────────────────────────────────────────────────────

def format_expression(node) -> Tuple[str, List[str]]:
    """
    Format a concrete tree as a labeled expression string.
    Returns (expression_string, node_ids_in_eval_order).
    """
    counter = [0]
    ids: List[str] = []

    def _fmt(n):
        if isinstance(n, BoolLeaf):
            return str(n.value)
        left_s = _fmt(n.left)
        right_s = _fmt(n.right)
        counter[0] += 1
        nid = f"[{counter[0]:02d}]"
        ids.append(nid)
        return f"{nid}: ({left_s} {n.op} {right_s})"

    expr = _fmt(node)
    return expr, ids


# ──────────────────────────────────────────────────────────────────────
# Ground-truth CoT generation
# ──────────────────────────────────────────────────────────────────────

def generate_ground_truth_steps(tree: BoolOp) -> Tuple[List[str], List[dict], bool]:
    """
    Generate the correct step-by-step CoT for a boolean expression tree.
    Returns (steps, node_info_list, final_answer).
    """
    counter = [0]
    steps: List[str] = []
    node_infos: List[dict] = []

    def _traverse(node):
        """Returns (node_id_or_None, evaluated_value)."""
        if isinstance(node, BoolLeaf):
            return None, node.value

        left_id, left_val = _traverse(node.left)
        right_id, right_val = _traverse(node.right)

        counter[0] += 1
        nid = f"[{counter[0]:02d}]"

        left_ref = left_id if left_id is not None else str(left_val)
        right_ref = right_id if right_id is not None else str(right_val)
        logic_ref = f"{left_ref} {node.op} {right_ref}"

        result = evaluate(node)

        if left_id is not None or right_id is not None:
            logic_sub = f"{left_val} {node.op} {right_val}"
            logic_line = f"'{logic_ref}' \u2192 '{logic_sub}'"
        else:
            logic_line = f"'{logic_ref}'"

        step_text = f"**Node {nid}**\n* Logic: {logic_line}\n* Result: '{result}'"
        steps.append(step_text)
        node_infos.append({"id": nid, "result": result})

        return nid, result

    _, final_result = _traverse(tree)

    summary_lines = ["### Summary"]
    for info in node_infos:
        summary_lines.append(f"* {info['id']}: {info['result']}")
    summary_lines.append(f"**Final Answer: {final_result}**")
    steps.append("\n".join(summary_lines))

    return steps, node_infos, final_result


def build_problem_text(expression: str, node_ids: List[str]) -> str:
    ids_str = ", ".join(node_ids)
    return (
        f"### Problem Statement\n"
        f"1. **Expression**: '{expression}'\n"
        f"2. **Node IDs**: {ids_str}"
    )


# ──────────────────────────────────────────────────────────────────────
# Model-based CoT generation
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
    model, tokenizer, system_prompt: str, problem_text: str
) -> str:
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=1.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def parse_model_cot(response: str) -> Tuple[List[str], Optional[bool]]:
    """Parse model-generated CoT into steps and extract the final answer."""
    steps: List[str] = []

    node_pattern = re.compile(
        r"(\*\*Node \[\d+\]\*\*.*?)(?=\*\*Node \[\d+\]\*\*|###\s*Summary|$)",
        re.DOTALL,
    )
    for m in node_pattern.finditer(response):
        steps.append(m.group(1).strip())

    summary_match = re.search(r"(###\s*Summary.*?)$", response, re.DOTALL)
    if summary_match:
        steps.append(summary_match.group(1).strip())

    answer_match = re.search(
        r"\*\*Final Answer:\s*(True|False)\*\*", response, re.IGNORECASE
    )
    model_answer = None
    if answer_match:
        model_answer = answer_match.group(1).strip().lower() == "true"

    return steps, model_answer


# ──────────────────────────────────────────────────────────────────────
# Dataset generation
# ──────────────────────────────────────────────────────────────────────

def generate_dataset(
    num_structures: int,
    instances_per: int,
    depths: List[int],
    mode: str = "ground_truth",
    model_id: Optional[str] = None,
    device: str = "cuda:0",
    system_prompt: str = "",
    seed: int = 42,
) -> Tuple[Dict, List[dict]]:
    """
    Generate the full boolean CoT dataset.

    Returns:
        geometry_data: dict keyed by logic_key, compatible with geometry pipeline
        fur_data: list of dicts, compatible with FUR pipeline
    """
    random.seed(seed)

    tokenizer = None
    hf_model = None
    if mode == "model":
        assert model_id is not None, "--hf_model required for model mode"
        print(f"[INFO] Loading model: {model_id}")
        tokenizer, hf_model = load_hf_model(model_id, device)

    geometry_data: Dict[str, list] = {}
    fur_data: List[dict] = []

    structures_per_depth = max(1, num_structures // len(depths))

    for d in depths:
        for s_idx in range(structures_per_depth):
            topo = random_topology(d)
            logic_key = topo.logic_key()
            n_leaves = topo.num_leaves()

            base_key = logic_key
            suffix = 0
            while logic_key in geometry_data:
                suffix += 1
                logic_key = f"{base_key}_v{suffix}"

            print(
                f"[INFO] Structure: {logic_key} "
                f"(depth={d}, nodes={topo.num_nodes()}, leaves={n_leaves})"
            )
            geometry_data[logic_key] = []

            max_combos = 2 ** n_leaves
            if instances_per >= max_combos:
                all_combos = list(itertools.product([True, False], repeat=n_leaves))
                random.shuffle(all_combos)
            else:
                combo_set: set = set()
                while len(combo_set) < instances_per:
                    combo = tuple(random.choice([True, False]) for _ in range(n_leaves))
                    combo_set.add(combo)
                all_combos = list(combo_set)

            for i, leaves in enumerate(all_combos[:instances_per]):
                tree = topo.instantiate(list(leaves))
                ground_truth = evaluate(tree)
                expression, node_ids = format_expression(tree)
                problem_text = build_problem_text(expression, node_ids)

                if mode == "ground_truth":
                    steps, _, final_result = generate_ground_truth_steps(tree)
                    model_answer = final_result
                else:
                    response = prompt_model_for_cot(
                        hf_model, tokenizer, system_prompt, problem_text
                    )
                    parsed_steps, model_answer = parse_model_cot(response)
                    if parsed_steps:
                        steps = parsed_steps
                    else:
                        print(f"  [WARN] Parsing failed for {logic_key} instance {i}, using ground truth")
                        steps, _, final_result = generate_ground_truth_steps(tree)
                        model_answer = final_result

                full_cot = "\n".join(steps)
                topic = f"instance_{i + 1}"

                geo_record = {
                    "topic": topic,
                    "lang": "en",
                    "steps": steps,
                    "expression": expression,
                    "ground_truth": ground_truth,
                    "model_answer": model_answer,
                    "faithful": (
                        model_answer == ground_truth
                        if model_answer is not None
                        else None
                    ),
                    "leaf_values": list(leaves),
                }
                geometry_data[logic_key].append(geo_record)

                correct_letter = "A" if ground_truth else "B"
                fur_record = {
                    "id": f"{logic_key}_{topic}",
                    "question": problem_text,
                    "expression": expression,
                    "correct_letter": correct_letter,
                    "cot_prompt": problem_text,
                    "cot": full_cot,
                    "options": ["A): True", "B): False"],
                    "segmented_cot": steps,
                    "ground_truth": ground_truth,
                    "model_answer": model_answer,
                    "node_ids": node_ids,
                    "logic_key": logic_key,
                    "leaf_values": list(leaves),
                    "raw_instance": {
                        "expression": expression,
                        "node_ids": node_ids,
                        "ground_truth": ground_truth,
                    },
                }
                fur_data.append(fur_record)

    return geometry_data, fur_data


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate boolean CoT datasets for Faithfulness-as-Geometry"
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="ground_truth",
        choices=["ground_truth", "model"],
    )
    ap.add_argument("--hf_model", type=str, default=None)
    ap.add_argument("--num_structures", type=int, default=5)
    ap.add_argument("--instances_per", type=int, default=10)
    ap.add_argument("--depths", type=str, default="2,3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--system_prompt_file", type=str, default="boolean_task")
    ap.add_argument("--output_dir", type=str, default="data")
    args = ap.parse_args()

    depths = [int(d.strip()) for d in args.depths.split(",")]

    with open(args.system_prompt_file, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    os.makedirs(args.output_dir, exist_ok=True)

    geometry_data, fur_data = generate_dataset(
        num_structures=args.num_structures,
        instances_per=args.instances_per,
        depths=depths,
        mode=args.mode,
        model_id=args.hf_model,
        device=args.device,
        system_prompt=system_prompt,
        seed=args.seed,
    )

    geo_path = os.path.join(args.output_dir, "boolean_cots.json")
    with open(geo_path, "w", encoding="utf-8") as f:
        json.dump(geometry_data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Geometry data saved to {geo_path}")

    fur_path = os.path.join(args.output_dir, "boolean_cots_fur.jsonl")
    with open(fur_path, "w", encoding="utf-8") as f:
        for record in fur_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[INFO] FUR data saved to {fur_path}")

    total = sum(len(v) for v in geometry_data.values())
    print(f"\n[SUMMARY]")
    print(f"  Structures: {len(geometry_data)}")
    print(f"  Total instances: {total}")
    for key, records in geometry_data.items():
        n_faithful = sum(1 for r in records if r.get("faithful") is True)
        print(f"    {key}: {len(records)} instances, {n_faithful} faithful")


if __name__ == "__main__":
    main()
