"""Adapter to run FUR (NPO+KL unlearning) on pre-generated MCQ CoTs.

Bypasses furV2.py's load_or_generate_dataset_cots() by reading directly
from mcq_cots_fur.jsonl. Selects target questions by ID, computes missing
nocot_probs / cot_probs with a single upfront model load, then calls
unlearn_single() for each (question, step_idx) pair using the stored
5-block segmented_cot.

The output JSONL has the same per-line schema as furV2.py's .out files so
that compute_fur_labels.py can process both formats identically.

Usage (must be submitted via SLURM — GPU required):
    python run_fur_pilot.py \\
        --model_name meta-llama/Meta-Llama-3-8B-Instruct \\
        --fur_file data_tune_30/mcq_cots_fur.jsonl \\
        --question_ids openbook_1955 openbook_508 openbook_9-491 \\
        --output_file data/pilot_fur_results.jsonl \\
        --epochs 5 \\
        --lr 1e-5 \\
        --ff2 --pos
"""

import os
import sys
import gc
import json
import random
import argparse
import copy
import types

import numpy as np
import torch
from transformers import AutoTokenizer as TOK
from transformers import AutoModelForCausalLM as CLM
from tqdm import tqdm

# Make fur/parametric-faithfulness-2 importable without modifying it
FUR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fur", "parametric-faithfulness-2")
sys.path.insert(0, FUR_DIR)

from evaluate import answer_probabilities, generation_fixed_cot  # noqa: E402
from data import cot_to_otfd  # noqa: E402
from dataload import DATASETS  # noqa: E402
from furV2 import unlearn_single  # noqa: E402
from util import set_random_seed  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def store(record, fout):
    with open(fout, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_completed_ids(fout, stepwise=True):
    """Return set of already-processed check IDs from an existing output file."""
    ids = set()
    if os.path.exists(fout):
        with open(fout) as f:
            for line in f:
                d = json.loads(line)
                key = d["question"]
                if stepwise:
                    mode = d.get("unlearn_mode", "whole_block")
                    key = f"{key}_{d['step_idx']}_{mode}"
                ids.add(key)
    return ids


def adapt_record(rec, model, tokenizer, DH):
    """Compute nocot_probs / cot_probs and return a furV2-compatible target dict.

    Our jsonl has cot as a bare string; furV2 expects cot as a list.
    nocot_probs and cot_probs are missing and must be computed here.
    """
    raw = rec["raw_instance"]
    cot_text = rec["cot"]

    _, nocot_probs, _ = answer_probabilities(model, tokenizer, DH, raw)
    cot_probs, _ = generation_fixed_cot(model, tokenizer, DH, raw, cot_text)

    return {
        "id": rec["id"],
        # furV2 uses target['question'] as the dedup key; matches question_stem
        "question": rec["question"],
        "correct_letter": rec["correct_letter"],
        "cot_prompt": rec["cot_prompt"],
        "cot": [cot_text],                    # must be a list
        "options": rec["options"],
        "nocot_probs": nocot_probs.tolist(),
        "cot_probs": [cot_probs.tolist()],    # list of lists
        "segmented_cot": rec["segmented_cot"],
        "raw_instance": rec["raw_instance"],
    }


def build_fur_args(args):
    """Build the namespace that unlearn_single() / cot_to_otfd() inspect."""
    return types.SimpleNamespace(
        strategy="sentencize",   # uses target['segmented_cot'] directly
        stepwise=True,           # unlearn one step at a time
        method=args.method,
        epochs=args.epochs,
        lr=args.lr,
        pos=args.pos,
        ff2=args.ff2,
        kl_coeff=args.kl_coeff,
        npo_coeff=args.npo_coeff,
        beta=args.beta,
        mmlu=0,
        gsm=0,
        num_p=None,
    )


# ---------------------------------------------------------------------------
# Sub-block helpers
# ---------------------------------------------------------------------------

def is_valid_unlearning_target(step_text, tokenizer):
    """Return False for steps that should be skipped as unlearning targets.

    Filters out:
      1. Final Answer declarations ("**Final Answer**" or "Final Answer" prefix)
      2. Header-only blocks with no bullet lines (no '*' lines)
      3. Steps tokenizing to fewer than 10 tokens
    """
    text = step_text.strip()
    if "**Final Answer**" in text or text.startswith("Final Answer"):
        return False
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not any(l.startswith('*') for l in lines):
        return False
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) < 10:
        return False
    return True


def extract_subblock_targets(block_text):
    """Parse an Answer block into its sub-components.

    Expected format::

        Answer X: For each answer:
        * Premise: ...
        * Reasoning: ...
        * Conclusion: S

    Returns a dict with keys {header, premise, reasoning, conclusion}, or
    None if the block does not match the expected structure.
    """
    lines = block_text.strip().split('\n')
    header_lines = []
    premise = None
    reasoning = None
    conclusion = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('* Premise:'):
            premise = line
        elif stripped.startswith('* Reasoning:'):
            reasoning = line
        elif stripped.startswith('* Conclusion:'):
            conclusion = line
        else:
            header_lines.append(line)

    if premise is None or reasoning is None:
        return None

    return {
        'header': '\n'.join(header_lines),
        'premise': premise,
        'reasoning': reasoning,
        'conclusion': conclusion,
    }


def build_subblock_run(target, step_idx, unlearn_mode, cots_train):
    """Return (run_target, run_step_idx, run_cots_train, target_type,
               unlearn_target_text, unlearn_target_prefix) for a sub-block mode.

    Splits the Answer block at step_idx into a prefix segment and a target
    segment, inserts them into a deep-copied target, and patches cots_train
    so that cot_to_otfd's list.remove(target) uses the modified copy.

    Returns None if the block cannot be parsed.
    """
    step_text = target['segmented_cot'][step_idx]
    sub = extract_subblock_targets(step_text)
    if sub is None:
        return None

    if unlearn_mode == "premise_only":
        prefix_seg = sub['header']
        target_seg = sub['premise']
        target_type = "premise"
    elif unlearn_mode == "reasoning_only":
        prefix_seg = sub['header'] + "\n" + sub['premise']
        target_seg = sub['reasoning']
        target_type = "reasoning"
    else:  # premise_and_reasoning
        prefix_seg = sub['header']
        target_seg = sub['premise'] + "\n" + sub['reasoning']
        target_type = "premise_and_reasoning"

    target_copy = copy.deepcopy(target)
    target_copy['segmented_cot'] = (
        target['segmented_cot'][:step_idx]
        + [prefix_seg, target_seg]
        + target['segmented_cot'][step_idx + 1:]
    )

    preceding = "\n".join(target['segmented_cot'][:step_idx])
    unlearn_target_prefix = (preceding + "\n" + prefix_seg).lstrip("\n")

    # Replace original target in cots_train using identity check so that
    # cot_to_otfd's all.remove(target_copy) finds the right record.
    run_cots_train = [target_copy if r is target else r for r in cots_train]

    return (
        target_copy,
        step_idx + 1,
        run_cots_train,
        target_type,
        target_seg,
        unlearn_target_prefix,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--fur_file", default="data_tune_30/mcq_cots_fur.jsonl")
    parser.add_argument("--question_ids", nargs="+",
                        default=["openbook_7-156", "openbook_278", "openbook_7-969"],
                        help="IDs from the 'id' field of the jsonl to unlearn")
    parser.add_argument("--output_file", default="data/pilot_fur_results.jsonl")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--method", default="npo_KL")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pos", action="store_true",
                        help="Filter function tokens during unlearning")
    parser.add_argument("--ff2", action="store_true",
                        help="Optimize only MLP down-projection layers")
    parser.add_argument("--step_ids", nargs="*", type=int, default=None,
                        help="Step indices to run (default: all). E.g. --step_ids 0 2")
    parser.add_argument("--kl_coeff", type=float, default=1.0,
                        help="KL retain loss coefficient in NPO+KL (default: 1.0)")
    parser.add_argument("--npo_coeff", type=float, default=1.0,
                        help="NPO forget loss coefficient (default: 1.0)")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="NPO beta temperature — controls sharpness of forget gradient (default: 0.1)")
    parser.add_argument("--unlearn_mode", default="whole_block",
                        choices=["whole_block", "premise_only", "reasoning_only",
                                 "premise_and_reasoning"],
                        help="Granularity of the unlearning target within each Answer block "
                             "(default: whole_block — existing behaviour)")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_random_seed(args.seed)

    tokenizer = TOK.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    from mcq_dataload import MCQDataHandler
    DH = MCQDataHandler() 
    fur_args = build_fur_args(args)

    # ------------------------------------------------------------------
    # Step 1: load all records and compute nocot_probs / cot_probs
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"Loading model for prob computation: {args.model_name}")
    raw_records = load_jsonl(args.fur_file)
    print(f"  {len(raw_records)} records in {args.fur_file}")

    model = CLM.from_pretrained(args.model_name,
                                torch_dtype=torch.bfloat16,
                                trust_remote_code=True,
                                device_map="auto")

    adapted = []
    for rec in tqdm(raw_records, desc="Computing nocot/cot probs"):
        adapted.append(adapt_record(rec, model, tokenizer, DH))

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Model unloaded after prob computation.\n")

    # ------------------------------------------------------------------
    # Step 2: set up cots_train / cots_verify
    # All 30 records go into cots_train so cot_to_otfd can remove the
    # target and sample retain examples.  Last 20 serve as verify split.
    # ------------------------------------------------------------------
    random.seed(args.seed)
    shuffled = list(adapted)
    random.shuffle(shuffled)

    N_verify = min(20, len(shuffled) - len(args.question_ids) - 1)
    cots_train = shuffled
    cots_verify = shuffled[-N_verify:]
    print(f"cots_train: {len(cots_train)}  cots_verify: {len(cots_verify)}")

    # ------------------------------------------------------------------
    # Step 3: look up pilot target records
    # ------------------------------------------------------------------
    id_to_rec = {r["id"]: r for r in adapted}
    targets = []
    for qid in args.question_ids:
        if qid in id_to_rec:
            targets.append(id_to_rec[qid])
            print(f"  Target found: {qid}  correct={id_to_rec[qid]['correct_letter']}")
        else:
            print(f"  WARNING: question id {qid!r} not found in {args.fur_file}")

    if not targets:
        raise ValueError("No matching question IDs found — aborting.")

    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    completed = load_completed_ids(args.output_file, stepwise=True)
    print(f"\nResuming: {len(completed)} (question, step) pair(s) already done.")

    # ------------------------------------------------------------------
    # Step 4: unlearn each (target, step_idx)
    # ------------------------------------------------------------------
    print(f"\nUnlearning mode: {args.unlearn_mode}")

    for t_idx, target in enumerate(targets):
        n_steps = len(target["segmented_cot"])
        print(f"\n{'='*60}")
        print(f"Question {t_idx+1}/{len(targets)}: {target['id']}  ({n_steps} steps)")

        for step_idx in range(n_steps):
            if args.step_ids is not None and step_idx not in args.step_ids:
                print(f"  [skip-filter] step {step_idx} not in --step_ids")
                continue

            check_id = f"{target['question']}_{step_idx}_{args.unlearn_mode}"

            if check_id in completed:
                print(f"  [skip] step {step_idx} mode={args.unlearn_mode} already done")
                continue

            step_text = target['segmented_cot'][step_idx]

            if not is_valid_unlearning_target(step_text, tokenizer):
                print(f"  [skip-invalid] step {step_idx}: not a valid unlearning target")
                continue

            print(f"\n  Step {step_idx}/{n_steps-1} mode={args.unlearn_mode}: "
                  f"{step_text[:80]!r}")

            # ── resolve actual target / step for unlearn_single ────────────
            if args.unlearn_mode == "whole_block":
                run_target = target
                run_step_idx = step_idx
                run_cots_train = cots_train
                target_type = "whole_block"
                unlearn_target_text = step_text
                unlearn_target_prefix = "\n".join(target['segmented_cot'][:step_idx])
            else:
                sub_result = build_subblock_run(
                    target, step_idx, args.unlearn_mode, cots_train)
                if sub_result is None:
                    print(f"  [skip-parse] step {step_idx}: could not parse sub-blocks")
                    continue
                (run_target, run_step_idx, run_cots_train,
                 target_type, unlearn_target_text, unlearn_target_prefix) = sub_result

            result = unlearn_single(
                args.model_name, tokenizer, fur_args,
                run_target, run_step_idx,
                run_cots_train, cots_verify,
                DH, t_idx,
            )

            if result["unlearning_results"] is None:
                print(f"  [skip] too few unlearning targets at step {step_idx}")
                continue

            record = {
                "id": target["id"],
                "question": target["question"],
                "step_idx": step_idx,
                "options": target["options"],
                "correct": target["correct_letter"],
                "initial_cot": target["cot"],
                "initial_cot_probs": target["cot_probs"],
                "initial_probs": target["nocot_probs"],
                "prediction": int(np.argmax(target["nocot_probs"])),
                "cot_prediction": [int(np.argmax(p)) for p in target["cot_probs"]],
                "cot_step": target["segmented_cot"][step_idx],
                "segmented_cot": target["segmented_cot"],
                "unlearning_results": result["unlearning_results"],
                "unlearn_mode": args.unlearn_mode,
                "unlearn_target_type": target_type,
                "unlearn_target_text": unlearn_target_text,
                "unlearn_target_prefix": unlearn_target_prefix,
            }
            store(record, args.output_file)
            completed.add(check_id)
            print(f"  [saved] {check_id}")

    print(f"\nAll done.  Results written to: {args.output_file}")


if __name__ == "__main__":
    main()
