#!/usr/bin/env python3
"""
Phase 2: Anchor-token trajectory embedding extraction.

For each of the N pilot MCQ questions, this script:
  1. Loads the greedy CoT text from data_tune_30/mcq_cots_fur.jsonl
  2. Reconstructs the exact prompt+CoT sequence (with chat template)
  3. Runs ONE forward pass with output_hidden_states=True
  4. Identifies 9 structural anchor token positions via character-offset mapping
  5. Extracts hidden states at those positions for middle layer and last layer
  6. Extracts mean-pooled span embeddings for Premise and Reasoning spans
  7. Stores Conclusion token embeddings separately (decision probes)
  8. Computes displacement vectors from anchor embeddings
  9. Saves everything to a pickle file
 10. Runs a PCA sanity check on the 9-point anchor trajectory

Anchor token order (trajectory, 9 points):
  Answer_A → Reasoning_A → Answer_B → Reasoning_B →
  Answer_C → Reasoning_C → Answer_D → Reasoning_D → Final_Answer

Usage:
    python extract_anchor_embeddings.py \\
        --hf_model meta-llama/Meta-Llama-3-8B-Instruct \\
        --fur_file data_tune_30/mcq_cots_fur.jsonl \\
        --n_questions 10 \\
        --middle_layer 14 \\
        --output_file data/pilot_trajectories.pkl
"""

import argparse
import json
import os
import pickle
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────────
# Regex patterns for structural anchor detection
# ──────────────────────────────────────────────────────────────────────────────

# "Answer A:" — captures the letter; match ends at the colon
ANSWER_HEADER_RE = re.compile(r"\bAnswer\s+([A-D])\s*:", re.IGNORECASE)

# "* Reasoning:" or "Reasoning:" — match ends at the colon
REASONING_RE = re.compile(r"\*?\s*Reasoning\s*:", re.IGNORECASE)

# "* Conclusion: S" or "Conclusion: R" — group 1 is the S/R token
CONCLUSION_RE = re.compile(r"\*?\s*Conclusion\s*:\s*([SR])\b", re.IGNORECASE)

# "**Final Answer**" ... "Answer:" — match ends at the final colon
FINAL_ANSWER_RE = re.compile(
    r"\*\*Final Answer\*\*.*?Answer\s*:", re.IGNORECASE | re.DOTALL
)


# ──────────────────────────────────────────────────────────────────────────────
# Character-to-token mapping
# ──────────────────────────────────────────────────────────────────────────────

def char_pos_to_token_idx(
    offsets: List[Tuple[int, int]], char_pos: int
) -> Optional[int]:
    """
    Return the index of the token whose character range contains char_pos.
    offsets[i] = (start_char, end_char) for token i.
    Special tokens (with (0,0) offsets) are skipped.
    Returns None if no token covers the position.
    """
    # Forward scan — take the last token that starts at or before char_pos
    best = None
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:
            continue  # special token, no character span
        if s <= char_pos < e:
            return i  # exact hit
        if s <= char_pos:
            best = i  # keep as fallback (last token starting before char_pos)
    return best


def find_anchor_token_positions(
    full_text: str,
    offsets: List[Tuple[int, int]],
    cot_start_char: int,
) -> Dict[str, Optional[int]]:
    """
    Find token indices for all 9 trajectory anchor points plus Conclusion probes.

    Searches only in full_text[cot_start_char:] to avoid matching
    patterns that appear in the system prompt / example format.

    Returns a dict with keys:
      Answer_A, Reasoning_A, Answer_B, Reasoning_B,
      Answer_C, Reasoning_C, Answer_D, Reasoning_D,
      Final_Answer,
      Conclusion_A, Conclusion_B, Conclusion_C, Conclusion_D
    """
    positions: Dict[str, Optional[int]] = {}

    # ── Collect all Answer X: header positions (char index of the colon) ──
    answer_char_positions: Dict[str, int] = {}  # letter → char index of colon
    answer_start_chars: List[Tuple[str, int]] = []  # (letter, start of match)

    for m in ANSWER_HEADER_RE.finditer(full_text, cot_start_char):
        letter = m.group(1).upper()
        if letter not in answer_char_positions:  # take first occurrence per letter
            colon_char = m.end() - 1  # colon is the last char of the match
            answer_char_positions[letter] = colon_char
            answer_start_chars.append((letter, m.start()))
            positions[f"Answer_{letter}"] = char_pos_to_token_idx(offsets, colon_char)

    # Sort by position in text
    answer_start_chars.sort(key=lambda x: x[1])

    # ── For each answer block, find Reasoning and Conclusion anchors ──
    # A block spans from its Answer X: header to the next Answer X: header
    # (or end of CoT for the last block).

    def block_char_range(letter: str) -> Tuple[int, int]:
        idx = next((i for i, (l, _) in enumerate(answer_start_chars) if l == letter), None)
        if idx is None:
            return (cot_start_char, len(full_text))
        start = answer_start_chars[idx][1]
        end = (
            answer_start_chars[idx + 1][1]
            if idx + 1 < len(answer_start_chars)
            else len(full_text)
        )
        return start, end

    for letter in ["A", "B", "C", "D"]:
        blk_start, blk_end = block_char_range(letter)

        # * Reasoning: colon
        m = REASONING_RE.search(full_text, blk_start, blk_end)
        if m:
            colon_char = m.end() - 1
            positions[f"Reasoning_{letter}"] = char_pos_to_token_idx(offsets, colon_char)
        else:
            positions[f"Reasoning_{letter}"] = None

        # * Conclusion: S or R (S/R token itself, not the colon)
        m = CONCLUSION_RE.search(full_text, blk_start, blk_end)
        if m:
            sr_char = m.start(1)  # character position of S or R
            positions[f"Conclusion_{letter}"] = char_pos_to_token_idx(offsets, sr_char)
        else:
            positions[f"Conclusion_{letter}"] = None

    # ── Final Answer: colon ──
    m = FINAL_ANSWER_RE.search(full_text, cot_start_char)
    if m:
        colon_char = m.end() - 1
        positions["Final_Answer"] = char_pos_to_token_idx(offsets, colon_char)
    else:
        positions["Final_Answer"] = None

    return positions


# ──────────────────────────────────────────────────────────────────────────────
# Span mean-pooling
# ──────────────────────────────────────────────────────────────────────────────

def span_mean(
    layer_hs: torch.Tensor,  # [1, seq_len, hidden_dim]
    start_tok: int,
    end_tok: int,
) -> np.ndarray:
    """Mean-pool hidden states over token range [start_tok, end_tok)."""
    if start_tok >= end_tok or start_tok < 0 or end_tok > layer_hs.shape[1]:
        # Fall back to single token at start_tok if span is degenerate
        pos = max(0, min(start_tok, layer_hs.shape[1] - 1))
        return layer_hs[0, pos, :].detach().float().cpu().numpy()
    return layer_hs[0, start_tok:end_tok, :].mean(dim=0).detach().float().cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Block text parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_cot_blocks(cot_text: str) -> Dict[str, Dict[str, str]]:
    """Parse CoT into per-block premise/reasoning/conclusion strings."""
    blocks = {}
    for letter in ["A", "B", "C", "D"]:
        next_letters = [l for l in ["A", "B", "C", "D"] if l > letter]
        next_pat = "|".join(rf"Answer\s+{l}\s*:" for l in next_letters)
        block_end_pat = rf"(?={next_pat}|\*\*Final Answer\*\*|$)" if next_letters else r"(?=\*\*Final Answer\*\*|$)"
        blk_re = re.compile(
            rf"Answer\s+{letter}\s*:.*?{block_end_pat}",
            re.IGNORECASE | re.DOTALL,
        )
        m = blk_re.search(cot_text)
        if not m:
            blocks[letter] = {"premise": "", "reasoning": "", "conclusion": ""}
            continue
        blk = m.group(0)
        prem_m = re.search(
            r"\*?\s*Premise\s*:\s*(.+?)(?=\*?\s*Reasoning\s*:|$)", blk, re.IGNORECASE | re.DOTALL
        )
        reas_m = re.search(
            r"\*?\s*Reasoning\s*:\s*(.+?)(?=\*?\s*Conclusion\s*:|$)", blk, re.IGNORECASE | re.DOTALL
        )
        conc_m = re.search(r"\*?\s*Conclusion\s*:\s*([SR])\b", blk, re.IGNORECASE)
        blocks[letter] = {
            "premise": prem_m.group(1).strip() if prem_m else "",
            "reasoning": reas_m.group(1).strip() if reas_m else "",
            "conclusion": conc_m.group(1).upper() if conc_m else "",
        }
    return blocks


# ──────────────────────────────────────────────────────────────────────────────
# Per-question processing
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def process_question(
    fur_record: dict,
    model,
    tokenizer,
    middle_layer: int,
    device: str,
) -> dict:
    """
    Run a single forward pass and extract all anchor/span embeddings.

    Returns the full data structure for this question.
    """
    question_id = fur_record.get("id", "unknown")
    cot_prompt = fur_record.get("cot_prompt", "")
    cot_text = fur_record.get("cot", "")

    print(f"\n[Q] {question_id}", flush=True)

    # ── Reconstruct the exact context the model saw ──
    # Apply the same chat template that was used during generation.
    formatted_prompt = cot_prompt
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": cot_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass  # fall back to raw prompt

    full_text = formatted_prompt + cot_text
    cot_start_char = len(formatted_prompt)

    # ── Tokenize with character-offset mapping ──
    encoding = tokenizer(
        full_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding.get("attention_mask", torch.ones_like(input_ids)).to(device)
    offsets: List[Tuple[int, int]] = encoding["offset_mapping"][0].tolist()
    seq_len = input_ids.shape[1]

    # offset_mapping must be removed from kwargs before passing to model
    # (it's a tokenizer output, not a model input)

    print(f"  seq_len={seq_len}  cot_start_char={cot_start_char}", flush=True)

    # ── Single forward pass ──
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )

    # hidden_states: tuple of (n_layers+1) tensors, each [1, seq_len, hidden_dim]
    # Index 0 = embedding layer; index k = output of transformer block k.
    hidden_states = outputs.hidden_states
    n_layers = len(hidden_states) - 1  # number of transformer blocks
    last_layer = n_layers              # index of last transformer block output

    print(f"  n_transformer_layers={n_layers}  middle_layer={middle_layer}  last_layer={last_layer}", flush=True)

    # ── Find anchor token positions ──
    anchor_pos = find_anchor_token_positions(full_text, offsets, cot_start_char)
    print(f"  anchor_positions={anchor_pos}", flush=True)

    # Validate: warn if any trajectory anchor is missing
    traj_keys = [
        "Answer_A", "Reasoning_A", "Answer_B", "Reasoning_B",
        "Answer_C", "Reasoning_C", "Answer_D", "Reasoning_D", "Final_Answer",
    ]
    missing = [k for k in traj_keys if anchor_pos.get(k) is None]
    if missing:
        print(f"  WARNING: missing anchor positions for {missing}", flush=True)

    def get_hidden(layer_idx: int, tok_pos: Optional[int]) -> Optional[np.ndarray]:
        if tok_pos is None or tok_pos < 0 or tok_pos >= seq_len:
            return None
        return hidden_states[layer_idx][0, tok_pos, :].detach().float().cpu().numpy()

    # ── Anchor embeddings (middle layer) ──
    anchor_embeddings: Dict[str, Optional[np.ndarray]] = {}
    for key in traj_keys:
        anchor_embeddings[key] = get_hidden(middle_layer, anchor_pos.get(key))

    # ── Conclusion embeddings (decision probes, middle layer) ──
    conclusion_embeddings: Dict[str, Optional[np.ndarray]] = {}
    for letter in ["A", "B", "C", "D"]:
        conclusion_embeddings[letter] = get_hidden(
            middle_layer, anchor_pos.get(f"Conclusion_{letter}")
        )

    # ── Span embeddings ──
    # Premise span:  tokens in (Answer_X colon, * Reasoning: colon)  exclusive
    # Reasoning span: tokens in (* Reasoning: colon, Conclusion token)  exclusive

    expression_mid: Dict[str, Optional[np.ndarray]] = {}
    expression_last: Dict[str, Optional[np.ndarray]] = {}

    for letter in ["A", "B", "C", "D"]:
        ans_pos = anchor_pos.get(f"Answer_{letter}")
        reas_pos = anchor_pos.get(f"Reasoning_{letter}")
        conc_pos = anchor_pos.get(f"Conclusion_{letter}")

        # Premise span: [ans_pos+1, reas_pos)
        if ans_pos is not None and reas_pos is not None and reas_pos > ans_pos + 1:
            expression_mid[f"premise_{letter}"] = span_mean(
                hidden_states[middle_layer], ans_pos + 1, reas_pos
            )
            expression_last[f"premise_{letter}"] = span_mean(
                hidden_states[last_layer], ans_pos + 1, reas_pos
            )
        else:
            expression_mid[f"premise_{letter}"] = None
            expression_last[f"premise_{letter}"] = None

        # Reasoning span: [reas_pos+1, conc_pos)
        if reas_pos is not None and conc_pos is not None and conc_pos > reas_pos + 1:
            expression_mid[f"reasoning_{letter}"] = span_mean(
                hidden_states[middle_layer], reas_pos + 1, conc_pos
            )
            expression_last[f"reasoning_{letter}"] = span_mean(
                hidden_states[last_layer], reas_pos + 1, conc_pos
            )
        else:
            expression_mid[f"reasoning_{letter}"] = None
            expression_last[f"reasoning_{letter}"] = None

    # ── Displacement vectors (derived, no extra extraction) ──
    displacement_vectors: Dict[str, Optional[np.ndarray]] = {}
    letters = ["A", "B", "C", "D"]

    for i, letter in enumerate(letters):
        ans_e = anchor_embeddings.get(f"Answer_{letter}")
        reas_e = anchor_embeddings.get(f"Reasoning_{letter}")
        next_letter = letters[i + 1] if i + 1 < len(letters) else None

        # premise_delta[K] = Reasoning_K − Answer_K
        if ans_e is not None and reas_e is not None:
            displacement_vectors[f"premise_delta_{letter}"] = reas_e - ans_e
        else:
            displacement_vectors[f"premise_delta_{letter}"] = None

        # reasoning_delta[K] = Answer_{K+1} − Reasoning_K
        if next_letter is not None:
            next_ans_e = anchor_embeddings.get(f"Answer_{next_letter}")
            if reas_e is not None and next_ans_e is not None:
                displacement_vectors[f"reasoning_delta_{letter}"] = next_ans_e - reas_e
            else:
                displacement_vectors[f"reasoning_delta_{letter}"] = None
        else:
            displacement_vectors[f"reasoning_delta_{letter}"] = None

    # final_answer_delta = hidden_state(letter token) − anchor(Final Answer: colon)
    fa_pos = anchor_pos.get("Final_Answer")
    fa_e = anchor_embeddings.get("Final_Answer")
    if fa_pos is not None and fa_pos + 1 < seq_len and fa_e is not None:
        letter_tok_e = get_hidden(middle_layer, fa_pos + 1)
        displacement_vectors["final_answer_delta"] = (
            letter_tok_e - fa_e if letter_tok_e is not None else None
        )
    else:
        displacement_vectors["final_answer_delta"] = None

    # Free GPU memory for this forward pass
    del outputs, hidden_states
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Build result ──
    options = fur_record.get("options", [])
    if isinstance(options, list):
        answer_choices = {chr(ord("A") + i): opt for i, opt in enumerate(options)}
    else:
        answer_choices = options  # already a dict

    return {
        "question_id": question_id,
        "question_text": fur_record.get("question", ""),
        "answer_choices": answer_choices,
        "correct_answer": fur_record.get("correct_letter", ""),
        "cot_greedy": {
            "full_text": cot_text,
            "model_answer": fur_record.get("model_answer", ""),
            "steps": parse_cot_blocks(cot_text),
        },
        "anchor_embeddings": anchor_embeddings,
        "conclusion_embeddings": conclusion_embeddings,
        "expression_mid": expression_mid,
        "expression_last": expression_last,
        "displacement_vectors": displacement_vectors,
        # m_prime_labels: not computed in this script (requires re-sampling)
        "m_prime_labels": {},
        "meta": {
            "middle_layer": middle_layer,
            "last_layer": last_layer,
            "n_model_layers": n_layers,
            "seq_len": seq_len,
            "anchor_positions": anchor_pos,
            "cot_start_char": cot_start_char,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Sanity check: PCA of 9 anchor embeddings
# ──────────────────────────────────────────────────────────────────────────────

def pca_sanity_check(results: List[dict], save_dir: str) -> None:
    """
    For each question, project the 9 anchor embeddings to 2D PCA and save a CSV.
    Also print a summary: explained variance and mean L2 distance between
    consecutive anchor points (trajectory length proxy).
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("[WARN] sklearn not available, skipping PCA sanity check")
        return

    os.makedirs(save_dir, exist_ok=True)
    traj_keys = [
        "Answer_A", "Reasoning_A", "Answer_B", "Reasoning_B",
        "Answer_C", "Reasoning_C", "Answer_D", "Reasoning_D", "Final_Answer",
    ]

    print("\n[SANITY CHECK] PCA of 9-point anchor trajectories", flush=True)
    print(f"{'Q':30s}  {'PC1_var':>8s}  {'PC2_var':>8s}  {'traj_len':>10s}  {'n_missing':>10s}")
    print("-" * 72)

    for rec in results:
        qid = rec["question_id"]
        anc = rec["anchor_embeddings"]

        vecs = [anc.get(k) for k in traj_keys]
        n_missing = sum(1 for v in vecs if v is None)

        valid_vecs = [v for v in vecs if v is not None]
        if len(valid_vecs) < 3:
            print(f"  {qid[:30]:30s}  [too few anchors: {len(valid_vecs)}/9]")
            continue

        X = np.stack(valid_vecs).astype(np.float32)
        pca = PCA(n_components=min(3, len(valid_vecs)), random_state=42).fit(X)
        ev = pca.explained_variance_ratio_

        # Trajectory length: sum of consecutive L2 distances
        traj_len = float(np.sum(np.linalg.norm(np.diff(X, axis=0), axis=1)))

        # Save 2D projection CSV
        proj = pca.transform(X)
        import csv
        csv_path = os.path.join(save_dir, f"{qid.replace('/', '_')}_pca2d.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["anchor", "pc1", "pc2"])
            valid_keys = [k for k, v in zip(traj_keys, vecs) if v is not None]
            for key, row in zip(valid_keys, proj):
                writer.writerow([key, row[0], row[1]])

        pc1_v = ev[0] if len(ev) > 0 else 0.0
        pc2_v = ev[1] if len(ev) > 1 else 0.0
        print(
            f"  {qid[:30]:30s}  {pc1_v:8.3f}  {pc2_v:8.3f}  {traj_len:10.2f}  {n_missing:10d}",
            flush=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    ap = argparse.ArgumentParser(
        description="Extract anchor-token trajectory embeddings for pilot MCQ questions"
    )
    ap.add_argument(
        "--hf_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model ID",
    )
    ap.add_argument(
        "--fur_file", type=str, default="data_tune_30/mcq_cots_fur.jsonl",
        help="Path to JSONL with stored CoT prompts and texts",
    )
    ap.add_argument(
        "--n_questions", type=int, default=10,
        help="Number of questions to process",
    )
    ap.add_argument(
        "--middle_layer", type=int, default=14,
        help="Representative middle layer index (1-indexed transformer block)",
    )
    ap.add_argument(
        "--output_file", type=str, default="data/pilot_trajectories.pkl",
        help="Output pickle file path",
    )
    ap.add_argument(
        "--device", type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument(
        "--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"],
        help="Model weight dtype",
    )
    ap.add_argument(
        "--sanity_check_dir", type=str, default="data/pca_sanity",
        help="Directory to save PCA sanity-check CSVs",
    )
    args = ap.parse_args()

    # ── Load FUR records ──
    fur_path = args.fur_file
    if not os.path.isabs(fur_path):
        fur_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fur_path)

    records: List[dict] = []
    with open(fur_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Filter to questions with all 4 answer blocks present in the CoT
    usable = []
    for rec in records:
        cot = rec.get("cot", "")
        if all(re.search(rf"\bAnswer\s+{l}\s*:", cot, re.IGNORECASE) for l in "ABCD"):
            usable.append(rec)
        else:
            print(f"[SKIP] {rec.get('id')} — CoT missing answer blocks", flush=True)

    print(f"[INFO] {len(usable)}/{len(records)} records have all 4 answer blocks", flush=True)
    selected = usable[: args.n_questions]
    print(f"[INFO] Processing {len(selected)} questions", flush=True)

    if not selected:
        raise SystemExit("No usable questions found. Check --fur_file path.")

    # ── Load model ──
    print(f"[INFO] Loading model: {args.hf_model}", flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Validate middle_layer against actual model depth
    n_layers = model.config.num_hidden_layers
    if args.middle_layer < 1 or args.middle_layer > n_layers:
        raise ValueError(
            f"--middle_layer={args.middle_layer} out of range for {n_layers}-layer model"
        )
    print(
        f"[INFO] Model has {n_layers} transformer layers. "
        f"middle_layer={args.middle_layer}, last_layer={n_layers}",
        flush=True,
    )

    # ── Process each question ──
    results: List[dict] = []
    for i, rec in enumerate(selected, start=1):
        print(f"\n[{i}/{len(selected)}] Processing: {rec.get('id', '?')}", flush=True)
        try:
            result = process_question(
                fur_record=rec,
                model=model,
                tokenizer=tokenizer,
                middle_layer=args.middle_layer,
                device=args.device,
            )
            results.append(result)
            # Print anchor coverage for this question
            anc = result["anchor_embeddings"]
            traj_keys = [
                "Answer_A", "Reasoning_A", "Answer_B", "Reasoning_B",
                "Answer_C", "Reasoning_C", "Answer_D", "Reasoning_D", "Final_Answer",
            ]
            found = sum(1 for k in traj_keys if anc.get(k) is not None)
            print(f"  anchor coverage: {found}/9", flush=True)
        except Exception as e:
            print(f"  [ERROR] Failed: {e}", flush=True)
            import traceback
            traceback.print_exc()

    # ── Save output ──
    out_path = args.output_file
    if not os.path.isabs(out_path):
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n[DONE] Saved {len(results)} records to {out_path}", flush=True)

    # ── Sanity check ──
    sanity_dir = args.sanity_check_dir
    if not os.path.isabs(sanity_dir):
        sanity_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), sanity_dir)
    pca_sanity_check(results, sanity_dir)
    print(f"[DONE] PCA CSVs saved to {sanity_dir}", flush=True)

    # ── Summary ──
    print("\n[SUMMARY]", flush=True)
    total_anc = 0
    for rec in results:
        traj_keys = [
            "Answer_A", "Reasoning_A", "Answer_B", "Reasoning_B",
            "Answer_C", "Reasoning_C", "Answer_D", "Reasoning_D", "Final_Answer",
        ]
        total_anc += sum(1 for k in traj_keys if rec["anchor_embeddings"].get(k) is not None)
    print(f"  Questions processed: {len(results)}")
    print(f"  Total anchor embeddings extracted: {total_anc} / {len(results) * 9}")
    conc_ok = sum(
        1 for rec in results
        for l in "ABCD"
        if rec["conclusion_embeddings"].get(l) is not None
    )
    print(f"  Conclusion probe embeddings: {conc_ok} / {len(results) * 4}")
    span_ok = sum(
        1 for rec in results
        for key in ["premise_A", "reasoning_A", "premise_B", "reasoning_B",
                    "premise_C", "reasoning_C", "premise_D", "reasoning_D"]
        if rec["expression_last"].get(key) is not None
    )
    print(f"  expression_last span embeddings: {span_ok} / {len(results) * 8}")


if __name__ == "__main__":
    main()
