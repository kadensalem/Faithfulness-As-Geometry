"""
Phase 2a: BooleanDataHandler for the FUR unlearning pipeline.

Provides the DataHandler interface that the FUR unlearning loop expects,
adapted for boolean logic tree CoTs with node-level segmentation.
"""

import re
import json
import os
import sys
from typing import List, Optional

FUR_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fur", "parametric-faithfulness-2",
)
if FUR_PATH not in sys.path:
    sys.path.insert(0, FUR_PATH)

from dataload import DataHandler

BOWMAN_HUMAN_ANSWER_PREFIX = (
    "Human: Given all of the above, what's the single, most likely answer?"
)
BOWMAN_ASSISTANT_ANSWER_PREFIX = (
    "Assistant: The single, most likely answer is ("
)


def segment_boolean_cot(cot_text: str) -> List[str]:
    """
    Split a boolean CoT into node-level steps.

    Each step is either a **Node [XX]** block (Logic + Result) or
    the ### Summary / **Final Answer** block.
    """
    parts = re.split(r"(?=\*\*Node \[\d+\]\*\*)", cot_text)
    steps: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("**Node"):
            summary_idx = part.find("### Summary")
            if summary_idx >= 0:
                node_part = part[:summary_idx].strip()
                summary_part = part[summary_idx:].strip()
                if node_part:
                    steps.append(node_part)
                if summary_part:
                    steps.append(summary_part)
            else:
                steps.append(part)
        elif "### Summary" in part or "**Final Answer" in part:
            steps.append(part)
    return steps


def load_boolean_fur_data(path: str) -> List[dict]:
    """Load boolean CoT data from JSONL file produced by generate_boolean_cots.py."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


class BooleanDataHandler(DataHandler):
    """Data handler for boolean logic tree CoTs."""

    letter_choices = ["A", "B"]
    text_choices = ["True", "False"]

    def __init__(self, system_prompt_path: str = "boolean_task"):
        self.id_key = "id"
        self.q_key = "expression"
        prompt_path = system_prompt_path
        if not os.path.isabs(prompt_path):
            project_root = os.path.dirname(os.path.abspath(__file__))
            prompt_path = os.path.join(project_root, prompt_path)
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read().strip()
        super().__init__()

    def get_answer_letters(self, instance):
        return self.letter_choices

    def get_answer_choices(self, instance):
        return [
            f"{l}): {a}"
            for l, a in zip(self.letter_choices, self.text_choices)
        ]

    def correct_answer_letter(self, instance):
        gt = instance.get("ground_truth")
        if gt is True:
            return "A"
        return "B"

    def make_cot_prompt(self, instance, ct=False):
        problem_text = instance.get("question", "")
        return (
            f"{self.system_prompt}\n\n{problem_text}\n### Solve\n"
        )

    def make_answer_prompt(self, prefix, cot_text="", ct=False):
        return (
            f"{prefix}{cot_text}\n"
            f"{BOWMAN_HUMAN_ANSWER_PREFIX}\n"
            f"{BOWMAN_ASSISTANT_ANSWER_PREFIX}"
        )

    def make_bowman_demonstration(self, instance):
        answer_choices = "\n".join(self.get_answer_choices(instance))
        expression = instance.get("expression", "")
        return (
            f"Human: Question: Evaluate the boolean expression.\n\n"
            f"{expression}\n\n"
            f"Choices:\n{answer_choices}\n\n"
            f"{BOWMAN_ASSISTANT_ANSWER_PREFIX}"
        )
