"""
MCQ DataHandler for the FUR unlearning pipeline.

Provides the DataHandler interface that the FUR unlearning loop expects,
adapted for OpenBookQA with structured Premise/Reasoning/Conclusion CoTs.
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

from dataload import DataHandler, OpenQA, BOWMAN_HUMAN_ANSWER_PREFIX, BOWMAN_ASSISTANT_ANSWER_PREFIX


def segment_mcq_cot(cot_text: str) -> List[str]:
    """
    Split a structured MCQ CoT into per-answer-choice segments.

    Each segment is either an ``Answer X: ... Premise ... Reasoning ...
    Conclusion`` block or the trailing ``**Final Answer** ...`` block.
    """
    text = cot_text or ""
    parts = re.split(r"(?=Answer\s+[A-Za-z]\s*:)", text)
    steps: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if re.match(r"Answer\s+[A-Za-z]\s*:", part):
            final_idx = part.find("**Final Answer**")
            if final_idx >= 0:
                answer_part = part[:final_idx].strip()
                final_part = part[final_idx:].strip()
                if answer_part:
                    steps.append(answer_part)
                if final_part:
                    steps.append(final_part)
            else:
                steps.append(part)
        elif "**Final Answer**" in part or re.search(r"Answer\s*:", part):
            steps.append(part)
    return steps


def load_mcq_fur_data(path: str) -> List[dict]:
    """Load MCQ CoT data from JSONL file produced by generate_mcq_cots.py."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


class MCQDataHandler(DataHandler):
    """Data handler for OpenBookQA with structured reasoning CoTs."""

    def __init__(self, system_prompt_path: str = "context/structured_reasoning_prompt.tx"):
        self._openqa = OpenQA()
        self.id_key = self._openqa.id_key
        self.q_key = self._openqa.q_key

        prompt_path = system_prompt_path
        if not os.path.isabs(prompt_path):
            project_root = os.path.dirname(os.path.abspath(__file__))
            prompt_path = os.path.join(project_root, prompt_path)
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read().strip()
        super().__init__()

    def get_dataset_splits(self):
        return self._openqa.get_dataset_splits()

    def get_answer_letters(self, instance):
        return self._openqa.get_answer_letters(instance)

    def get_answer_choices(self, instance):
        return self._openqa.get_answer_choices(instance)

    def correct_answer_letter(self, instance):
        return self._openqa.correct_answer_letter(instance)

    def make_cot_prompt(self, instance, ct=False):
        question = instance.get("question_stem", instance.get("question", ""))
        choices = self.get_answer_choices(instance)
        choices_str = "\n".join(choices)
        return (
            f"{self.system_prompt}\n\n"
            f"### Problem Statement\n"
            f"1. **Question**: {question}\n"
            f"2. **Possible Answers**:\n{choices_str}\n"
            f"### Solve\n"
        )

    def make_answer_prompt(self, prefix, cot_text="", ct=False):
        return (
            f"{prefix}{cot_text}\n"
            f"{BOWMAN_HUMAN_ANSWER_PREFIX}\n"
            f"{BOWMAN_ASSISTANT_ANSWER_PREFIX}"
        )

    def make_bowman_demonstration(self, instance):
        return self._openqa.make_bowman_demonstration(instance)
