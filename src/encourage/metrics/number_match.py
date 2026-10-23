"""Evaluation metric for numeric and boolean matches in exact answers and free text."""

import math
import re
from typing import Any

import numpy as np

from encourage.llm.response_wrapper import ResponseWrapper
from encourage.metrics.metric import Metric, MetricOutput
from encourage.metrics.registry import register_metric


@register_metric("NumberMatch")
class NumberMatch(Metric):
    """Computes exact and free-text number matching scores for generated answers.

    Supports exact numeric matching (with configurable epsilon tolerance), boolean
    answers, and automatic fallback to numerical recall extraction when answers
    are embedded in natural language sentences.
    """

    def __init__(self, epsilon: float = 1e-2, as_percentage: bool = False) -> None:
        """Initialize the NumberMatch metric.

        Args:
            epsilon: Tolerance for floating-point comparisons.
            as_percentage: If True, scale final score to [0, 100], otherwise [0.0, 1.0].

        """
        super().__init__(
            name="number_match",
            description="Match numbers, numeric ranges, and booleans in generated answers",
            required_meta_data=["reference_answer"],
        )
        self.epsilon = epsilon
        self.as_percentage = as_percentage

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calculate the NumberMatch metric across all responses."""
        self.validate_nested_keys(responses)

        scores: list[float] = []
        details: list[dict[str, Any]] = []

        for r in responses:
            pred = str(r.response) if r.response is not None else ""
            ref_raw = r.meta_data["reference_answer"] if r.meta_data else ""
            gt = str(ref_raw) if ref_raw is not None else ""
            score, detail = self.compute_score(pred, gt, self.epsilon)
            scores.append(score)
            details.append(detail)

        mean_score = float(np.mean(scores)) if scores else 0.0
        if self.as_percentage:
            mean_score *= 100.0

        return MetricOutput(score=mean_score, raw=scores, misc={"details": details})

    @staticmethod
    def _is_number(val: str) -> bool:
        """Check if string can be converted to float."""
        try:
            float(str(val).strip().replace(",", "."))
            return True
        except (ValueError, TypeError, AttributeError):
            return False

    @staticmethod
    def _check_boolean(prediction: str, ground_truth: str) -> bool | None:
        """Check if inputs represent matching boolean values."""
        bool_map = {
            "yes": True,
            "ja": True,
            "true": True,
            "1": True,
            "no": False,
            "nein": False,
            "false": False,
            "0": False,
        }
        gt_norm = str(ground_truth).strip().lower()
        if gt_norm in bool_map:
            pred_norm = str(prediction).strip().lower()
            if pred_norm in bool_map:
                return bool_map[pred_norm] == bool_map[gt_norm]
            first_word = pred_norm.split()[0].rstrip(",.:;!?") if pred_norm else ""
            if first_word in bool_map:
                return bool_map[first_word] == bool_map[gt_norm]
        return None

    @staticmethod
    def extract_numbers(text: str) -> list[str]:
        """Extract all numerical strings, norm identifiers, and decimal values from text."""
        if not text:
            return []
        clean_text = re.sub(r"(?<=\d),(?=\d)", ".", str(text))
        found = re.findall(r"\b\d+(?:\.\d+)?\b", clean_text)
        return sorted(set(found))

    @classmethod
    def compute_score(
        cls, prediction: str, ground_truth: str, epsilon: float = 1e-2
    ) -> tuple[float, dict[str, Any]]:
        """Compute match score between prediction and ground truth.

        Checks direct boolean matches, exact numeric equality (using epsilon),
        and falls back to numerical recall when answers contain natural text.
        """
        # 1. Boolean check
        bool_res = cls._check_boolean(prediction, ground_truth)
        if bool_res is not None:
            score = 1.0 if bool_res else 0.0
            return score, {"type": "boolean", "matched": bool_res}

        # 2. Pure numeric exact match
        if cls._is_number(prediction) and cls._is_number(ground_truth):
            matched = cls.compute_exact_score(prediction, ground_truth, epsilon)
            score = 1.0 if matched else 0.0
            return score, {
                "type": "exact_number",
                "matched": matched,
                "prediction": prediction,
                "ground_truth": ground_truth,
            }

        # 3. Text fallback: extract numbers and compute recall
        ref_nums = cls.extract_numbers(ground_truth)
        pred_nums = cls.extract_numbers(prediction)

        if not ref_nums:
            return 1.0, {
                "reference_numbers": [],
                "predicted_numbers": pred_nums,
                "recall": 1.0,
            }

        matched_count = 0
        for g in ref_nums:
            if g in pred_nums:
                matched_count += 1
                continue
            if cls._is_number(g):
                g_val = float(g)
                if any(cls._is_number(p) and abs(float(p) - g_val) < epsilon for p in pred_nums):
                    matched_count += 1

        recall = float(matched_count / len(ref_nums))
        return recall, {
            "reference_numbers": ref_nums,
            "predicted_numbers": pred_nums,
            "recall": recall,
        }

    @staticmethod
    def compute_exact_score(prediction: str, ground_truth: str, epsilon: float = 1e-2) -> bool:
        """Given two strings pd and gt, find out whether pd is a correct answer to gt."""
        try:
            ground_truth_num = float(str(ground_truth).strip().replace(",", "."))
            prediction_num = float(str(prediction).strip().replace(",", "."))
        except (ValueError, TypeError):
            return False

        ground_truth_abs = abs(ground_truth_num)
        prediction_abs = abs(prediction_num)
        if ground_truth_abs < epsilon:
            return prediction_abs < epsilon
        if prediction_abs < epsilon:
            return False
        quotient = prediction_abs / ground_truth_abs
        quotient *= 0.1 ** math.floor(math.log10(quotient))
        if quotient > 5:
            quotient *= 0.1
        return abs(quotient - 1) < epsilon
