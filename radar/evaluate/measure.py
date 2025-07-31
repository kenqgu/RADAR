import re
from typing import List, Union


def extract_value_from_answer(text: str):
    """Extracts the value within <answer> tags in a provided string."""
    prefix = "The answer is:"
    lower_text = text.lower()
    if prefix.lower() not in lower_text:
        raise ValueError(
            'No "The answer is:" found in the response. Please write your precise'
            ' answer after the prefix "The answer is:". Do not return any more'
            " thinking just the answer. If there is no answer, return The answer"
            " is: no answer found"
        )
    # Find the last occurrence (case-insensitive)
    idx = lower_text.rfind(prefix.lower())
    return text[idx + len(prefix) :].strip()


def extract_first_number(s: str):
    match = re.search(r"-?\d+(?:\.\d+)?", s)
    return match.group() if match else None


def match_answer(
    predicted: str,
    ground_truth: Union[
        str,
        float,
        int,
        List[str],
        List[float],
        List[int],
        List[List[Union[str, float, int]]],
    ],
) -> bool:
    """Evaluates a predicted answer against the ground truth.
    Args:
      predicted: The predicted answer string.
      ground_truth: The ground truth value, which can be a string, number, or a
        list of strings or numbers.
    Returns:
      True if the predicted answer matches the ground truth, False otherwise.
    """

    def normalize(val):
        if isinstance(val, str):
            return val.strip().lower()
        return val

    def is_float_match(predicted_str, gt):
        if not isinstance(gt, (float)):
            return False
        num_str = extract_first_number(predicted_str)
        if num_str is None:
            return False
        predicted_val = float(num_str)
        gt_str = str(gt)
        if "." in gt_str:
            decimal_places = len(gt_str.split(".")[-1])
            decimal_places_num_str = len(num_str.split(".")[-1])
            if decimal_places_num_str <= 3:
                decimal_places = max(decimal_places, decimal_places_num_str)
        else:
            decimal_places = 0
        float_tol = 10**-decimal_places
        return abs(predicted_val - float(gt)) <= float_tol

    def is_int_match(predicted_str, gt):
        if not isinstance(gt, int):
            return False
        num_str = extract_first_number(predicted_str)
        if num_str is None:
            return False
        try:
            num = float(num_str)
            return num.is_integer() and int(num) == gt
        except ValueError:
            return False

    def is_string_match(predicted_str, gt):
        return isinstance(gt, str) and normalize(predicted_str) == normalize(gt)

    def is_list_of_strings_match(predicted_str, gt_list):
        predicted_items = [normalize(p) for p in predicted_str.split(",")]
        gt_items = [normalize(g) for g in gt_list]
        return set(predicted_items) == set(gt_items)

    def is_list_of_numbers_match(predicted_str, gt_list):
        try:
            predicted_items = [float(p.strip()) for p in predicted_str.split(",")]
        except ValueError:
            return False
        if not all(isinstance(g, (int, float)) for g in gt_list):
            return False
        # Convert to rounded versions based on ground truth precision
        if len(gt_list) != len(predicted_items):
            return False
        for pred_val, gt_val in zip(sorted(predicted_items), sorted(gt_list)):
            if not is_float_match(str(pred_val), gt_val):
                return False
        return True

    def is_list_match(predicted_str, gt_list):
        if all(isinstance(g, str) for g in gt_list):
            return is_list_of_strings_match(predicted_str, gt_list)
        if all(isinstance(g, (int, float)) for g in gt_list):
            return is_list_of_numbers_match(predicted_str, gt_list)
        return False

    # Handle list of lists
    if isinstance(ground_truth, list) and all(
        isinstance(sub, list) for sub in ground_truth
    ):
        for sublist in ground_truth:
            if is_list_match(predicted, sublist):
                return True
        return False
    # Handle flat list of scalar values
    if isinstance(ground_truth, list):
        for gt in ground_truth:
            if is_string_match(predicted, gt):
                return True
            if is_int_match(predicted, gt):
                return True
            if is_float_match(predicted, gt):
                return True
        return False
    # Handle single scalar value
    if is_string_match(predicted, ground_truth):
        return True
    if is_int_match(predicted, ground_truth):
        return True
    if is_float_match(predicted, ground_truth):
        return True
    return False
