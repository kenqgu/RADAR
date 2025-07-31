from typing import List, Optional, Set, Union

import numpy as np
import pandas as pd

from radar.data import datamodel, transform_spec
from radar.tasks import datamodel as build_datamodel


def get_random_indices(
    df: pd.DataFrame, percentage: float, seed: int = 42
) -> List[int]:
    """Helper function to get random indices from a dataframe based on percentage."""
    num_random = max(1, int(len(df) * percentage))
    np.random.seed(seed)
    return np.random.choice(df.index, num_random, replace=False)


def get_valid_indices(df: pd.DataFrame, columns: List[str]) -> pd.Index:
    """Helper function to get indices where specified columns are not empty."""
    valid_indices = df.index
    for col in columns:
        valid_indices = valid_indices.intersection(df.index[df[col] != ""])
    return valid_indices


def create_perturbation_return(
    df_perturbed: pd.DataFrame,
    df_recovered: Union[pd.DataFrame, List[pd.DataFrame]],
    artifact_type: datamodel.ArtifactType,
    artifact_scope: datamodel.ArtifactScope,
    artifact_reasoning_cols: List[str],
    perturbation_note: Optional[str] = None,
) -> build_datamodel.PerturbationReturn:
    """Helper function to create a PerturbationReturn object."""

    df_perturbed = df_perturbed.round(4).astype(str)

    table = datamodel.DataTable(
        headers=df_perturbed.columns.tolist(),
        rows=df_perturbed.values.tolist(),
    )

    if isinstance(df_recovered, list):
        df_recovered = [df.round(4) for df in df_recovered]
    else:
        df_recovered = [df_recovered.round(4)]

    return build_datamodel.PerturbationReturn(
        table=table,
        recovered_tables=df_recovered,
        recovered_delta_specs=[
            transform_spec.generate_transform_spec_delete_overwrite(
                df_perturbed, df.astype(str)
            )
            for df in df_recovered
        ],
        artifact_type=artifact_type,
        artifact_scope=artifact_scope,
        artifact_reasoning_cols=artifact_reasoning_cols,
        perturbation_note=perturbation_note,
    )


def get_non_overlapping_random_indices(
    df: pd.DataFrame,
    percentages: List[float],
    inds_to_include: Optional[Set[int]] = None,
    inds_to_exclude: Optional[Set[int]] = None,
    seed: int = 42,
) -> List[List[int]]:
    """
    Helper function to get multiple sets of non-overlapping random indices from a dataframe.
    Each set is based on a different percentage of the total rows.

    Args:
        df: The input DataFrame
        percentages: List of percentages (0-1) for each set of indices
        inds_to_exclude: Set of indices to exclude from the random selection
        seed: Random seed for reproducibility

    Returns:
        List of lists containing non-overlapping indices for each percentage
    """
    np.random.seed(seed)
    if inds_to_include is not None:
        all_indices = set(inds_to_include)
    else:
        all_indices = set(df.index)
    if inds_to_exclude:
        all_indices -= set(inds_to_exclude)
    result_indices = []

    for percentage in percentages:
        num_random = max(1, int(len(all_indices) * percentage))
        if len(all_indices) < num_random:
            raise ValueError(
                f"Not enough remaining indices for percentage {percentage}"
            )

        selected_indices = np.random.choice(
            list(all_indices), num_random, replace=False
        )
        result_indices.append(selected_indices.tolist())
        all_indices -= set(selected_indices)

    return result_indices


def scale_values_above_threshold(
    df: pd.DataFrame,
    index: int,
    base_col: str,
    threshold: float = 10000,
) -> float:
    """
    Scale values in a DataFrame above a given threshold while maintaining proportional relationships.

    Args:
        df: DataFrame containing the data
        index: Index of the row to modify
        base_col: Name of the column to use as base for scaling
        threshold: Minimum value to scale above (default: 10000)
    """
    current_value = df.loc[index, base_col]
    if isinstance(current_value, pd.Series):
        current_value = current_value.iloc[0]
    if current_value >= threshold:
        return  # No need to scale if already above threshold

    min_multiplier = threshold / current_value
    random_multiplier = min_multiplier + np.random.random()
    df.loc[index, base_col] = int(current_value * random_multiplier)
    return random_multiplier


def scale_values_below_threshold(
    df: pd.DataFrame,
    index: int,
    base_col: str,
    threshold: float = 30,
) -> float:
    """
    Scale values in a DataFrame below a given threshold while maintaining proportional relationships.

    Args:
        df: DataFrame containing the data
        index: Index of the row to modify
        base_col: Name of the column to use as base for scaling
        threshold: Maximum value to scale below (default: 30)
    """
    current_value = df.loc[index, base_col]
    if isinstance(current_value, pd.Series):
        current_value = current_value.iloc[0]
    if current_value <= threshold:
        return  # No need to scale if already below threshold

    min_value = current_value / threshold
    random_multiplier = min_value + np.random.random()
    df.loc[index, base_col] = int(current_value / random_multiplier)
    return random_multiplier


def number_to_words(n: int) -> str:
    if n == 0:
        return "zero"

    def one(num):
        return [
            "",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
        ][num]

    def two_less_20(num):
        return [
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ][num - 10]

    def ten(num):
        return [
            "",
            "",
            "twenty",
            "thirty",
            "forty",
            "fifty",
            "sixty",
            "seventy",
            "eighty",
            "ninety",
        ][num]

    def two(num):
        if num == 0:
            return ""
        elif num < 10:
            return one(num)
        elif num < 20:
            return two_less_20(num)
        else:
            tenner = num // 10
            rest = num % 10
            return ten(tenner) + ("-" + one(rest) if rest != 0 else "")

    def three(num):
        hundred = num // 100
        rest = num % 100
        if hundred and rest:
            return one(hundred) + " hundred " + two(rest)
        elif not hundred and rest:
            return two(rest)
        elif hundred and not rest:
            return one(hundred) + " hundred"
        else:
            return ""

    billion = n // 1_000_000_000
    million = (n // 1_000_000) % 1_000
    thousand = (n // 1_000) % 1_000
    remainder = n % 1_000

    result = ""
    if billion:
        result += three(billion) + " billion "
    if million:
        result += three(million) + " million "
    if thousand:
        result += three(thousand) + " thousand "
    if remainder:
        result += three(remainder)

    return result.strip()
