"""Utility for sampling rows and columns from a Pandas DataFrame."""

import random
from typing import List, Optional, Union
import pandas as pd


def sample_columns(
    df: pd.DataFrame,
    keep_columns: List[str],
    total_columns: int,
    id_columns: Optional[List[str]] = None,
):
    """Sample a specified number of columns from a DataFrame.
    Args:
      df: Input DataFrame to sample from
      keep_columns: List of column names that must be kept
      total_columns: Total number of columns to include in the sample
      id_columns: List of column names that must be included
    Returns:
      List of sampled column names
    Raises:
      KeyError: If any specified column doesn't exist in the DataFrame
    """
    # Get remaining columns to sample from
    remaining_columns = list(set(df.columns) - set(keep_columns) - set(id_columns))
    num_additional_columns = total_columns - len(keep_columns) - len(id_columns)
    # Randomly select additional columns
    additional_columns = random.sample(remaining_columns, num_additional_columns)
    # Combine keep_columns with additional columns
    # Split additional columns randomly into left and right portions
    num_left = random.randint(0, len(additional_columns))
    left_cols = random.sample(additional_columns, num_left)
    right_cols = [c for c in additional_columns if c not in left_cols]
    # Combine in order: left random cols + first keep col
    # + rest of keep cols + right random cols
    selected_columns = id_columns + left_cols + keep_columns + right_cols
    return selected_columns


def sample_table(
    df: pd.DataFrame,
    keep_columns: List[str],
    total_columns: int,
    num_rows: int,
    id_columns: Optional[List[str]] = None,
    random_state: Union[int, None] = None,
) -> pd.DataFrame:
    """Sample a specified number of rows and columns from a DataFrame.
    Args:
      df: Input DataFrame to sample from
      keep_columns: List of column names that must be kept
      total_columns: Total number of columns to include in the sample
      num_rows: Number of rows to sample
      id_columns: List of column names that must be included
      random_state: Random seed for reproducibility (optional)
    Returns:
      Sampled DataFrame with specified rows and columns
    Raises:
      ValueError: If requested number of rows exceeds available rows or
             if total_columns is less than keep_columns or greater than available
             columns
      KeyError: If any specified column doesn't exist in the DataFrame
    """
    if id_columns is None:
        id_columns = []
    # Validate keep_columns exist
    missing_cols = set(keep_columns) - set(df.columns)
    missing_id_cols = set(id_columns) - set(df.columns)
    if missing_cols:
        raise KeyError(f"Columns not found in DataFrame: {missing_cols}")
    if missing_id_cols:
        raise KeyError(f"Columns not found in DataFrame: {missing_id_cols}")
    # Validate total_columns constraints
    if total_columns < len(keep_columns) + len(id_columns):
        raise ValueError(
            f"total_columns ({total_columns}) must be greater than or equal to"
            f" keep_columns ({len(keep_columns)})"
        )
    if total_columns > len(df.columns):
        raise ValueError(
            f"total_columns ({total_columns}) cannot be greater than available"
            f" columns ({len(df.columns)})"
        )
    # Validate number of rows
    if num_rows > len(df):
        raise ValueError(
            f"Cannot sample {num_rows} rows from DataFrame with {len(df)} rows"
        )
    # Set random seed if provided
    if random_state is not None:
        random.seed(random_state)
    # Get remaining columns to sample from
    selected_columns = sample_columns(df, keep_columns, total_columns, id_columns)
    # Sample rows and select columns
    sampled_df = df.iloc[:num_rows][selected_columns]
    return sampled_df.reset_index(drop=True)
