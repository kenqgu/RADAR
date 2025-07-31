"""Builds tables of various sizes based on task metadata."""

import os
import os.path as osp
from typing import Dict, List, Tuple, Union, Callable
from pathlib import Path

import pandas as pd
import tqdm

from radar.build import sample_table
from radar import utils
from gemma import gm

from radar.tasks import datamodel


def count_tokens_gemma(text: str) -> int:
    tokenizer = gm.text.Gemma2Tokenizer()
    return len(tokenizer.encode(text))


def load_df_and_metadata(task_path: str):
    """Loads the dataframe and metadata from the task path."""
    task_path = Path(task_path)
    orig_df = pd.read_csv(osp.join(task_path, "data.csv"))
    metadata = datamodel.TaskMetadata(
        **utils.read_yaml(osp.join(task_path, "metadata.yaml"))
    )
    return orig_df, metadata


def sample_table_cols(
    orig_df: pd.DataFrame,
    metadata: datamodel.TaskMetadata,
    num_cols: List[int],
) -> Dict[int, pd.DataFrame]:
    """Builds and writes datasets of min and max sizes based on the task metadata.
    Args:
      orig_df: The original dataframe from which we sample subsets of columns.
      metadata: The metadata of the task.
      num_cols: The number of columns to sample.
    Returns:
      A dictionary of dataframes, where the keys are the number of columns in the dataframe
      and the values are the corresponding dataframes.
    """
    if min(num_cols) < (len(metadata.minimum_columns) + len(metadata.id_columns or [])):
        raise ValueError(
            f"Minimum number of columns based on metadata is {len(metadata.minimum_columns) + len(metadata.id_columns or [])}, but the smallest number of columns to sample is {min(num_cols)}"
        )
    if max(num_cols) > len(orig_df.columns):
        raise ValueError(
            f"Maximum number of columns in the df is {len(orig_df.columns)}, but the largest number of columns to sample is {max(num_cols)}"
        )

    columns_order_cache: Dict[int, List[str]] = {}
    dfs: Dict[Tuple[str, int, int], pd.DataFrame] = {}
    for num_col in num_cols:
        if num_col not in columns_order_cache:
            sample_df = sample_table.sample_table(
                orig_df,
                keep_columns=metadata.minimum_columns,
                total_columns=num_col,
                num_rows=len(orig_df),
                id_columns=metadata.id_columns,
                random_state=42,
            )
            columns_order_cache[num_col] = sample_df.columns.tolist()
        else:
            sample_df = sample_table.sample_table(
                orig_df,
                keep_columns=columns_order_cache[num_col],
                total_columns=num_col,
                num_rows=len(orig_df),
                random_state=42,
            )
        dfs[num_col] = sample_df
    return dfs


def filter_df_based_on_token_count(
    dfs: Dict[int, pd.DataFrame],
    token_buckets: List[int],
    count_token_func: Callable[[str], int] = count_tokens_gemma,
    min_rows: int = 10,
) -> Tuple[Dict[Tuple[int, int, int], pd.DataFrame], pd.DataFrame]:
    """Saves the DataFrame to the task path based on the token count."""
    df_metadata_list = []
    dfs_filt = {}

    for num_cols, df in dfs.items():
        for token_bucket in token_buckets:
            df_filt, num_tokens = filter_df_based_on_token_count_helper(
                df, token_bucket, count_token_func, min_rows
            )
            dfs_filt[(num_cols, num_tokens, token_bucket)] = df_filt
            df_metadata_list.append(
                {
                    "num_cols": num_cols,
                    "num_tokens": num_tokens,
                    "token_bucket": token_bucket,
                    "num_rows": len(df_filt),
                }
            )

    df_metadata = pd.DataFrame(df_metadata_list)
    return dfs_filt, df_metadata


def filter_df_based_on_token_count_helper(
    df: pd.DataFrame,
    token_count: int,
    count_token_func: Callable[[str], int],
    min_rows: int = 10,
) -> Tuple[pd.DataFrame, int]:
    """Finds the minimum number of rows such that df.to_csv(index=False) exceeds token_count.
    Returns full DataFrame if total token count never exceeds threshold.
    Args:
      df: Input DataFrame.
      token_count: Target token count.
      count_token_func: Function to count tokens in a string.
      min_rows: Minimum number of rows to consider.
    Returns:
      A tuple containing:
        - The scaled DataFrame.
        - The token count of the scaled DataFrame.
    """
    low = min_rows
    high = len(df)
    best_n = None
    best_token_count = float("inf")
    best_diff = float("inf")
    while low <= high:
        mid = (low + high) // 2
        csv_text = df.iloc[:mid].to_csv(index=False)
        tokens = count_token_func(csv_text)
        diff = abs(tokens - token_count)
        if mid >= min_rows and diff < best_diff:
            best_n = mid
            best_token_count = tokens
            best_diff = diff
        if tokens < token_count:
            low = mid + 1
        else:
            high = mid - 1
    if best_n is not None:
        result_df = df.iloc[:best_n]
        return result_df, best_token_count
    else:
        # Fallback: return full DataFrame
        full_csv = df.to_csv(index=False)
        full_tokens = count_token_func(full_csv)
        return df, full_tokens
