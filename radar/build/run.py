from typing import List, Optional
import pandas as pd
import os.path as osp
import os
from typing import Dict, Tuple
from radar.build import add_perturbations, size_by_tokens
from radar.logger import logger
from radar.data import datamodel
from radar import utils
from tqdm import tqdm


def build_data(
    task_dir: str,
    num_cols_list: Optional[List[int]] = [10],
    token_buckets: Optional[List[int]] = [2000, 4000, 8000, 16000],
    save_dir: Optional[str] = None,
):
    df, metadata = size_by_tokens.load_df_and_metadata(task_dir)
    logger.info(
        f"Building task {metadata.task_id}, sampling tables with different numbers of columns..."
    )
    dfs = size_by_tokens.sample_table_cols(df, metadata, num_cols_list)
    logger.info(
        f"Building task {metadata.task_id}, filtering tables based on token count..."
    )
    dfs_filt, df_filt_summary = size_by_tokens.filter_df_based_on_token_count(
        dfs, token_buckets
    )
    logger.info(f"Building task {metadata.task_id}, adding perturbations...")
    all_task_instances: List[datamodel.TaskInstance] = []

    # Create progress bar for processing each table configuration
    pbar = tqdm(
        dfs_filt.items(),
        desc=f"Processing tables for {metadata.task_id}",
        total=len(dfs_filt),
    )

    for (num_cols, num_tokens, token_bucket), df in pbar:
        pbar.set_description(
            f"Processing ncols={num_cols}, token_bucket={token_bucket}"
        )
        if add_perturbations.is_percent_difference_exceeded(
            num_tokens, token_bucket, 10
        ):
            logger.warning(
                f"Skipping token bucket because the token count difference is too large(token_count={num_tokens} vs bucket={token_bucket})"
            )
            continue
        task_instances = add_perturbations.add_perturbations_from_df(
            df, num_tokens, token_bucket, metadata
        )
        all_task_instances.extend(task_instances)

    logger.info(f"Saving {len(all_task_instances)} task instances to {save_dir}")
    if save_dir is None:
        save_dir = task_dir
    save_build_data(save_dir, dfs_filt, df_filt_summary, all_task_instances)


def save_build_data(
    save_dir: str,
    dfs_filt: Dict[Tuple[int, int, int], pd.DataFrame],
    df_filt_summary: pd.DataFrame,
    all_task_instances: List[datamodel.TaskInstance],
):
    """Saves the build data to the given directory.
    Args:
        save_dir: The directory to save the build data to.
        dfs_filt: The filtered dataframes. A dictionary of dataframes, where the keys are the number of columns, number of tokens, and the token bucket in the dataframe
        and the values are the corresponding dataframes.
        df_filt_summary: The metadata of the filtered dataframes.
        all_task_instances: The task instances. A dictionary of task instances, where the keys are the number of columns, number of tokens, and the token bucket in the task instance
        and the values are the corresponding task instances.
    """
    save_dir_token_buckets = osp.join(save_dir, "tables_token_buckets")
    os.makedirs(save_dir_token_buckets, exist_ok=True)
    df_filt_summary.to_csv(osp.join(save_dir_token_buckets, "summary.csv"), index=False)
    for (num_cols, num_tokens, token_bucket), df in dfs_filt.items():
        save_path = osp.join(
            save_dir_token_buckets,
            f"ncols={num_cols}_ntokens={num_tokens}_token_bucket={token_bucket}.csv",
        )
        df.round(4).astype(str).to_csv(save_path, index=False)

    save_dir_tasks = osp.join(save_dir, "tasks")
    os.makedirs(save_dir_tasks, exist_ok=True)
    for task_instance in all_task_instances:
        save_path = osp.join(
            save_dir_tasks,
            f"{task_instance.task_id}__artifact={task_instance.artifact_type}__token_bucket={task_instance.base_data_token_bucket}__ncols={task_instance.num_cols}.json",
        )
        utils.write_json(task_instance.model_dump(mode="json"), save_path)
    logger.info(f"Saved {len(all_task_instances)} task instances to {save_dir_tasks}")


if __name__ == "__main__":
    build_data(
        task_dir="/projects/bdata/kenqgu/Research/Year4/radar_release/RADAR/task_example/influenza-like-illness",
        num_cols_list=[10, 20],
        token_buckets=[2000, 4000, 8000],
    )
