from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from radar.evaluate import datamodel


def round_floats(d: Any, decimals=3) -> Any:
    if isinstance(d, dict):
        return {k: round_floats(v, decimals) for k, v in d.items()}
    elif isinstance(float(d), float):  # catches floats and NaNs
        try:
            return round(float(d), decimals)
        except (TypeError, ValueError):
            return d
    return d


def process_results(
    task_results: List[datamodel.TaskInstanceRunResult],
    filter_by_successful_tasks: bool = False,
    filter_by_unsuccessful_tasks: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    assert not (
        filter_by_successful_tasks and filter_by_unsuccessful_tasks
    ), "Cannot filter by both successful and unsuccessful tasks"
    if filter_by_successful_tasks:
        task_results = [result for result in task_results if result.is_correct]
    if filter_by_unsuccessful_tasks:
        task_results = [result for result in task_results if not result.is_correct]

    df_combined = pd.DataFrame(
        [
            result.to_dict()
            for result in task_results
            if isinstance(result, datamodel.TaskInstanceRunResult)
        ]
    )

    num_error_encountered = sum(1 for result in task_results if isinstance(result, str))

    df_combined["is_correct"] = df_combined["is_correct"].apply(
        lambda x: 1 if x == True or x == "TRUE" else 0
    )

    def agg_with_count(grouped_data):
        result = {}
        for key, group in grouped_data:
            if isinstance(key, Iterable) and not isinstance(key, (str, bytes)):
                sanitized_key = "_".join(
                    str(int(k)) if isinstance(k, np.integer) else str(k) for k in key
                )
            else:
                sanitized_key = str(key)
            result[sanitized_key] = {
                "accuracy": round(group["is_correct"].mean(), 3),
                "count": len(group),
            }
        return result

    results: Dict[str, Dict[str, Any]] = {
        "overall": {
            "all": {
                "accuracy": round(df_combined["is_correct"].mean(), 3),
                "count": len(df_combined),
            },
            "num_error_encountered": {
                "accuracy": -1,
                "count": num_error_encountered,
            },
        },
    }

    if "artifact_type" in df_combined.columns:
        results["by_artifact_type"] = agg_with_count(
            df_combined.groupby(df_combined["artifact_type"])
        )
    if "token_bucket" in df_combined.columns:
        results["by_token_bucket"] = agg_with_count(
            df_combined.groupby(df_combined["token_bucket"])
        )
    if "num_cols" in df_combined.columns:
        results["by_num_cols"] = agg_with_count(
            df_combined.groupby(df_combined["num_cols"])
        )
    if "task_id" in df_combined.columns:
        results["by_task"] = agg_with_count(df_combined.groupby("task_id"))

    results = round_floats(results)
    return df_combined, flatten_results_to_df(results)


def flatten_results_to_df(results: Dict[str, Any]):
    """Flattens a nested results dictionary into a Pandas DataFrame.
    Args:
        results: A dictionary where keys are categories and values are either
          single success rates or dictionaries of subgroup success rates.
    Returns:
        A Pandas DataFrame with 'category', 'subgroup', and 'accuracy'
        columns.
    """
    records = []
    for category, data in results.items():
        if isinstance(data, dict):
            sorted_items = sorted(data.items())
            for subgroup, stats in sorted_items:
                records.append(
                    (
                        category,
                        str(subgroup),
                        stats["accuracy"],
                        int(stats["count"]),
                    )
                )
        else:
            records.append(
                (category, "all", results["overall"], int(len(results)))
            )  # 'count' is len of all results, fallback
    df = pd.DataFrame(
        records,
        columns=[
            "category",
            "subgroup",
            "accuracy",
            "count",
        ],
    )
    df.set_index(["category", "subgroup"], inplace=True)
    return df
