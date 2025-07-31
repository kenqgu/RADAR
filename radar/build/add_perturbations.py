"""Adds perturbations to a given task path."""

import pandas as pd
from typing import List
from radar.tasks import datamodel
from radar.data import transform_spec, datamodel as datamodel_data


def is_percent_difference_exceeded(a: float, b: float, x: float) -> bool:
    """Returns True if a and b differ by more than x percent."""
    if a == 0 and b == 0:
        return False  # No difference
    if a == 0 or b == 0:
        return True  # One is zero, the other is not → infinite percent difference
    percent_diff = (abs(a - b) / (abs(a))) * 100
    return percent_diff > x


def add_perturbations_from_df(
    df: pd.DataFrame,
    num_tokens: int,
    token_bucket: int,
    metadata: datamodel.TaskMetadata,
) -> List[datamodel_data.TaskInstance]:
    """Adds perturbations to a given dataframe.
    Args:
      df: The dataframe to add perturbations to.
      metadata: The metadata for the task.
    Returns:
      A dictionary of task instances where the key is a tuple of (num_cols, token_bucket, artifact_type).
    """
    spec = transform_spec.generate_transform_spec_delete_overwrite(df, df)
    answer_func = metadata.get_answer_func()
    if answer_func is None:
        raise ValueError(f"No answer function found for task: {metadata.task_id}")

    task_instances = []
    task_instance = datamodel_data.TaskInstance(
        task_id=metadata.task_id,
        query=metadata.query,
        query_cols=metadata.query_cols,
        table=datamodel_data.DataTable.from_df(df),
        recovered_tables_transform_spec=[spec.model_dump()],
        artifact_scope="clean",
        artifact_type="clean",
        artifact_reasoning_cols=[],
        base_data_num_tokens=num_tokens,
        base_data_token_bucket=token_bucket,
        num_rows=df.shape[0],
        num_cols=len(df.columns),
        answer=answer_func(df),
    )

    task_instances.append(task_instance)

    for perturb_type in [
        "missingness",
        "bad-values",
        "inconsistent-formatting",
        "inconsistent-commonsense-logic",
        "outliers",
    ]:
        perturb_func = metadata.get_perturbation_func(perturb_type)
        if perturb_func is None:
            continue
        perturbation_return: datamodel.PerturbationReturn = perturb_func(df)
        if len(perturbation_return.recovered_tables) == 1:
            answer = answer_func(perturbation_return.recovered_tables[0])
        else:
            answer = [answer_func(df) for df in perturbation_return.recovered_tables]
            if isinstance(answer[0], list):
                answer = [item for sublist in answer for item in sublist]
        task_instance = datamodel_data.TaskInstance(
            task_id=metadata.task_id,
            query=metadata.query,
            artifact_type=perturbation_return.artifact_type,
            artifact_scope=perturbation_return.artifact_scope,
            query_cols=metadata.query_cols,
            artifact_reasoning_cols=perturbation_return.artifact_reasoning_cols,
            table=perturbation_return.table,
            base_data_num_tokens=num_tokens,
            base_data_token_bucket=token_bucket,
            num_rows=len(perturbation_return.table.rows),
            num_cols=len(perturbation_return.table.headers),
            recovered_tables_transform_spec=perturbation_return.recovered_delta_specs,
            perturbation_note=perturbation_return.perturbation_note,
            answer=answer,
        )
        task_instances.append(task_instance)
    return task_instances
