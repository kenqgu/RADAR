from functools import cached_property
from typing import Any, Callable, List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict

from radar.data import datamodel, perturb
from radar.logger import logger
from radar.tasks import registry


class PerturbationReturn(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    table: datamodel.DataTable
    recovered_tables: List[pd.DataFrame]
    recovered_delta_specs: List[perturb.TableDeltaSpec]
    artifact_type: datamodel.ArtifactType
    artifact_scope: datamodel.ArtifactScope
    artifact_reasoning_cols: List[str]
    perturbation_note: Optional[str] = None

    @cached_property
    def table_df(self) -> pd.DataFrame:
        return self.table.to_df()


class TaskMetadata(BaseModel):
    """Metadata for a task."""

    task_id: str
    query: str
    query_cols: List[str]
    minimum_columns: List[str]
    dataset_source: Optional[str] = None
    id_columns: Optional[List[str]] = (
        None  # columns that should always be on the left side of the table
    )

    def get_answer_func(self) -> Callable[[pd.DataFrame], Any] | None:
        try:
            return registry.get_registered_answer_function(self.task_id)
        except KeyError:
            logger.warning(f"No answer function found for task: {self.task_id}")
            return None

    def get_perturbation_func(
        self, artifact_type: datamodel.ArtifactType
    ) -> Callable[[pd.DataFrame], PerturbationReturn] | None:
        try:
            if artifact_type == "missingness":
                return registry.get_registered_missing_data_function(self.task_id)
            elif artifact_type == "bad-values":
                return registry.get_registered_bad_values_function(self.task_id)
            elif artifact_type == "inconsistent-formatting":
                return registry.get_registered_inconsistent_formatting_function(
                    self.task_id
                )
            elif artifact_type == "inconsistent-commonsense-logic":
                return registry.get_registered_inconsistent_logic_function(self.task_id)
            elif artifact_type == "outliers":
                return registry.get_registered_outlier_function(self.task_id)
            else:
                logger.warning(
                    f"No perturbation function found for type: {artifact_type}"
                )
                return None
        except KeyError:
            logger.warning(f"No perturbation function found for type: {artifact_type}")
            return None
