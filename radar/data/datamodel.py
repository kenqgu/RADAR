from functools import cached_property
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel

from radar.data import perturb
from radar.utils import convert_dict_of_lists_to_list_of_dicts

ArtifactType = Literal[
    "bad-values",
    "inconsistent-formatting",
    "inconsistent-commonsense-logic",
    "missingness",
    "outliers",
    "clean",
]

ArtifactScope = Literal[
    "single-column", "naive-multi-column", "connected-multi-column", "clean"
]


class DataTable(BaseModel):
    headers: List[str]
    rows: List[List[str]]

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "DataTable":
        return cls(headers=df.columns.tolist(), rows=df.astype(str).values.tolist())

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.headers)


class TaskInstance(BaseModel):
    task_id: str
    query: str
    artifact_type: ArtifactType
    artifact_scope: ArtifactScope
    query_cols: List[str]
    artifact_reasoning_cols: List[str]
    table: DataTable
    num_rows: int
    num_cols: int
    base_data_num_tokens: int
    base_data_token_bucket: int
    recovered_tables_transform_spec: (
        List[perturb.TableDeltaSpec]
        | Dict[Literal["drop_rows", "overwrite_cells"], List[Any]]
    )
    answer: Optional[Any] = None
    perturbation_note: Optional[str] = None
    _table_df: Optional[pd.DataFrame] = None
    _recovered_table_dfs: Optional[List[pd.DataFrame]] = None

    @cached_property
    def task_instance_id(self) -> str:
        return f"tid={self.task_id}__artifact-type={self.artifact_type}__ncols={self.num_cols}__token-bucket={self.base_data_token_bucket}"

    def model_post_init(self, __context: Any) -> None:
        if isinstance(self.recovered_tables_transform_spec, dict):
            self.recovered_tables_transform_spec = [
                perturb.TableDeltaSpec(**spec)
                for spec in convert_dict_of_lists_to_list_of_dicts(
                    self.recovered_tables_transform_spec
                )
            ]
        if self._table_df is None:
            self._table_df = self.table.to_df()
        recovered_table_dfs = []
        for i, spec in enumerate(self.recovered_tables_transform_spec):
            df = perturb.apply_transform_spec(self._table_df, spec)
            recovered_table_dfs.append(df)
        self._recovered_table_dfs = recovered_table_dfs

    @property
    def table_df(self) -> pd.DataFrame:
        return self._table_df

    @property
    def recovered_table_dfs(self) -> List[pd.DataFrame]:
        return self._recovered_table_dfs

    def get_prompt_info(self) -> Dict[str, Any]:
        if self.table_df is None:
            raise ValueError("Table is not available.")
        df = self.table_df
        return {
            "table": df.to_csv(index=False),
            "question": self.query,
        }

    def get_prompt_info_codegen_agent(self) -> Dict[str, Any]:
        return {
            "table": self.table_df,
            "question": self.query,
        }
