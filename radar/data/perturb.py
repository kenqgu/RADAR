from typing import List, Union

import pandas as pd
from pydantic import BaseModel


class OverwriteCell(BaseModel):
    row: int
    col: str
    new_value: Union[str, float, int, bool, None]


class TableDeltaSpec(BaseModel):
    drop_rows: List[int] = []
    overwrite_cells: List[OverwriteCell] = []


def apply_transform_spec(
    df: pd.DataFrame,
    spec: TableDeltaSpec,
) -> pd.DataFrame:
    """Applies a TableDeltaDeleteSpec to a DataFrame to recover the original state.

    Args:
        df: The DataFrame to apply the spec to.
        spec: The TableDeltaDeleteSpec containing delete and overwrite operations.
    Returns:
        The recovered DataFrame after applying the spec.
    """
    # Create a copy to avoid modifying the original
    df_recovered = df.copy(deep=True)

    # Apply overwrite operations
    for overwrite_op in spec.overwrite_cells:
        if overwrite_op.row < len(df_recovered):
            df_recovered.loc[overwrite_op.row, overwrite_op.col] = (
                overwrite_op.new_value
            )

    # Sort delete operations by index in reverse order to avoid index shifting
    delete_rows = sorted(spec.drop_rows, reverse=True)

    # Apply delete operations
    for delete_op in delete_rows:
        if delete_op < len(df_recovered):
            df_recovered = pd.concat(
                [
                    df_recovered.iloc[:delete_op],
                    df_recovered.iloc[delete_op + 1 :],
                ]
            ).reset_index(drop=True)

    return df_recovered
