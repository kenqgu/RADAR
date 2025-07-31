"""Module for generating TableDeltaSpecs to transform a perturbed table to a recovered table table."""

import difflib
import io
import warnings
from typing import List, Union

import pandas as pd

from radar.data import datamodel, perturb

# Suppress FutureWarnings
warnings.simplefilter(action="ignore", category=FutureWarning)


def generate_transform_spec_delete_overwrite(
    df: pd.DataFrame,
    df_recovered: pd.DataFrame,
) -> perturb.TableDeltaSpec:
    """Generates a TableDeltaSpec to transform a perturbed table to a recovered table, focusing on deletions and overwrites.

    Args:
      df: The perturbed DataFrame.
      df_recovered: The recovered DataFrame.
    Returns:
      A TableDeltaSpec that describes the changes needed to transform the clean
      table to the perturbed table, only including deletions and overwrites.
    """

    df_str_lines = df.to_csv(index=False).strip().splitlines()
    df_recovered_str_lines = df_recovered.to_csv(index=False).strip().splitlines()

    header = df_str_lines[0]
    columns = header.split(",")

    sm = difflib.SequenceMatcher(None, df_str_lines[1:], df_recovered_str_lines[1:])
    delete_rows = []
    overwrite_cells = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        elif tag == "delete":
            # Add all rows in the deleted range to delete_rows
            for i in range(i1, i2):
                delete_rows.append(i)
        elif tag == "replace":
            # Handle overwrites for matching rows
            for offset in range(min(i2 - i1, j2 - j1)):
                i = i1 + offset
                line1 = df_str_lines[i + 1]
                line2 = df_recovered_str_lines[j1 + offset + 1]
                row1 = pd.read_csv(io.StringIO(f"{header}\n{line1}"), dtype=str).iloc[0]
                row2 = pd.read_csv(io.StringIO(f"{header}\n{line2}"), dtype=str).iloc[0]
                for col in columns:
                    val1 = row1[col]
                    val2 = row2[col]
                    if pd.isna(val1) and pd.isna(val2):
                        continue
                    if val1 != val2:
                        overwrite_cells.append(
                            perturb.OverwriteCell(
                                row=i,
                                col=col,
                                new_value=None if pd.isna(val2) else val2,
                            )
                        )
            # If there are more rows in clean than perturbed, delete the extra rows
            if i2 - i1 > j2 - j1:
                for i in range(i1 + (j2 - j1), i2):
                    delete_rows.append(i)

    return perturb.TableDeltaSpec(
        delete_rows=delete_rows, overwrite_cells=overwrite_cells
    )
