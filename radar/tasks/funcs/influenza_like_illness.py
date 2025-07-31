import random

import pandas as pd

from radar.tasks import datamodel
from radar.tasks import registry as task_registry
from radar.tasks import utils as perturb_utils

# "Of the recordings in the dataset, what is the median number of ILI cases with AGE 25-64 for a week?
# Return your answer rounded to the nearest 1 decimal place."


@task_registry.register_answer_function("influenza-like-illness")
def influenza_like_illness_answer(df: pd.DataFrame) -> int:
    df = df.copy(deep=True)
    return round(df["ILI AGE 25-64"].median(), 1)


@task_registry.register_missing_data_function("influenza-like-illness")
def influenza_like_illness_missing_data(
    df: pd.DataFrame,
) -> datamodel.PerturbationReturn:
    df_clean = df.copy(deep=True)
    df_perturbed = df.copy(deep=True)
    inds = perturb_utils.get_non_overlapping_random_indices(
        df,
        [0.05],
    )

    df_perturbed.loc[inds[0], "ILI AGE 25-64"] = ""

    return perturb_utils.create_perturbation_return(
        df_perturbed=df_perturbed,
        df_recovered=df_clean,
        artifact_type="missingness",
        artifact_scope="connected-multi-column",
        artifact_reasoning_cols=[
            "ILI AGE 25-64",
        ],
        perturbation_note="Introduced missingness in ILI AGE 25-64 column. Can be recovered by using the formula ILI AGE 25-64 = ILI AGE 25-49 + ILI AGE 50-64.",
    )


@task_registry.register_outliers_function("influenza-like-illness")
def influenza_like_illness_outliers(
    df: pd.DataFrame,
) -> datamodel.PerturbationReturn:
    df_clean = df.copy(deep=True)
    df_perturbed = df.copy(deep=True)
    inds = perturb_utils.get_non_overlapping_random_indices(
        df,
        [0.05],
    )
    # TOTAL be a few million so its an obvious outlier
    perturb_utils.scale_values_above_threshold(
        df_perturbed, inds[0], "ILI AGE 25-64", 30000000
    )
    df_perturbed["ILI AGE 25-64"] = df_perturbed["ILI AGE 25-64"].astype(int)
    df_perturbed.loc[inds[0], "ILI AGE 25-49"] = (
        df_perturbed.loc[inds[0], "ILI AGE 25-64"]
        - df_perturbed.loc[inds[0], "ILI AGE 50-64"]
    )
    df_clean = df_clean.loc[~df_clean.index.isin(inds[0])]
    return perturb_utils.create_perturbation_return(
        df_perturbed=df_perturbed,
        df_recovered=df_clean,
        artifact_type="outliers",
        artifact_scope="naive-multi-column",
        artifact_reasoning_cols=[
            "ILI AGE 25-64",
            "ILI AGE 25-49",
        ],
        perturbation_note="Introduced a large obvious outliers in ILI AGE 25-64 and ILI AGE 25-49 columns. Need to ignore this row.",
    )


@task_registry.register_bad_values_function("influenza-like-illness")
def influenza_like_illness_bad_values(
    df: pd.DataFrame,
) -> datamodel.PerturbationReturn:
    df_clean = df.copy(deep=True)
    df_perturbed = df.copy(deep=True)
    inds = perturb_utils.get_non_overlapping_random_indices(
        df,
        [0.05],
    )
    df_perturbed.loc[inds[0], "ILI AGE 25-64"] = -9999
    df_perturbed.loc[inds[0], "ILI AGE 25-49"] = "000000"

    df_clean = df_clean.loc[~df_clean.index.isin(inds[0])]
    return perturb_utils.create_perturbation_return(
        df_perturbed=df_perturbed,
        df_recovered=df_clean,
        artifact_type="bad-values",
        artifact_scope="naive-multi-column",
        artifact_reasoning_cols=[
            "ILI AGE 25-64",
            "ILI AGE 25-49",
        ],
        perturbation_note="Introduced bad values in ILI AGE 25-64 and ILI AGE 25-49 columns. Need to ignore these rows.",
    )


@task_registry.register_inconsistent_formatting_function("influenza-like-illness")
def influenza_like_illness_formatting(
    df: pd.DataFrame,
) -> datamodel.PerturbationReturn:
    df_clean = df.copy(deep=True)
    df_perturbed = df.copy(deep=True)
    inds = perturb_utils.get_non_overlapping_random_indices(
        df,
        [0.05, 0.05],
    )
    df_perturbed.loc[inds[0], "ILI AGE 25-64"] = (
        df_perturbed.loc[inds[0], "ILI AGE 25-64"]
        .astype(int)
        .apply(lambda x: f"{x:,} people")
    )
    df_perturbed.loc[inds[1], "ILI AGE 25-49"] = df_perturbed.loc[
        inds[1], "ILI AGE 25-49"
    ].apply(lambda x: f"{x:,} people")

    return perturb_utils.create_perturbation_return(
        df_perturbed=df_perturbed,
        df_recovered=df_clean,
        artifact_type="inconsistent-formatting",
        artifact_scope="naive-multi-column",
        artifact_reasoning_cols=[
            "ILI AGE 25-64",
            "ILI AGE 25-49",
        ],
        perturbation_note="Introduced formatting inconsistencies in ILI AGE 25-64 and ILI AGE 25-49 columns.",
    )


@task_registry.register_inconsistent_logic_function("influenza-like-illness")
def influenza_like_illness_inconsistent_logic(
    df: pd.DataFrame,
) -> datamodel.PerturbationReturn:
    df_clean = df.copy(deep=True)
    df_perturbed = df.copy(deep=True)
    inds = perturb_utils.get_non_overlapping_random_indices(
        df,
        [0.05],
    )
    random.seed(42)
    # duplicate year and state values
    df_perturbed.loc[inds[0], "ILI AGE 25-64"] = df_perturbed.loc[
        inds[0], "ILI AGE 25-64"
    ].apply(lambda x: x + random.randint(300, 600))

    df_clean1 = df_clean.loc[~df_clean.index.isin(inds[0])]

    # can be recovered by using the formula ILI AGE 25-64 = ILI AGE 25-49 + ILI AGE 50-64.
    return perturb_utils.create_perturbation_return(
        df_perturbed=df_perturbed,
        df_recovered=[df_clean1, df_clean],
        artifact_type="inconsistent-commonsense-logic",
        artifact_scope="connected-multi-column",
        artifact_reasoning_cols=[
            "ILI AGE 25-64",
            "ILI AGE 25-49",
        ],
        perturbation_note="Introduced an inconsistency in the sales column is greater than the product of quantity ordered and price each column. Can be recovered by using the formula SALES = QUANTITYORDERED * PRICEEACH.",
    )
