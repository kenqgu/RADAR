import glob
import os
from typing import List, Literal, Tuple

import dotenv
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from radar import utils
from radar.data import datamodel

dotenv.load_dotenv()


def load_task_instances_hf(
    split: Literal["full", "tasks", "sizes"],
) -> Tuple[List[datamodel.TaskInstance], pd.DataFrame]:
    if split == "full":
        suffix = ""
    else:
        suffix = f"-{split}"
    ds = load_dataset("kenqgu/radar", data_dir=f"radar{suffix}")
    df_summary: pd.DataFrame = ds["test"].to_pandas()  # type: ignore
    return [datamodel.TaskInstance(**d) for d in ds["test"]], df_summary  # type: ignore


if __name__ == "__main__":
    task_instances, df = load_task_instances_hf("tasks")
    print(len(task_instances))
