import os.path as osp
import shutil
from pathlib import Path
from typing import List, Optional

import typer

from radar.build import run as run_build

build_task = typer.Typer(name="build-task")


def parse_comma_separated_ints(comma_separated_ints: str) -> List[int]:
    return [int(num_cols) for num_cols in comma_separated_ints.split(",")]


@build_task.command()
def build(
    task_dir: Path = typer.Argument(
        help="The directory containing a `data.csv` and `metadata.yaml` file to build tasks from.",
    ),
    num_cols_list: str = typer.Option(
        default="10",
        help=(
        "A comma-separated list of column counts to use when generating tables. "
        "Each value specifies the number of columns for one table configuration."
    ),
        callback=parse_comma_separated_ints,
    ),
    token_buckets: str = typer.Option(
        default="2000,4000,8000,16000",
        help=(
            "A comma-separated list of token bucket counts to use when generating tables. "
            "Each value specifies the token count for one table configuration."
        ),
        callback=parse_comma_separated_ints,
    ),
    save_dir: Optional[Path] = typer.Option(
        default=None, help="The directory to save the task to, defaults to task_dir"
    ),
    overwrite: bool = typer.Option(
        default=True,
        help="Overwrite existing directories if they exist.",
    ),
):

    if save_dir is None:
        save_dir = task_dir

    if (
        osp.exists(osp.join(save_dir, "tables_token_buckets"))
        or osp.exists(osp.join(save_dir, "tasks"))
    ) and not overwrite:
        confirm = typer.confirm(
            "The directory 'tables_token_buckets' or 'tasks' (where task instances are saved) already exists. Do you want to overwrite it?",
            default=False,
        )
        if not confirm:
            raise typer.Abort()

    # Delete existing folders if they exist
    if osp.exists(osp.join(save_dir, "tables_token_buckets")):
        typer.echo("Deleting existing tables_token_buckets directory...")
        shutil.rmtree(osp.join(save_dir, "tables_token_buckets"))

    if osp.exists(osp.join(save_dir, "tasks")):
        typer.echo("Deleting existing tasks directory...")
        shutil.rmtree(osp.join(save_dir, "tasks"))

    run_build.build_data(task_dir, num_cols_list, token_buckets, save_dir)


def main():
    build_task()


if __name__ == "__main__":
    build_task()
