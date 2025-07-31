import json
import os.path as osp
import time
from typing import Any, Dict, List

import orjson
import yaml

from radar.logger import logger


def get_src_dir():
    return osp.dirname(osp.abspath(__file__))


def get_project_dir():
    return osp.dirname(get_src_dir())


def get_conf_dir():
    return osp.join(get_project_dir(), "conf")


def format_bytes(num_bytes: int) -> str:
    """
    Format bytes into human readable string with appropriate unit (B, KB, MB, GB, TB).

    Args:
        num_bytes: Number of bytes to format

    Returns:
        Formatted string with appropriate unit
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            if unit == "B":
                return f"{num_bytes} {unit}"
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"


def read_json(path: str, time_it: bool = False):
    if time_it:
        start = time.time()
        file_size = osp.getsize(path)
        if file_size > 100:
            logger.info(
                f"Reading json file of size {format_bytes(file_size)} from {osp.basename(path)}"
            )
    with open(path, "r") as f:
        data = json.load(f)
    if time_it:
        end = time.time()
        logger.info(f"Time taken reading json: {end - start:.2f} seconds")
    return data


def write_json(data: dict, path: str, time_it: bool = False, indent: bool = False):
    if time_it:
        start = time.time()
    with open(path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2 if indent else None))
    if time_it:
        end = time.time()
        logger.info(f"Time taken writing json: {end - start:.2f} seconds")


def read_yaml(path: str, time_it: bool = False):
    if time_it:
        start = time.time()
        file_size = osp.getsize(path)
        if file_size > 100:
            logger.info(
                f"Reading yaml file of size {format_bytes(file_size)} from {osp.basename(path)}"
            )
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if time_it:
        end = time.time()
        logger.info(f"Time taken reading yaml: {end - start:.2f} seconds")
    return data


def convert_dict_of_lists_to_list_of_dicts(
    dict_of_lists: Dict[str, List[Any]],
) -> List[Dict[str, Any]]:
    """
    Convert a dictionary of lists to a list of dictionaries.

    Args:
        dict_of_lists: Dictionary where each value is a list

    Returns:
        List of dictionaries where each dictionary contains one value from each list
    """
    # Get the length of the first list to determine how many dictionaries to create
    if not dict_of_lists:
        return []

    first_key = next(iter(dict_of_lists))
    num_items = len(dict_of_lists[first_key])

    # Verify all lists have the same length
    for key, value in dict_of_lists.items():
        if len(value) != num_items:
            raise ValueError(
                f"All lists must have the same length. Found {len(value)} for key {key} but expected {num_items}"
            )

    # Create list of dictionaries
    return [
        {key: dict_of_lists[key][i] for key in dict_of_lists} for i in range(num_items)
    ]
