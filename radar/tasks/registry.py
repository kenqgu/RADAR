from typing import Any, Callable, Dict

import pandas as pd

from radar.tasks import datamodel

ANSWER_FUNCTION_REGISTRY: Dict[str, Callable[[pd.DataFrame], Any]] = {}
MISSING_DATA_FUNCTION_REGISTRY: Dict[
    str, Callable[[pd.DataFrame], "datamodel.PerturbationReturn"]
] = {}
INCONSISTENT_FORMATTING_FUNCTION_REGISTRY: Dict[
    str, Callable[[pd.DataFrame], "datamodel.PerturbationReturn"]
] = {}
INCONSISTENT_LOGIC_FUNCTION_REGISTRY: Dict[
    str, Callable[[pd.DataFrame], "datamodel.PerturbationReturn"]
] = {}

BAD_VALUES_FUNCTION_REGISTRY: Dict[
    str, Callable[[pd.DataFrame], "datamodel.PerturbationReturn"]
] = {}

OUTLIER_FUNCTION_REGISTRY: Dict[
    str, Callable[[pd.DataFrame], "datamodel.PerturbationReturn"]
] = {}


def register_answer_function(name: str):
    def wrapper(func: Callable[[pd.DataFrame], Any]):
        ANSWER_FUNCTION_REGISTRY[name] = func
        return func

    return wrapper


def get_registered_answer_function(name: str) -> Callable[[pd.DataFrame], Any]:
    return ANSWER_FUNCTION_REGISTRY[name]


def register_missing_data_function(name: str):
    def wrapper(func: Callable[[pd.DataFrame], "datamodel.PerturbationReturn"]):
        MISSING_DATA_FUNCTION_REGISTRY[name] = func
        return func

    return wrapper


def get_registered_missing_data_function(
    name: str,
) -> Callable[[pd.DataFrame], "datamodel.PerturbationReturn"]:
    return MISSING_DATA_FUNCTION_REGISTRY[name]


def register_inconsistent_formatting_function(name: str):
    def wrapper(func: Callable[[pd.DataFrame], "datamodel.PerturbationReturn"]):
        INCONSISTENT_FORMATTING_FUNCTION_REGISTRY[name] = func
        return func

    return wrapper


def get_registered_inconsistent_formatting_function(
    name: str,
) -> Callable[[pd.DataFrame], "datamodel.PerturbationReturn"]:
    return INCONSISTENT_FORMATTING_FUNCTION_REGISTRY[name]


def register_inconsistent_logic_function(name: str):
    def wrapper(func: Callable[[pd.DataFrame], "datamodel.PerturbationReturn"]):
        INCONSISTENT_LOGIC_FUNCTION_REGISTRY[name] = func
        return func

    return wrapper


def get_registered_inconsistent_logic_function(
    name: str,
) -> Callable[[pd.DataFrame], "datamodel.PerturbationReturn"]:
    return INCONSISTENT_LOGIC_FUNCTION_REGISTRY[name]


def register_bad_values_function(name: str):
    def wrapper(func: Callable[[pd.DataFrame], "datamodel.PerturbationReturn"]):
        BAD_VALUES_FUNCTION_REGISTRY[name] = func
        return func

    return wrapper


def get_registered_bad_values_function(
    name: str,
) -> Callable[[pd.DataFrame], "datamodel.PerturbationReturn"]:
    return BAD_VALUES_FUNCTION_REGISTRY[name]


def register_outliers_function(name: str):
    def wrapper(func: Callable[[pd.DataFrame], "datamodel.PerturbationReturn"]):
        OUTLIER_FUNCTION_REGISTRY[name] = func
        return func

    return wrapper


def get_registered_outlier_function(
    name: str,
) -> Callable[[pd.DataFrame], "datamodel.PerturbationReturn"]:
    return OUTLIER_FUNCTION_REGISTRY[name]
