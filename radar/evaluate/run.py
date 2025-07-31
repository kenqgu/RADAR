import contextlib
import json
import os
import os.path as osp
import random
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import langfun as lf
import tqdm
from dotenv import load_dotenv

from radar import utils as radar_utils
from radar.data import load_task_instances_hf
from radar.evaluate import datamodel, results, utils
from radar.evaluate.baselines import code_agent, direct_prompt
from radar.logger import logger

load_dotenv()


def get_llm(
    model_id: str,
    api_key: Optional[str] = None,
    max_attempts: int = 2,
    timeout: int = 1000,
    reasoning_effort: (
        Literal["low", "medium", "high"] | None
    ) = None,  # Used for OpenAI models reasoning models
    max_thinking_tokens: (
        int | None
    ) = None,  # Used for Gemini 2.5 models) -> lf.LanguageModel:
):
    temperature = (
        1 if model_id.startswith("o") else 0
    )  # OpenAI's o series models are temperature = 1
    llm = lf.LanguageModel.get(
        model_id,
        temperature=temperature,
        max_attempts=max_attempts,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
        max_thinking_tokens=max_thinking_tokens,
        api_key=api_key,
    )
    return llm


def run_eval(
    llm: Union[str, lf.LanguageModel],
    split: Literal["sizes", "tasks"] | str,
    local_split_dir: Optional[str] = None,
    baseline: Literal["code_agent", "direct_prompt"] = "direct_prompt",
    api_key: Optional[str] = None,
    max_attempts: int = 2,
    timeout: int = 1000,
    reasoning_effort: (
        Literal["low", "medium", "high"] | None
    ) = None,  # Used for OpenAI models reasoning models
    max_thinking_tokens: int | None = None,  # Used for Gemini 2.5 models
    cache_llm_dir: Optional[str] = None,
    cache_llm_calls: bool = True,
    run_parallel: bool = True,
    num_workers: int = 10,
    output_dir: Optional[str] = None,
    debug: bool = False,
):
    if isinstance(llm, str):
        llm = get_llm(
            llm,
            api_key,
            max_attempts,
            timeout,
            reasoning_effort,
            max_thinking_tokens,
        )

    run_params = {
        "baseline": baseline,
        "split": split,
        "llm": utils.get_llm_identifier(llm),
        "debug": debug,
    }
    # Save input parameters for reproducibility
    if output_dir is None:
        save_dir = f"{utils.get_llm_identifier(llm)}_baseline={baseline}_split={split}_debug={debug}"
        output_dir = osp.join("eval_outputs", save_dir)
    run_params["output_dir"] = output_dir
    logger.info(
        f"Running evaluation with parameters:\n{json.dumps(run_params, indent=2)}"
    )

    if split == "sizes":
        logger.info("Loading radar-sizes split")
        tasks, _ = load_task_instances_hf(split="sizes")
    elif split == "tasks":
        logger.info("Loading radar-tasks split")
        tasks, _ = load_task_instances_hf(split="tasks")
    else:
        raise ValueError(f"Unknown split: {split}")
    if debug:
        logger.info("Running in debug mode.")
        tasks = random.sample(tasks, 5)

    logger.info(f"Evaluating {len(tasks)} tasks on split {split}.")

    if cache_llm_calls:
        cache_llm_dir = cache_llm_dir or ".lm_cache"
        cache_file = osp.join(cache_llm_dir, utils.get_llm_cache_name(llm))
        logger.info(f"Reading and writing llm cached calls to {cache_file}")
        context_manager = lf.lm_cache(cache_file)
    else:
        context_manager = contextlib.nullcontext()

    def run_llm(prompt_messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        lf_messages = [
            lf.Message.from_value(m, format="openai") for m in prompt_messages
        ]
        response: lf.Message = lf.query(lf_messages, lm=llm, returns_message=True)
        return response.text, response.sym_jsonify()["metadata"]

    run_func_inputs = [
        {
            "task": task,
            "llm_call": run_llm,
        }
        for task in tasks
    ]

    if baseline == "direct_prompt":
        run_func = lambda x: direct_prompt.run_direct_prompt(**x)
    else:
        llm.sampling_options.stop = ["```\n```yaml", "\nDISCUSSION"]
        run_func = lambda x: code_agent.run_code_agent(**x)

    all_results: List[datamodel.TaskInstanceRunResult | str] = []
    with context_manager:
        if run_parallel:
            logger.info(f"Running in parallel with {num_workers} workers")
            for _, output, error in lf.concurrent_map(
                run_func,
                run_func_inputs,
                show_progress=True,
                ordered=True,
                max_workers=num_workers,
            ):
                if error is not None:
                    logger.error(f"Error running llm for {baseline}: {error}")
                    all_results.append(str(error))
                else:
                    all_results.append(output)
        else:
            for inp in tqdm.tqdm(run_func_inputs, desc="Running in serial"):
                output = run_func(inp)
                all_results.append(output)
    save_eval_results(all_results, run_params, output_dir)


def save_eval_results(
    all_results: List[datamodel.TaskInstanceRunResult | str],
    run_params: Dict[str, Any],
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Saving results to {output_dir}")

    df_results_combined, df_aggregate_metrics = results.process_results(all_results)
    df_results_combined.to_csv(osp.join(output_dir, f"results.csv"))
    df_aggregate_metrics.to_csv(osp.join(output_dir, f"results_summary.csv"))
    radar_utils.write_json(
        run_params, osp.join(output_dir, "run_params.json"), indent=True
    )

    os.makedirs(osp.join(output_dir, "individual_results"), exist_ok=True)
    for result in all_results:
        if isinstance(result, datamodel.TaskInstanceRunResult):
            radar_utils.write_json(
                result.model_dump(mode="json"),
                osp.join(
                    output_dir, "individual_results", f"{result.task_instance_id}.json"
                ),
                indent=True,
            )
    logger.info(f"Saved {len(all_results)} results to {output_dir}")


if __name__ == "__main__":
    run_eval(
        llm="google_genai://gemini-2.0-flash",
        split="tasks",
        baseline="direct_prompt",
        debug=True,
        run_parallel=True,
    )
