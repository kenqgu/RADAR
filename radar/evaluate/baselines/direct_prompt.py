from typing import Any, Callable, Dict, Tuple

from radar.data import datamodel
from radar.evaluate import datamodel as eval_datamodel
from radar.evaluate import measure

PREAMBLE = """
  You are an expert-level data scientist. Your job is to answer a data analysis question in rigorous manner given a data table.
  In your analysis:
  * Carefully address
    1) missing data: empty or null entries simulating incomplete information
    2) bad values: clearly erroneous or placeholder entries (e.g., `-1`, `9999`, `TEST`, `#REF!` etc.)
    3) outliers: implausible extreme values that distort analysis (e.g., 220 breathing rate per minute)
    4) inconsistent formatting: variations in representing the same value (e.g., `22 lbs`, `22 pounds`, `weight = 22`)
    5) inconsistent logic: cross-field contradictions violating common-sense logic (e.g., end time before start time)
  * Attempt to safely recover or correct flawed data when reasonable based on the existing data. If data is irrecoverable or suspect, discard the row.
  * Do NOT write or execute any code. Focus purely on logical reasoning and analytical judgment.
  You must conclude with your most reasonable answer.

  When you provide the final answer, please use the prefix "The answer is:" \
  without any modification, and provide the answer directly, with no formatting, no bolding, and \
  no markup. For instance: "The answer is: 42" or "The answer is: yes". If the question asks \
  for a list of values, then the answer should be a comma-separated list of values, \
  without any formatting, no bolding, and no markup. For instance: "The answer is: 42, 43, 44" or "The answer is: yes, no".
"""

TASK_PROMPT = """
  Data:
  {table}
  Based on the given table, answer the following question:
  {question}
"""


def run_direct_prompt(
    task: datamodel.TaskInstance,
    llm_call: Callable[[eval_datamodel.LLMMessages], Tuple[str, Dict[str, Any] | None]],
    num_retry_validate: int = 3,
) -> eval_datamodel.TaskInstanceRunResult:
    """
    Runs the direct prompt baseline.
    Args:
        task: The task to answer.
        llm_func: A function that calls a language model with the prompt messages and returns a Tuple of (response, metadata dict).
    Returns:
        A TaskInstanceRunResult object.
    """
    prompt_info = task.get_prompt_info()
    llm_messages = [
        {
            "role": "system",
            "content": PREAMBLE,
        },
        {
            "role": "user",
            "content": TASK_PROMPT.format(**prompt_info),
        },
    ]
    llm_messages_metadata = []
    answer = None
    for _ in range(num_retry_validate):
        try:
            resp, llm_metadata = llm_call(llm_messages)
            llm_messages.append({"role": "assistant", "content": resp})
            llm_messages_metadata.append(llm_metadata)
            answer = measure.extract_value_from_answer(resp)
            break
        except ValueError as e:
            next_message = str(e)
            llm_messages.append({"role": "user", "content": next_message})
    if answer is None:
        raise ValueError(f"Failed getting a valid LLM response: {next_message}")

    ret = eval_datamodel.TaskInstanceRunResult(
        baseline="direct_prompt",
        task_instance_id=task.task_instance_id,
        llm_extracted_answer=answer,
        ground_truth=task.answer,
        is_correct=measure.match_answer(answer, task.answer),
        llm_messages=llm_messages,
        llm_messages_metadata=llm_messages_metadata,
        task=task,
    )
    return ret
