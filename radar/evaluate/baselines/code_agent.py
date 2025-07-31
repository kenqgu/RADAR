from typing import Any, Callable, Dict, List, Tuple

import pandas as pd

from radar.data import datamodel
from radar.evaluate import datamodel as eval_datamodel
from radar.evaluate import measure
from radar.evaluate.baselines import tools

PREAMBLE = """
  SETTING: You are an expert-level data scientist. Your job is to answer a data driven question \
  in rigorous manner given a data table.
  In your analysis:
  * Carefully address
  1) missing data: empty or null entries simulating incomplete information
  2) bad values: clearly erroneous or placeholder entries (e.g., `-1`, `9999`, `TEST`, `#REF!` etc.)
  3) outliers: implausible extreme values that distort analysis (e.g., 220 breathing rate per minute)
  4) inconsistent formatting: variations in representing the same value (e.g., `22 lbs`, `22 pounds`, `weight = 22`)
  5) inconsistent logic: cross-field contradictions violating common-sense logic (e.g., end time before start time)
  * Attempt to safely recover or correct flawed data when reasonable based on the existing data. If data is irrecoverable or suspect, discard the row.
  You will be working within a Python shell and can use the following commands to answer the question.
  
  AVAILABLE COMMANDS:
  {command_docs}
  RESPONSE_FORMAT:
  Each response must include:
  1. A DISCUSSION field — where you will methodically break down the reasoning process, illustrating how you arrive at conclusions and decide what to do next.
  2. A command field — proprtly formatted YAML within triple backticks and following the structure from COMMANDS.
  Important rules:
  - Always include exactly one DISCUSSION and one command block.
  - Ensure the command block is properly formatted YAML with proper indents and newlines (see the example below).
  For example, given a question asking for the average income. You might respond:
  DISCUSSION
  Let's think step by step. \
  We need to first find the average income of the population. \
  We can do this by summing up the income column and dividing by the number of rows.
  ```yaml
  command: "python"
  kwargs:
    code: |-
      income_avg = df['income'].sum() / len(df)
      income_avg
  ```
"""

TASK_PROMPT = """
  Begin!
  Data table (stored in a pandas dataframe named `df`):
  {table}
  All cells in the `df` are `object` data type, regardless of their appearance.
  Question:
  {question}
"""

STATE_PROMPT = """
  Observation:
  {observation}
"""


class CodegenAgent:
    """Agent for code generation."""

    def run(
        self,
        question: str,
        table: pd.DataFrame,
        llm_func: Callable[
            [eval_datamodel.LLMMessages], Tuple[str, Dict[str, Any] | None]
        ],
        max_steps: int = 5,
    ) -> Tuple[List[Dict[str, str]], Any, List[Dict[str, Any]] | None]:
        """Runs the code generation agent.
        Args:
            question: The question to answer.
            table: The data table.
            llm_func: A function that calls a language model with the prompt messages and returns a Tuple of (response, metadata dict).
            max_steps: The maximum number of steps to run.
        Returns:
            A Tuple of (list of messages, answer, list of messages metadata).
        """
        agent_tools = tools.Tools([tools.PythonShell(), tools.Done()])
        df_str = table.astype(str)
        initial_state = tools.CodeState(memory={"df": df_str, **globals()})
        messages = [
            {
                "role": "system",
                "content": PREAMBLE.format(command_docs=agent_tools.get_tool_docs()),
            },
            {
                "role": "user",
                "content": TASK_PROMPT.format(
                    table=table.to_csv(index=False), question=question
                ),
            },
        ]

        states = [initial_state]
        messages_metadata = []
        for _ in range(max_steps):
            llm_resp, resp_metadata = llm_func(messages)
            messages.append({"role": "assistant", "content": llm_resp})
            if resp_metadata:
                messages_metadata.append(resp_metadata)
            try:
                tool_parsed = tools.parse_command(llm_resp)
            except Exception as e:
                msg = (
                    f"Failed to parse command from response with error: {e}\nPlease"
                    " ensure the YAML is properly formatted."
                )
                messages.append(
                    {
                        "role": "user",
                        "content": msg,
                    }
                )
                continue
            next_state = agent_tools.execute_tool(tool_parsed, states[-1])
            states.append(next_state)
            if next_state.is_done:
                return messages, next_state.answer, messages_metadata
            messages.append(
                {
                    "role": "user",
                    "content": STATE_PROMPT.format(observation=next_state.observation),
                }
            )
        states[-1].answer = f"Maximum {max_steps} steps reached."
        return messages[:-1], states[-1].answer, messages_metadata


def run_code_agent(
    task: datamodel.TaskInstance, llm_call: Callable[[List[Dict[str, str]]], str]
) -> Dict[str, Any]:
    """
    Runs the code generation agent.
    Args:
        task: The task to answer.
        llm_call: A function that takes a list of messages and returns a string.
    Returns:
        A TaskInstanceRunResult object.
    """
    prompt_info = task.get_prompt_info_codegen_agent()
    try:
        llm_messages, answer, messages_metadata = CodegenAgent().run(
            question=prompt_info["question"],
            table=prompt_info["table"],
            llm_func=llm_call,
        )
    except Exception as e:
        raise e

    ret = eval_datamodel.TaskInstanceRunResult(
        baseline="code_agent",
        task_instance_id=task.task_instance_id,
        llm_extracted_answer=answer,
        ground_truth=task.answer,
        is_correct=measure.match_answer(answer, task.answer),
        llm_messages=llm_messages,
        llm_messages_metadata=messages_metadata if messages_metadata else None,
        task=task,
    )
    return ret
