"""Data model for tools and their commands."""

import abc
import re
from typing import Any, Dict, List, Literal, Optional
import warnings
import langfun as lf
import pydantic
import yaml


class CodeState(pydantic.BaseModel):
    memory: Dict[str, Any] = pydantic.Field(..., exclude=True)
    observation: str | None = None
    is_done: Optional[bool] = False
    answer: Optional[str] = None


class ToolArgument(pydantic.BaseModel):
    name: str
    arg_type: Literal["str", "int", "float", "bool"]
    description: str
    required: bool


class ToolParsed(pydantic.BaseModel):
    command: str
    kwargs: Optional[Dict[str, str]] = None


class ToolCommand(pydantic.BaseModel):
    """Represents a command that can be executed by a tool."""

    docstring: str
    arguments: List[ToolArgument]
    command: str = pydantic.Field(..., exclude=True)

    @pydantic.computed_field
    @property
    def demonstration(self) -> str:
        kwargs = {arg.name: "<arg value>" for arg in self.arguments}
        dic = {
            "command": self.command,
        }
        if kwargs:
            dic["kwargs"] = kwargs
        return f"```\n{yaml.dump(dic)}```"


class Tool(abc.ABC):
    """A tool is a command that can be executed by the agent."""

    command: ToolCommand

    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def execute(self, parsed: ToolParsed, state: CodeState) -> CodeState:
        """Execute the tool and return the new state."""
        pass


def parse_command(llm_resp: str) -> ToolParsed:
    """Parses a tool command from an LLM response."""
    pattern: re.Pattern[str] = re.compile(r"```yaml\n(?P<yaml>.*?)(?=```)", re.DOTALL)
    try:
        # Greedy search for 1st yaml candidate.
        match = re.search(pattern, llm_resp.strip())
        if match:
            yaml_str = match.group("yaml")
        else:
            # If no backticks were present, try to parse the entire output as yaml.
            yaml_str = llm_resp
        json_object = yaml.safe_load(yaml_str)
        return ToolParsed(**json_object)
    except (yaml.YAMLError, pydantic.ValidationError) as e:
        msg = (
            "We failed to parse a tool command from your response. Here is the"
            f" error:\n{e}.\nPlease ensure the YAML is properly formatted."
        )
        raise ValueError(msg) from e


class Tools:
    """A collection of tools that can be executed by the agent."""

    def __init__(self, tools: List[Tool]):
        self.tools: Dict[str, Tool] = {tool.command.command: tool for tool in tools}

    def get_tool_docs(self) -> str:
        dic = {
            tool.command.command: tool.command.model_dump()
            for tool in self.tools.values()
        }
        return (
            yaml.dump(dic, sort_keys=False)
            + '\nPlease enclose all command kwargs values in ""'
        )

    def execute_tool(self, tool: ToolParsed, state: CodeState) -> CodeState:
        command = tool.command
        if command not in self.tools:
            supported_commands = ", ".join(self.tools.keys())
            raise ValueError(
                f"Invalid command: {tool.command}\nSupported commands are"
                f" {supported_commands}"
            )
        return self.tools[command].execute(tool, state)


class Done(Tool):
    """A tool to indicate that the task is done and provide the final answer."""

    command: ToolCommand = ToolCommand(
        command="done",
        docstring=(
            "Indicate that we arrived at the final answer and provide the answer."
            " Use this command only when you have arrived at the final answer."
        ),
        arguments=[
            ToolArgument(
                name="answer",
                description=(
                    "The final answer to the question. "
                    "Do not apply any formatting, bolding, or markup. "
                    "If the question asks for a list of values, then "
                    "the answer should be a comma-separated list of values "
                    "(e.g., '42, 43, 44')"
                ),
                required=True,
                arg_type="str",
            ),
        ],
    )

    def execute(self, parsed: ToolParsed, state: CodeState) -> CodeState:
        new_state = CodeState(
            memory=state.memory,
            is_done=True,
            answer=parsed.kwargs["answer"],
        )
        return new_state


class PythonShell(Tool):
    """A tool to execute python code in a python shell."""

    command: ToolCommand = ToolCommand(
        command="python",
        docstring=(
            "Execute Python code within a persistent Python shell. The shell "
            "maintains state across executions, so variables and imports from "
            "previous runs remain available. When first using this command, the "
            "data table is provided as a global variable named `df`, and "
            "`pandas` has already been imported as `pd`."
        ),
        arguments=[
            ToolArgument(
                name="code",
                description="The Python code to execute.",
                required=True,
                arg_type="str",
            ),
        ],
    )

    def execute(self, parsed: ToolParsed, state: CodeState) -> CodeState:
        new_state = state
        try:
            warnings.simplefilter(action="ignore", category=Warning)
            res = lf.coding.python.run(
                parsed.kwargs["code"],
                sandbox=False,
                outputs_intermediate=True,
                global_vars=state.memory,
            )
            observation = res.get("__result__", None)
            if observation is None:
                observation = res.get("__stdout__", None)
            for k, v in res.items():
                if not k.startswith("__"):
                    new_state.memory[k] = v
        except Exception as e:
            observation = str(e)
        new_state.observation = observation
        return new_state
