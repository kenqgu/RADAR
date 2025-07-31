from functools import cached_property
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel
from typing_extensions import TypedDict

from radar.data import datamodel
from radar.evaluate import measure


class LLMMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


LLMMessages = List[LLMMessage]


class TaskInstanceRunResult(BaseModel):
    baseline: str
    task_instance_id: str
    llm_extracted_answer: str
    ground_truth: Any
    is_correct: bool
    llm_messages: LLMMessages
    llm_messages_metadata: Optional[List[Dict[str, Any]]] = None
    task: Optional[datamodel.TaskInstance] = None

    @cached_property
    def llm_last_message(self) -> str:
        if len(self.llm_messages) == 0:
            raise ValueError("No LLM messages found.")
        return self.llm_messages[-1]["content"]

    def to_dict(self) -> Dict[str, Any]:
        ret = {
            "task_instance_id": self.task_instance_id,
            "baseline": self.baseline,
            "ground_truth": self.ground_truth,
            "llm_extracted_answer": self.llm_extracted_answer,
            "is_correct": self.is_correct,
        }
        if self.task is not None:
            ret["task_id"] = self.task.task_id
            ret["artifact_type"] = self.task.artifact_type
            ret["num_cols"] = self.task.num_cols
            ret["token_bucket"] = self.task.base_data_token_bucket
        return ret
