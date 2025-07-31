import json
import sys
from collections import OrderedDict
from typing import Dict, List

from loguru import logger

LLM_LEVEL_NAME = "LLM"
PROMPT_LEVEL_NAME = "PROMPT"
CACHE_LEVEL_NAME = "CACHE"
API_LEVEL_NAME = "API"


class Formatter:
    def __init__(self):
        self.padding = 0
        self.fmt = "[<green><b>{time:YYYY-MM-DD hh:mm:ss.SS}</b></green>][<cyan><b>{file}:{line}</b></cyan> - <cyan>{name:}:{function}</cyan>][<level>{level}</level>] {message}\n"

    def format(self, record):
        length = len("{file}:{line} - {name:}:{function}".format(**record))
        self.padding = max(self.padding, length)
        record["extra"]["padding"] = " " * (self.padding - length)
        fmt = ""
        if record["level"].name == LLM_LEVEL_NAME and "message" in record["extra"]:
            if record["extra"]["from_cache"]:
                fmt = "<LG>===================[[<b>Response (cache time={extra[cache_elapsed_time]}  completion tokens={extra[usage][completion_tokens]}  total_tokens={extra[usage][total_tokens]})</b>]]===================</LG>\n{extra[message]}\n"
            else:
                fmt = "<LY>===================[[<b>Response (API time={extra[api_elapsed_time]}  completion tokens={extra[usage][completion_tokens]}  total_tokens={extra[usage][total_tokens]})</b>]]===================</LY>\n{extra[message]}\n"
        elif (
            record["level"].name == PROMPT_LEVEL_NAME and "messages" in record["extra"]
        ):
            for i, message in enumerate(record["extra"]["messages"]):
                fmt += (
                    f"<LC>===================[[<b>{message['role']:}</b>]]===================</LC>\n"
                    f"{message['content']}\n"
                )
        ret_fmt = self.fmt

        ret_fmt = ret_fmt.replace("{serialized_short}", "")
        return ret_fmt + fmt


def serialize(record):
    subset = OrderedDict()
    subset["level"] = record["level"].name
    subset["message"] = record["message"]
    subset["time"] = record["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    subset["file"] = {
        "name": record["file"].name,
        "path": record["file"].path,
        "function": record["function"],
        "line": record["line"],
    }
    subset["extra"] = record["extra"]
    return json.dumps(subset)


def serialize_extras(record):
    return json.dumps(record["extra"])


def patching(record):
    extras = serialize_extras(record)
    record["serialized_short"] = extras[:50] + "..." if len(extras) > 50 else extras
    record["extra"]["serialized"] = serialize(record)

    if record["level"].name == LLM_LEVEL_NAME and "message" in record["extra"]:
        record["extra"]["message"] = record["extra"]["message"]
    elif record["level"].name == PROMPT_LEVEL_NAME and "messages" in record["extra"]:
        for i, message in enumerate(record["extra"]["messages"]):
            record["extra"]["messages"][i]["content"] = message["content"]


def parse_prompt(prompt: List[Dict[str, str]]):
    return "\n\n".join(
        [f"===={m['role'].upper()} ====:\n{m['content']}" for m in prompt]
    )


logger.remove(0)
logger = logger.patch(patching)
logger.level(PROMPT_LEVEL_NAME, no=10, color="<white><bold>", icon="📋")
logger.level(CACHE_LEVEL_NAME, no=10, color="<yellow><bold>", icon="💾")
logger.level(API_LEVEL_NAME, no=10, color="<red><bold>", icon="🛜")
logger.level(LLM_LEVEL_NAME, no=10, color="<lm><bold>", icon="🤖")


formatter = Formatter()
logger.add(sys.stdout, format=formatter.format)
