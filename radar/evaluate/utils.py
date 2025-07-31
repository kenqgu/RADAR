import re
from typing import Any, Dict

import langfun as lf


def get_llm_identifier(lm: lf.LanguageModel) -> str:
    options_dict: Dict[str, Any] = lm.sampling_options.to_json()
    config = {
        k: v
        for k, v in options_dict.items()
        if v is not None and not (k.startswith("_") or k == "max_tokens")
    }
    parts = [f"{k}={str(v).replace(' ', '')}" for k, v in sorted(config.items())]
    name = "_".join(parts)
    model_id = re.sub(r"[:/?=]", "_", lm.model_id)
    return f"{model_id}_{name}"


def get_llm_cache_name(lm: lf.LanguageModel) -> str:
    return f"{get_llm_identifier(lm)}.cache"
