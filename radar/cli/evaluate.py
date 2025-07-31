from enum import Enum

import langfun as lf
import typer

from radar.evaluate import run

evaluate_lm = typer.Typer(name="evaluate-lm")


class Baseline(str, Enum):
    direct_prompt = "direct_prompt"
    code_agent = "code_agent"


class ReasoningEffort(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


@evaluate_lm.command()
def evaluate(
    lm_model_id: str = typer.Argument(
        help="The LM model id to use for evaluation. Run the `langfun.LanguageModel.dir()` method to see the available model ids. For gemini models we use the google genai api (i.e., https://ai.google.dev/gemini-api/docs) with prefix 'google_genai://'"
    ),
    split: str = typer.Argument(
        help="The split to use for evaluation. It can be 'tasks' or 'sizes' or a custom split name.",
    ),
    local_split_dir: str = typer.Option(
        default=None,
        help="The local directory to load the tasks from if given a custom split name.",
    ),
    baseline: Baseline = typer.Option(
        default=Baseline.direct_prompt,
        help="The baseline to use for evaluation.",
        show_choices=True,
    ),
    api_key: str = typer.Option(
        default=None,
        help="The api key to use for evaluation. If not provided, the API key will be loaded from the corresponding environment variable (i.e., OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, and GOOGLE_API_KEY).",
    ),
    max_attempts: int = typer.Option(
        default=2, help="The maximum number of attempts to request the LM."
    ),
    run_parallel: bool = typer.Option(
        default=True, help="Whether to run the LM calls in parallel."
    ),
    num_workers: int = typer.Option(
        default=10, help="The number of parallel workers to use for LM calls."
    ),
    timeout: int = typer.Option(
        default=1000, help="Timeout in seconds for each request attempt."
    ),
    reasoning_effort: ReasoningEffort = typer.Option(
        default=None,
        help="The reasoning effort to use for OpenAI reasoning models (e.g., o4-mini).",
        show_choices=True,
    ),
    max_thinking_tokens: int = typer.Option(
        default=None,
        help="The maximum number of tokens to use for Gemini thikning models (e.g., google_genai://gemini-2.5-flash-preview-05-20).",
    ),
    cache_llm_dir: str = typer.Option(
        default=None,
        help="The directory to cache the LM calls. If not provided, will use a default '.lm_cache' directory in the current working directory.",
    ),
    cache_llm_calls: bool = typer.Option(
        default=True, help="Whether to cache the LM calls."
    ),
    output_dir: str = typer.Option(
        default=None,
        help="The directory to save the evaluation results. If not provided, will use a default 'eval_results' directory in the current working directory.",
    ),
    debug: bool = typer.Option(
        default=False,
        help="Whether to run in debug mode. Runs on a very small subset of the tasks.",
    ),
):
    run.run_eval(
        llm=lm_model_id,
        split=split,
        local_split_dir=local_split_dir,
        baseline=baseline.value,
        api_key=api_key,
        max_attempts=max_attempts,
        run_parallel=run_parallel,
        num_workers=num_workers,
        timeout=timeout,
        reasoning_effort=reasoning_effort.value if reasoning_effort else None,
        max_thinking_tokens=max_thinking_tokens,
        cache_llm_dir=cache_llm_dir,
        cache_llm_calls=cache_llm_calls,
        output_dir=output_dir,
        debug=debug,
    )


def main():
    evaluate_lm()


if __name__ == "__main__":
    evaluate_lm()
