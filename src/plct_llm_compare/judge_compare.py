"""Judge-compare command implementation."""

import json
from pathlib import Path

import click
from markdown_it import MarkdownIt
from pydantic import TypeAdapter
from plct_server.ai.client import AiClientFactory
from plct_server.ai.model_conf import ModelProvider, MODEL_CONFIGS_LIST

from .config import OPENAI_API_KEY, VLLM_URL
from .models import TestCase, TestCaseJudgeCompareResult


def _get_model_config(model_name: str):
    """Look up a model config by name without initialising AiEngine."""
    for cfg in MODEL_CONFIGS_LIST:
        if cfg.name == model_name:
            return cfg
    raise ValueError(f"Model '{model_name}' not found in MODEL_CONFIGS_LIST")


JUDGE_SYSTEM_PROMPT = (
    "You are an impartial evaluator. Compare two assistant answers to the same user prompt. "
    "Judge correctness, relevance, clarity, and educational usefulness. "
    "Do not prefer verbosity by default."
)


JUDGE_USER_PROMPT_TEMPLATE = """You are given one user prompt and two candidate answers.

User Prompt:
{prompt}

Answer A ({model_a}):
{answer_a}

Answer B ({model_b}):
{answer_b}

Return your evaluation in Markdown with exactly these sections:

## Winner
- Pick one: A, B, or Tie

## Rationale
- 3-6 bullet points comparing strengths and weaknesses

## Scores
- Correctness: A=<0-10>, B=<0-10>
- Relevance: A=<0-10>, B=<0-10>
- Clarity: A=<0-10>, B=<0-10>
- Educational usefulness: A=<0-10>, B=<0-10>

## Improvement Suggestions
- 1-3 concrete suggestions for Answer A
- 1-3 concrete suggestions for Answer B
"""


async def _generate_answer(
    *,
    client_factory: AiClientFactory,
    model_config,
    messages: list[dict[str, str]],
    temperature: float,
) -> str:
    """Generate a single completion for a configured model."""
    client = client_factory.get_client(model_config=model_config)
    completion = await client.chat.completions.create(
        model=model_config.name,
        messages=messages,
        max_completion_tokens=8000,
        temperature=temperature,
    )
    return completion.choices[0].message.content or ""


async def do_judge_compare(
    cases_fname: str,
    model_a: str,
    model_b: str,
    judge_model: str,
) -> None:
    """Run two models on each case, then judge with a third model."""
    cases_path = Path(cases_fname)
    with cases_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    test_cases = TypeAdapter(list[TestCase]).validate_python(data)

    client_factory = AiClientFactory(
        default_provider=ModelProvider.OPENAI,
        openai_api_key=OPENAI_API_KEY,
        azure_api_key=None,
        vllm_api_key="EMPTY",
        vllm_url=VLLM_URL,
    )

    model_a_config = _get_model_config(model_a)
    model_b_config = _get_model_config(model_b)
    judge_model_config = _get_model_config(judge_model)

    click.echo(f"Loaded {len(test_cases)} test cases:")
    md = MarkdownIt()

    for tc in test_cases:
        base_messages = [
            {"role": "system", "content": tc.system_message or ""},
            {"role": "user", "content": tc.prompt},
        ]

        for take in [1]:
            click.echo(f"  - {tc.course_key}/{tc.activity_key} (take {take}):")

            answer_a = await _generate_answer(
                client_factory=client_factory,
                model_config=model_a_config,
                messages=base_messages,
                temperature=0.5,
            )
            answer_b = await _generate_answer(
                client_factory=client_factory,
                model_config=model_b_config,
                messages=base_messages,
                temperature=0.5,
            )

            judge_user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
                prompt=tc.prompt,
                model_a=model_a,
                answer_a=answer_a,
                model_b=model_b,
                answer_b=answer_b,
            )
            judge_messages = [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": judge_user_prompt},
            ]
            judge_response = await _generate_answer(
                client_factory=client_factory,
                model_config=judge_model_config,
                messages=judge_messages,
                temperature=0.2,
            )

            judge_html = md.render(judge_response)
            model_a_safe = model_a.replace("/", "--")
            model_b_safe = model_b.replace("/", "--")
            judge_model_safe = judge_model.replace("/", "--")
            base_name = (
                f"{tc.case_key}_{take}_{model_a_safe}_vs_{model_b_safe}_judge_{judge_model_safe}"
            )

            html_file = cases_path.parent / f"{base_name}.html"
            html_file.write_text(judge_html, encoding="utf-8")

            metadata = TestCaseJudgeCompareResult(
                case_key=tc.case_key,
                activity_url=tc.activity_url,
                activity_desc=tc.activity_desc,
                prompt=tc.prompt,
                model_a=model_a,
                model_b=model_b,
                judge_model=judge_model,
                take=take,
                answer_a=answer_a,
                answer_b=answer_b,
                judge_response=judge_response,
            )
            meta_file = cases_path.parent / f"{base_name}.json"
            meta_file.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")

            click.echo(f"    Saved judge comparison to {html_file}")
