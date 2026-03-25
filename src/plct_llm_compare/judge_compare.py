"""Judge-compare command implementation."""

import json
from pathlib import Path
from typing import Any

import click
from markdown_it import MarkdownIt
from pydantic import TypeAdapter
from plct_server.ai.client import AiClientFactory
from plct_server.ai.model_conf import ModelProvider, MODEL_CONFIGS_LIST

from .config import OPENAI_API_KEY, VLLM_URL
from .models import (
    JudgeCompareStructuredResult,
    TestCase,
    TestCaseJudgeCompareResult,
)


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

Return a structured evaluation.

Work in this order:
1. Identify the most important comparative rationale points.
2. Give concrete improvement suggestions for each answer.
3. Assign scores for both answers.
4. Decide the winner last, after reviewing the full comparison.

Do not reveal hidden chain-of-thought. Keep rationale concise and evidence-based.
"""


def _make_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively prepare schema for OpenAI strict mode:
    - Add additionalProperties:false to all object schemas.
    - Strip sibling keywords from $ref nodes (strict mode forbids them).
    """
    schema = dict(schema)
    if "$ref" in schema:
        # $ref must stand alone — drop all sibling keywords
        return {"$ref": schema["$ref"]}
    if schema.get("type") == "object":
        schema["additionalProperties"] = False
    for key in ("properties", "definitions", "$defs"):
        if key in schema:
            schema[key] = {k: _make_strict_schema(v) for k, v in schema[key].items()}
    if "items" in schema:
        schema["items"] = _make_strict_schema(schema["items"])
    return schema


def _judge_response_format() -> dict[str, Any]:
    """Build the JSON schema response format for the judge output."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "judge_compare_result",
            "strict": True,
            "schema": _make_strict_schema(JudgeCompareStructuredResult.model_json_schema()),
        },
    }


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


async def _generate_structured_judgement(
    *,
    client_factory: AiClientFactory,
    model_config,
    messages: list[dict[str, str]],
    temperature: float,
) -> JudgeCompareStructuredResult:
    """Generate a structured judge result using JSON-schema output."""
    client = client_factory.get_client(model_config=model_config)
    completion = await client.chat.completions.create(
        model=model_config.name,
        messages=messages,
        max_completion_tokens=4000,
        temperature=temperature,
        response_format=_judge_response_format(),
    )
    content = completion.choices[0].message.content or ""
    return JudgeCompareStructuredResult.model_validate_json(content)


def _render_judge_result(judge_result: JudgeCompareStructuredResult) -> str:
    """Render the structured judge result to markdown in the preferred order."""
    lines = ["## Rationale"]
    lines.extend(f"- {item}" for item in judge_result.rationale)

    lines.extend(["", "## Improvement Suggestions", "### Answer A"])
    lines.extend(f"- {item}" for item in judge_result.improvement_suggestions_a)

    lines.extend(["", "### Answer B"])
    lines.extend(f"- {item}" for item in judge_result.improvement_suggestions_b)

    lines.extend([
        "",
        "## Scores",
        (
            f"- Correctness: A={judge_result.scores_a.correctness}, "
            f"B={judge_result.scores_b.correctness}"
        ),
        (
            f"- Relevance: A={judge_result.scores_a.relevance}, "
            f"B={judge_result.scores_b.relevance}"
        ),
        (
            f"- Clarity: A={judge_result.scores_a.clarity}, "
            f"B={judge_result.scores_b.clarity}"
        ),
        (
            f"- Educational usefulness: A={judge_result.scores_a.educational_usefulness}, "
            f"B={judge_result.scores_b.educational_usefulness}"
        ),
        "",
        "## Winner",
        f"- {judge_result.winner}",
    ])
    return "\n".join(lines)


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
            judge_result = await _generate_structured_judgement(
                client_factory=client_factory,
                model_config=judge_model_config,
                messages=judge_messages,
                temperature=0.2,
            )
            judge_response = _render_judge_result(judge_result)

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
                judge_result=judge_result,
                judge_response=judge_response,
            )
            meta_file = cases_path.parent / f"{base_name}.json"
            meta_file.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")

            click.echo(f"    Saved judge comparison to {html_file}")
