"""Prepare command implementation."""

import json
from pathlib import Path

import click
import yaml
from pydantic import TypeAdapter

from plct_server.ai.engine import AiEngine
from plct_server.ai.model_conf import ModelProvider
from plct_server.ai.client import AiClientFactory

from .config import OPENAI_API_KEY, PLCT_AI_CTX_URL
from .models import TestCase


async def do_prepare(cases_fname: str) -> None:
    """Prepare resources for LLM comparison."""
    cases_path = Path(cases_fname)
    with cases_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    test_cases = TypeAdapter(list[TestCase]).validate_python(data)

    client_factory = AiClientFactory(
        default_provider = ModelProvider.OPENAI,
        openai_api_key = OPENAI_API_KEY,
        azure_api_key = None,
        vllm_api_key= None
    )

    ai_engine = AiEngine(ai_ctx_url=PLCT_AI_CTX_URL, client_factory=client_factory)

    click.echo(f"Loaded {len(test_cases)} test cases:")
    for tc in test_cases:
        click.echo(f"  - {tc.course_key}/{tc.activity_key}: {tc.prompt[:40]}...")
        tc.system_message, followup_questions = await ai_engine.make_system_message(history=[], query=tc.prompt, course_key=tc.course_key, 
                                                  activity_key=tc.activity_key, condensed_history="")

    # Save test cases with system messages to JSON file
    output_dir = cases_path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (cases_path.stem + "-sysmsg.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([tc.model_dump() for tc in test_cases], f, indent=2, ensure_ascii=False)
    click.echo(f"Saved test cases with system messages to {output_path}")
