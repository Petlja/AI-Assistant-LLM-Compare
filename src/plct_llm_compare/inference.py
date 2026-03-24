import json
from pathlib import Path

import click
from markdown_it import MarkdownIt
from pydantic import TypeAdapter
from plct_server.ai.engine import AiEngine
from plct_server.ai.model_conf import ModelProvider
from plct_server.ai.client import AiClientFactory

from .models import TestCaseResponce, TestCase
from .config import OPENAI_API_KEY, VLLM_URL, PLCT_AI_CTX_URL


async def do_inference(cases_fname: str, model:str) -> None:
    cases_path = Path(cases_fname)
    with cases_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    test_cases = TypeAdapter(list[TestCase]).validate_python(data)

    client_factory = AiClientFactory(
        default_provider = ModelProvider.OPENAI,
        openai_api_key = OPENAI_API_KEY,
        azure_api_key = None,
        vllm_api_key= "EMPTY",
        vllm_url = VLLM_URL
    )

    ai_engine = AiEngine(ai_ctx_url=PLCT_AI_CTX_URL, client_factory=client_factory)

    model_config = ai_engine.get_model_config(model)
    if not model_config:
        raise ValueError(f"Unsupported model: {model}")


    click.echo(f"Loaded {len(test_cases)} test cases:")
    for tc in test_cases:
        messages = [{
            "role": "system",
            "content": tc.system_message
        }, {
            "role": "user",
            "content": tc.prompt
        }]
        for take in [1]:  # For now, we only do one take per test case
            click.echo(f"  - {tc.course_key}/{tc.activity_key} (take {take}):")
            client = client_factory.get_client(model_config=model_config)
            completion = await client.chat.completions.create(
                model=model_config.name,
                messages=messages,
                max_completion_tokens=8000,
                temperature=0.5
            )

            response = completion.choices[0].message.content
            md = MarkdownIt()
            html_content = md.render(response)
            model_safe = model.replace("/", "--")
            base_name = f"{tc.case_key}_{take}_{model_safe}"
            output_file = cases_path.parent / f"{base_name}.html"
            output_file.write_text(html_content, encoding="utf-8")

            metadata = TestCaseResponce(
                case_key=tc.case_key,
                activity_url=tc.activity_url,
                activity_desc=tc.activity_desc,
                prompt=tc.prompt,
                model=model,
                take=take,
            )
            meta_file = cases_path.parent / f"{base_name}.json"
            meta_file.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")

            click.echo(f"    Saved response to {output_file}")



