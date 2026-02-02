"""CLI entrypoint for plcmp."""

import asyncio
import click

from .prepare import do_prepare
from .inference import do_inference


@click.group()
def main() -> None:
    """PLCT LLM Compare CLI."""
    pass


@main.command()
@click.option(
    "--cases",
    "-c",
    default="eval/test-cases.yml",
    type=click.Path(exists=True),
    help="Path to the test cases YAML file.",
)
def prepare(cases: str) -> None:
    """Prepare resources for LLM comparison."""
    asyncio.run(do_prepare(cases))


@main.command()
@click.option(
    "--cases",
    "-c",
    default="eval/output/test-cases-sysmsg.json",
    type=click.Path(exists=True),
    help="Path to the test cases with system messages JSON file.",
)
@click.option(
    "--model",
    "-m",
    default="gpt-4o",
    help="Model to use for inference.",
)
def inference(cases: str, model: str) -> None:
    """Inference of model"""
    asyncio.run(do_inference(cases, model))

if __name__ == "__main__":
    main()
