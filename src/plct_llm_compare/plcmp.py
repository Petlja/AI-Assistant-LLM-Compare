"""CLI entrypoint for plcmp."""

import asyncio
import click

from .prepare import do_prepare
from .inference import do_inference
from .judge_compare import do_judge_compare
from .survey import do_survey


class OrderedGroup(click.Group):
    """A Click group that lists commands in registration order."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        return list(self.commands)


@click.group(cls=OrderedGroup)
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


@main.command(name="judge_compare")
@click.option(
    "--cases",
    "-c",
    default="eval/output/test-cases-sysmsg.json",
    type=click.Path(exists=True),
    help="Path to the test cases with system messages JSON file.",
)
@click.option(
    "--model-a",
    default="gpt-4o-mini",
    help="First model to generate an answer.",
)
@click.option(
    "--model-b",
    default="gpt-4o",
    help="Second model to generate an answer.",
)
@click.option(
    "--judge-model",
    default="gpt-4o",
    help="Judge model that compares answers from model A and model B.",
)
def judge_compare(cases: str, model_a: str, model_b: str, judge_model: str) -> None:
    """Run model-vs-model comparison judged by a third model."""
    asyncio.run(do_judge_compare(cases, model_a, model_b, judge_model))


@main.command()
@click.option(
    "--output-dir",
    "-o",
    default="eval/output",
    type=click.Path(exists=True, file_okay=False),
    help="Path to the output directory with HTML and JSON files.",
)
def survey(output_dir: str) -> None:
    """Generate a SurveyJS survey.json from inference outputs."""
    do_survey(output_dir)


if __name__ == "__main__":
    main()
