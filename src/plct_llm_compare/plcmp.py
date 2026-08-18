"""CLI entrypoint for plcmp."""

import asyncio
import click

from .prepare import do_prepare
from .inference import do_inference
from .judge_compare import do_judge_compare
from .human_eval import do_human_eval
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
@click.option(
    "--take",
    default=1,
    type=int,
    help="Take index for this inference run.",
)
@click.option(
    "--temperature",
    "-t",
    default=0.5,
    type=float,
    help="Sampling temperature (use ~0.7 when generating answer pairs).",
)
def inference(cases: str, model: str, take: int, temperature: float) -> None:
    """Inference of model"""
    asyncio.run(do_inference(cases, model, take, temperature))


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
    "--model-a-take",
    default=1,
    type=int,
    help="Take index for model A's pre-generated outputs.",
)
@click.option(
    "--model-b-take",
    default=1,
    type=int,
    help="Take index for model B's pre-generated outputs.",
)
@click.option(
    "--judge-model",
    default="gpt-4o",
    help="Judge model that compares answers from model A and model B.",
)
def judge_compare(
    cases: str,
    model_a: str,
    model_b: str,
    model_a_take: int,
    model_b_take: int,
    judge_model: str,
) -> None:
    """Run model-vs-model comparison judged by a third model."""
    asyncio.run(do_judge_compare(cases, model_a, model_b, model_a_take, model_b_take, judge_model))


@main.command(name="judge_compare_sysmsg")
@click.option(
    "--cases-a",
    required=True,
    type=click.Path(exists=True),
    help="Path to the first prepared cases JSON file with system messages.",
)
@click.option(
    "--cases-b",
    required=True,
    type=click.Path(exists=True),
    help="Path to the second prepared cases JSON file with system messages.",
)
@click.option(
    "--model-a",
    default="gpt-4o",
    help="Model used for answer set A.",
)
@click.option(
    "--model-b",
    default="gpt-4o",
    help="Model used for answer set B.",
)
@click.option(
    "--model-a-take",
    default=1,
    type=int,
    help="Take index for model A's pre-generated outputs.",
)
@click.option(
    "--model-b-take",
    default=2,
    type=int,
    help="Take index for model B's pre-generated outputs.",
)
@click.option(
    "--judge-model",
    default="gpt-4o",
    help="Judge model that compares answers from model A and model B.",
)
def judge_compare_sysmsg(
    cases_a: str,
    cases_b: str,
    model_a: str,
    model_b: str,
    model_a_take: int,
    model_b_take: int,
    judge_model: str,
) -> None:
    """Run a system-message comparison judged by a third model."""
    asyncio.run(
        do_judge_compare(
            cases_a,
            model_a,
            model_b,
            model_a_take,
            model_b_take,
            judge_model,
            cases_b_fname=cases_b,
        )
    )


@main.command(name="human_eval")
@click.option(
    "--cases",
    "-c",
    default="eval/output/test-cases-sysmsg.json",
    type=click.Path(exists=True),
    help="Path to the test cases with system messages JSON file.",
)
@click.option(
    "--model-a",
    default="gpt-4o",
    help="Model whose answers form side A (before the blind shuffle).",
)
@click.option(
    "--model-b",
    default="gpt-4o",
    help="Model whose answers form side B (before the blind shuffle).",
)
@click.option(
    "--model-a-take",
    default=1,
    type=int,
    help="Take index for model A's pre-generated outputs.",
)
@click.option(
    "--model-b-take",
    default=2,
    type=int,
    help="Take index for model B's pre-generated outputs.",
)
@click.option(
    "--out-dir",
    "-o",
    default="eval/output/human_eval",
    type=click.Path(file_okay=False),
    help="Directory for the annotation YAML, assignment key, and HTML viewer.",
)
@click.option(
    "--seed",
    default=0,
    type=int,
    help="Seed for the per-case blind shuffle (same seed = same layout).",
)
@click.option(
    "--no-shuffle",
    is_flag=True,
    default=False,
    help="Disable the blind shuffle; A is always (model-a, model-a-take).",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Regenerate from scratch, discarding existing human input.",
)
def human_eval(
    cases: str,
    model_a: str,
    model_b: str,
    model_a_take: int,
    model_b_take: int,
    out_dir: str,
    seed: int,
    no_shuffle: bool,
    force: bool,
) -> None:
    """Generate a combined annotation YAML and a side-by-side viewer for human input."""
    do_human_eval(
        cases,
        model_a,
        model_b,
        model_a_take,
        model_b_take,
        out_dir,
        seed=seed,
        shuffle=not no_shuffle,
        force=force,
    )


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
