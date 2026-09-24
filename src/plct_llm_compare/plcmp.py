"""CLI entrypoint for plcmp."""

import asyncio
import click

from .prepare import do_prepare
from .inference import do_inference
from .judge_compare import do_judge_compare
from .human_eval import do_human_eval
from .calibrate import do_calibrate
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
    "--cases-b",
    default=None,
    type=click.Path(exists=True),
    help=(
        "Optional second prepared cases file. When given, each side is judged "
        "against its own system message, so a system-message change can be "
        "compared instead of a model or a take."
    ),
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
    cases_b: str | None,
    model_a: str,
    model_b: str,
    model_a_take: int,
    model_b_take: int,
    judge_model: str,
) -> None:
    """Compare two sets of pre-generated answers, judged by a third model.

    With one cases file both sides share a system message and only the model or
    the take differs. Pass --cases-b to give each side its own system message
    and compare a system-message change instead.
    """
    asyncio.run(
        do_judge_compare(
            cases,
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
    "--scorer-model",
    default=None,
    help=(
        "Override the teacher model the judge runs on (e.g. gpt-5.4). Sweeps "
        "the model without touching the prompts."
    ),
)
@click.option(
    "--tie-band",
    default=5,
    type=int,
    help="Score difference at or below which the derived verdict is Tie.",
)
@click.option(
    "--out-dir",
    "-o",
    default="eval/output/calibrate",
    type=click.Path(file_okay=False),
    help="Directory for the annotation and scorer artifacts.",
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
@click.option(
    "--concurrency",
    default=4,
    type=int,
    help="Parallel scoring calls.",
)
def calibrate(
    cases: str,
    model_a: str,
    model_b: str,
    model_a_take: int,
    model_b_take: int,
    scorer_model: str | None,
    tie_band: int,
    out_dir: str,
    seed: int,
    no_shuffle: bool,
    force: bool,
    concurrency: int,
) -> None:
    """Score both sides of each pair independently, for comparison against humans.

    Runs the fine-tuning pipeline's judge on each answer alone — it never sees
    a pair — then derives an A/B/Tie verdict from the two substance scores.
    Writes eval_answers.yml alongside the blind human_feedback.yml so the two
    can be read side by side.
    """
    do_calibrate(
        cases,
        model_a,
        model_b,
        model_a_take,
        model_b_take,
        out_dir,
        tie_band,
        scorer_model=scorer_model,
        seed=seed,
        shuffle=not no_shuffle,
        force=force,
        concurrency=concurrency,
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
