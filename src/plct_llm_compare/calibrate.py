"""Calibrate the fine-tuning pipeline's pointwise scorer against human preference.

`astft gen-td` routes an answer to SFT or DPO using a POINTWISE score: the
teacher model sees one answer, alone, and returns a number. Nobody has checked
that those numbers mean anything.

Humans are unreliable at absolute scores but good at "which of these two is
better", so this command bridges the two: it scores each side of a pair
independently — never showing the scorer both answers, exactly as in `gen-td` —
and derives a pairwise verdict from the difference. That verdict can then be
compared against a human's blind verdict on the same pair.

It writes four files per pair and stops there. Computing agreement is done by
hand or with outside tools, deliberately: see PLAN.md.

    <out-dir>/<pair>/
      index.html          blind side-by-side reading aid
      human_feedback.yml  the annotator fills this in
      eval_answers.yml    what the scorer said, row-aligned with the above
      assignment.yml      answer key: displayed A/B -> true (model, take)
      improved/           the scorer's improved_answer per case and side

The first three are meant to be read together, so eval_answers.yml is written in
DISPLAYED frame: its `score_a` is the score of whatever the annotator saw as A.
"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
from openai import OpenAI
from pydantic import TypeAdapter

from .config import OPENAI_API_KEY
from .human_eval import (
    ANNOTATIONS_FILE,
    PairedCase,
    build_paired_cases,
    dump_yaml,
    pair_dir_name,
    write_annotation_artifacts,
)
from .models import (
    CalibrationCaseResult,
    CalibrationResultsFile,
    TestCase,
    safe_model_name,
)
from .scorer_bridge import load_scoring

RESULTS_FILE = "eval_answers.yml"
IMPROVED_DIR = "improved"
CACHE_DIR = ".score_cache"

RESULTS_HEADER = (
    "# What the pointwise scorer said. DISPLAYED frame: score_a is the score of\n"
    "# the answer shown as A in human_feedback.yml, so the two files line up row\n"
    "# for row. `swapped: true` means displayed A is the pair's canonical B.\n"
    "#\n"
    "# Raw scores are kept so the tie band can be re-derived by hand at another\n"
    "# epsilon without re-running: |score_a - score_b| <= tie_band  =>  Tie.\n"
)


def _derive_verdict(score_a: int, score_b: int, tie_band: int) -> str:
    """Turn two independent scores into a pairwise verdict."""
    if abs(score_a - score_b) <= tie_band:
        return "Tie"
    return "A" if score_a > score_b else "B"


def _cache_key(answer: str, model: str, take: int, scorer: str) -> str:
    """Identify a scored answer.

    Keyed on the answer text too, not just (model, take): re-running inference
    replaces the answer while leaving the model and take identical, and a stale
    score there would be invisible.
    """
    digest = hashlib.sha256(answer.encode("utf-8")).hexdigest()[:16]
    return f"{scorer}_{safe_model_name(model)}_t{take}_{digest}"


class _Scorer:
    """Scores answers through the fine-tuning repo's scorer, with a disk cache.

    Calibration re-runs the same pairs repeatedly while sweeping variants, so
    paying twice for an identical (answer, scorer) call is pure waste.
    """

    def __init__(self, scoring, config, cache_dir: Path, client):
        self._scoring = scoring
        self._config = config
        self._cache_dir = cache_dir
        self._client = client
        self.hits = 0
        self.misses = 0

    def score(self, *, system_message: str, question: str, answer: str,
              model: str, take: int) -> dict:
        key = _cache_key(answer, model, take, self._config.name)
        cache_file = self._cache_dir / f"{key}.json"
        if cache_file.exists():
            self.hits += 1
            return json.loads(cache_file.read_text(encoding="utf-8"))

        result = self._scoring.score_answer(
            self._client,
            system_message=system_message,
            question=question,
            answer=answer,
            config=self._config,
        )
        payload = {
            "score": result.score,
            "scale_max": result.scale_max,
            "improved_answer": result.improved_answer,
            "scorer": result.scorer,
        }
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.misses += 1
        return payload


def _score_pair(scorer: _Scorer, pc: PairedCase, model_a: str, take_a: int,
                model_b: str, take_b: int) -> tuple[dict, dict]:
    """Score both sides of one case, in canonical order."""
    system_message = pc.case.system_message or ""
    a = scorer.score(
        system_message=system_message,
        question=pc.case.prompt,
        answer=pc.canonical_a,
        model=model_a,
        take=take_a,
    )
    b = scorer.score(
        system_message=system_message,
        question=pc.case.prompt,
        answer=pc.canonical_b,
        model=model_b,
        take=take_b,
    )
    return a, b


def do_calibrate(
    cases_fname: str,
    model_a: str,
    model_b: str,
    model_a_take: int,
    model_b_take: int,
    out_dir: str,
    scorer_name: str,
    tie_band: int,
    seed: int = 0,
    shuffle: bool = True,
    force: bool = False,
    concurrency: int = 4,
) -> None:
    try:
        scoring = load_scoring()
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    try:
        config = scoring.get_scorer(scorer_name)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if tie_band >= config.scale_max:
        raise click.ClickException(
            f"--tie-band {tie_band} on a 1-{config.scale_max} scale makes every "
            "case a Tie."
        )

    cases_path = Path(cases_fname)
    with cases_path.open("r", encoding="utf-8") as f:
        test_cases = TypeAdapter(list[TestCase]).validate_python(json.load(f))

    if model_a == model_b and model_a_take == model_b_take:
        click.echo(
            "Warning: sides A and B are the same model and take — "
            "both sides will score identically."
        )

    pair_dir = Path(out_dir) / pair_dir_name(model_a, model_a_take, model_b, model_b_take)
    pair_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Loaded {len(test_cases)} test cases:")
    paired, skipped = build_paired_cases(
        cases_path,
        test_cases,
        model_a,
        model_a_take,
        model_b,
        model_b_take,
        seed=seed,
        shuffle=shuffle,
    )
    if not paired:
        click.echo("No cases with both answer files present — nothing generated.")
        return

    click.echo(f"Scorer: {config.name} (model {config.model}, temp {config.temperature})")
    click.echo(f"Tie band: {tie_band} on a 1-{config.scale_max} scale")

    scorer = _Scorer(
        scoring,
        config,
        Path(out_dir) / CACHE_DIR,
        OpenAI(api_key=OPENAI_API_KEY),
    )

    # 2N independent calls; the scorer never sees a pair, so they parallelise
    # freely. score_answer is synchronous, hence threads rather than asyncio.
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        scored = list(
            pool.map(
                lambda pc: _score_pair(
                    scorer, pc, model_a, model_a_take, model_b, model_b_take
                ),
                paired,
            )
        )

    improved_dir = pair_dir / IMPROVED_DIR
    improved_dir.mkdir(parents=True, exist_ok=True)

    results: list[CalibrationCaseResult] = []
    verdict_counts = {"A": 0, "B": 0, "Tie": 0}

    for pc, (canon_a, canon_b) in zip(paired, scored):
        # Flip into displayed frame so this file reads alongside human_feedback.yml.
        shown_a, shown_b = (canon_b, canon_a) if pc.swapped else (canon_a, canon_b)
        verdict = _derive_verdict(shown_a["score"], shown_b["score"], tie_band)
        verdict_counts[verdict] += 1

        results.append(
            CalibrationCaseResult(
                id=pc.case.case_key,
                swapped=pc.swapped,
                score_a=shown_a["score"],
                score_b=shown_b["score"],
                verdict=verdict,
            )
        )
        for side, payload in (("a", shown_a), ("b", shown_b)):
            (improved_dir / f"{pc.case.case_key}_{side}.md").write_text(
                payload["improved_answer"], encoding="utf-8"
            )

        click.echo(
            f"  - {pc.case.case_key}: A={shown_a['score']} B={shown_b['score']} "
            f"-> {verdict}"
        )

    stats = write_annotation_artifacts(
        pair_dir,
        paired,
        model_a,
        model_a_take,
        model_b,
        model_b_take,
        seed=seed,
        shuffle=shuffle,
        force=force,
    )

    results_model = CalibrationResultsFile(
        scorer=config.name,
        scale_max=config.scale_max,
        tie_band=tie_band,
        model_a=model_a,
        take_a=model_a_take,
        model_b=model_b,
        take_b=model_b_take,
        cases=results,
    )
    (pair_dir / RESULTS_FILE).write_text(
        RESULTS_HEADER + dump_yaml(results_model.model_dump()), encoding="utf-8"
    )

    n = len(results)
    click.echo("")
    click.echo(f"=== Scorer verdicts ({n} cases, tie band {tie_band}) ===")
    for key in ("A", "B", "Tie"):
        click.echo(f"  {key:<4s} {verdict_counts[key]:>3d}  ({verdict_counts[key] / n:.0%})")
    click.echo(f"  API calls: {scorer.misses} made, {scorer.hits} served from cache")
    if skipped:
        click.echo(f"  Skipped {len(skipped)} case(s) with missing answer files")
    if stats["preserved"]:
        note = f"  Preserved {stats['preserved']} existing human annotation(s)"
        if stats["flipped"]:
            note += f" ({stats['flipped']} remapped to a changed layout)"
        click.echo(note)

    click.echo("")
    click.echo(f"  Scorer verdicts : {pair_dir / RESULTS_FILE}")
    click.echo(f"  For annotators  : {pair_dir / ANNOTATIONS_FILE}")
    click.echo(f"  Reading aid     : {pair_dir / 'index.html'}")
    click.echo(f"  Answer key      : {pair_dir / 'assignment.yml'}")
