"""Blind annotation artifacts: a combined YAML file + a side-by-side HTML viewer.

Reads pre-generated inference answers for a (model_a, take_a) / (model_b, take_b)
pair and produces, per pair, three artifacts in one directory:

- human_feedback.yml — ONE combined file the human annotator reads and fills in
  (human_verdict / human_notes per case). It contains no model/take/swap
  metadata, so the annotator stays blind to which side is which.
- assignment.yml — the answer key: per-case blind-shuffle assignment mapping
  displayed A/B back to the true (model, take) sources. Not for annotators.
- index.html — a self-contained side-by-side viewer as a reading aid.

Which answer is displayed as "A" is blind-shuffled per case (seeded,
deterministic) so human annotators are position-debiased like the judge.
Regeneration carries existing human input over instead of destroying it.

`plcmp calibrate` reuses `build_paired_cases` and `write_annotation_artifacts`
from this module rather than reimplementing them. That is deliberate: the
scorer's eval_answers.yml has to line up row for row with human_feedback.yml,
and a second copy of the shuffle would eventually disagree with this one
silently. Keep exactly one implementation.
"""

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import click
import yaml
from markdown_it import MarkdownIt
from pydantic import TypeAdapter, ValidationError

from .human_eval_template import VIEWER_TEMPLATE
from .models import (
    HumanEvalAnnotation,
    HumanEvalAnnotationsFile,
    HumanEvalAssignmentsFile,
    HumanEvalCaseAssignment,
    TestCase,
    safe_model_name,
)

ANNOTATIONS_FILE = "human_feedback.yml"
ASSIGNMENT_FILE = "assignment.yml"

ANNOTATIONS_HEADER = (
    "# Uputstvo: za svaki slučaj pročitajte oba odgovora (najlakše u index.html\n"
    '# pored ovog fajla) i u polje human_verdict upišite "A", "B" ili "Tie"\n'
    "# (nerešeno). U human_notes možete kratko obrazložiti izbor. Ne menjajte\n"
    "# ostala polja.\n"
)

ASSIGNMENT_HEADER = (
    "# Answer key: which (model, take) was displayed as A/B per case.\n"
    "# Do not share with annotators — it breaks the blind evaluation.\n"
)


class _AnnotationDumper(yaml.SafeDumper):
    """SafeDumper with literal block style for multiline strings."""


def _str_representer(dumper: yaml.SafeDumper, data: str):
    # Literal block style (|) keeps answers readable/editable by annotators.
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_AnnotationDumper.add_representer(str, _str_representer)


def dump_yaml(data: dict) -> str:
    return yaml.dump(
        data,
        Dumper=_AnnotationDumper,
        allow_unicode=True,
        sort_keys=False,
        width=10000,
    )


def pair_dir_name(model_a: str, take_a: int, model_b: str, take_b: int) -> str:
    return (
        f"{safe_model_name(model_a)}_t{take_a}_vs_{safe_model_name(model_b)}_t{take_b}"
    )


def _yaml_text(text: str) -> str:
    """Normalize text for readable YAML: trailing whitespace on any line forces
    pyyaml out of literal block style into an unreadable quoted scalar, and model
    output routinely contains markdown hard breaks (two trailing spaces)."""
    return "\n".join(line.rstrip() for line in text.split("\n"))


def _flip_verdict(verdict: str) -> str:
    if verdict == "A":
        return "B"
    if verdict == "B":
        return "A"
    return verdict


@dataclass
class PairedCase:
    """One test case with both answers loaded and the blind shuffle applied.

    `displayed_*` is what the annotator sees; `canonical_*` is always the
    (model_a, take_a) / (model_b, take_b) given on the command line. Anything
    written in canonical frame must use the canonical fields, and anything that
    has to line up with the annotator's view must use the displayed ones.
    """

    case: TestCase
    swapped: bool
    displayed_a: str
    displayed_b: str
    side_a: tuple[str, int]
    side_b: tuple[str, int]
    canonical_a: str
    canonical_b: str


def build_paired_cases(
    cases_path: Path,
    test_cases: list[TestCase],
    model_a: str,
    take_a: int,
    model_b: str,
    take_b: int,
    *,
    seed: int = 0,
    shuffle: bool = True,
) -> tuple[list[PairedCase], list[str]]:
    """Load both answer files per case and apply the seeded blind shuffle.

    Returns (paired cases, skipped case keys). Cases missing either answer file
    are skipped with a message rather than aborting the run.
    """
    model_a_safe = safe_model_name(model_a)
    model_b_safe = safe_model_name(model_b)

    paired: list[PairedCase] = []
    skipped: list[str] = []

    for tc in test_cases:
        answer_a_file = cases_path.parent / f"{tc.case_key}_{take_a}_{model_a_safe}.txt"
        answer_b_file = cases_path.parent / f"{tc.case_key}_{take_b}_{model_b_safe}.txt"
        missing = [p.name for p in (answer_a_file, answer_b_file) if not p.exists()]
        if missing:
            click.echo(
                f"  - {tc.case_key}: skipping, missing answer file(s): "
                f"{', '.join(missing)}"
            )
            skipped.append(tc.case_key)
            continue

        answer_a = answer_a_file.read_text(encoding="utf-8").replace("\r\n", "\n")
        answer_b = answer_b_file.read_text(encoding="utf-8").replace("\r\n", "\n")

        # Seeded per case: the same seed always yields the same layout, so
        # regeneration never scrambles already-annotated cases.
        swapped = shuffle and random.Random(f"{seed}:{tc.case_key}").random() < 0.5
        if swapped:
            displayed_a, displayed_b = answer_b, answer_a
            side_a, side_b = (model_b, take_b), (model_a, take_a)
        else:
            displayed_a, displayed_b = answer_a, answer_b
            side_a, side_b = (model_a, take_a), (model_b, take_b)

        paired.append(
            PairedCase(
                case=tc,
                swapped=swapped,
                displayed_a=displayed_a,
                displayed_b=displayed_b,
                side_a=side_a,
                side_b=side_b,
                canonical_a=answer_a,
                canonical_b=answer_b,
            )
        )

    return paired, skipped


def load_existing_annotations(
    pair_dir: Path,
) -> tuple[dict[str, HumanEvalAnnotation], dict[str, bool]]:
    """Collect human input from a previous run so regeneration never destroys it.

    Returns ({case_id: annotation with human input}, {case_id: previous swapped
    flag}). Aborts rather than overwrite an annotations file it cannot parse.
    """
    annotated: dict[str, HumanEvalAnnotation] = {}
    ann_path = pair_dir / ANNOTATIONS_FILE
    if ann_path.exists():
        try:
            parsed = HumanEvalAnnotationsFile.model_validate(
                yaml.safe_load(ann_path.read_text(encoding="utf-8"))
            )
        except (yaml.YAMLError, ValidationError) as exc:
            raise click.ClickException(
                f"Existing {ann_path} could not be parsed — it may contain "
                f"annotations, so it will not be overwritten. Fix it or rerun "
                f"with --force.\n{exc}"
            )
        for case in parsed.cases:
            if case.human_verdict.strip() or case.human_notes.strip():
                annotated[case.id] = case

    previous_swapped: dict[str, bool] = {}
    asg_path = pair_dir / ASSIGNMENT_FILE
    if asg_path.exists():
        try:
            assignment = HumanEvalAssignmentsFile.model_validate(
                yaml.safe_load(asg_path.read_text(encoding="utf-8"))
            )
            previous_swapped = {c.id: c.swapped for c in assignment.cases}
        except (yaml.YAMLError, ValidationError):
            # The assignment file is regenerated from scratch anyway; without a
            # readable previous one, carried verdicts are assumed un-flipped.
            pass
    return annotated, previous_swapped


def _build_viewer_html(pair_meta: dict, cases: list[dict]) -> str:
    def embed(obj) -> str:
        # </ escaped so an answer containing </script> cannot break the page.
        return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")

    return (
        VIEWER_TEMPLATE.replace("__TITLE__", f"Human eval: {pair_meta['pair_id']}")
        .replace("__PAIR_META__", embed(pair_meta))
        .replace("__CASE_DATA__", embed(cases))
    )


def write_annotation_artifacts(
    pair_dir: Path,
    paired: list[PairedCase],
    model_a: str,
    take_a: int,
    model_b: str,
    take_b: int,
    *,
    seed: int = 0,
    shuffle: bool = True,
    force: bool = False,
) -> dict[str, int]:
    """Write human_feedback.yml, assignment.yml and index.html for a pair.

    Existing human input is carried over, and remapped if the displayed layout
    changed since it was collected. Returns counts for the caller's summary.
    """
    if force:
        annotated, previous_swapped = {}, {}
    else:
        annotated, previous_swapped = load_existing_annotations(pair_dir)

    md = MarkdownIt()
    annotations: list[HumanEvalAnnotation] = []
    assignments: list[HumanEvalCaseAssignment] = []
    viewer_cases: list[dict] = []
    preserved = 0
    flipped = 0

    for pc in paired:
        tc = pc.case
        verdict, notes = "", ""
        if tc.case_key in annotated:
            verdict = annotated[tc.case_key].human_verdict
            notes = annotated[tc.case_key].human_notes
            # Verdicts refer to the displayed order; if the layout changed
            # (e.g. a different --seed), remap them so they keep their meaning.
            if previous_swapped.get(tc.case_key, False) != pc.swapped:
                verdict = _flip_verdict(verdict)
                flipped += 1
            preserved += 1

        annotations.append(
            HumanEvalAnnotation(
                id=tc.case_key,
                prompt=_yaml_text(tc.prompt),
                answer_a=_yaml_text(pc.displayed_a),
                answer_b=_yaml_text(pc.displayed_b),
                human_verdict=verdict,
                human_notes=notes,
                activity_url=tc.activity_url,
                activity_desc=tc.activity_desc,
            )
        )
        assignments.append(
            HumanEvalCaseAssignment(
                id=tc.case_key,
                swapped=pc.swapped,
                model_a=pc.side_a[0],
                take_a=pc.side_a[1],
                model_b=pc.side_b[0],
                take_b=pc.side_b[1],
            )
        )
        # The viewer renders the unnormalized text so markdown hard breaks
        # (trailing double spaces) survive.
        viewer_cases.append(
            {
                "id": tc.case_key,
                "prompt_html": md.render(tc.prompt),
                "answer_a_html": md.render(pc.displayed_a),
                "answer_b_html": md.render(pc.displayed_b),
                "activity_url": tc.activity_url,
                "activity_desc": tc.activity_desc,
            }
        )

    dir_name = pair_dir.name
    # Opaque, deterministic, and derived from the directory name, so a
    # regenerated pair keeps its id and a returned file can still be matched to
    # the pair it came from — without naming either model to the annotator.
    pair_id = hashlib.sha256(dir_name.encode("utf-8")).hexdigest()[:12]
    annotations_model = HumanEvalAnnotationsFile(pair_id=pair_id, cases=annotations)
    (pair_dir / ANNOTATIONS_FILE).write_text(
        ANNOTATIONS_HEADER + dump_yaml(annotations_model.model_dump()),
        encoding="utf-8",
    )

    assignment_model = HumanEvalAssignmentsFile(
        pair_id=pair_id,
        model_a=model_a,
        take_a=take_a,
        model_b=model_b,
        take_b=take_b,
        shuffled=shuffle,
        seed=seed,
        cases=assignments,
    )
    (pair_dir / ASSIGNMENT_FILE).write_text(
        ASSIGNMENT_HEADER + dump_yaml(assignment_model.model_dump()),
        encoding="utf-8",
    )

    # The viewer goes to the annotator, so it gets the id and nothing else.
    # Model, take and seed live in assignment.yml.
    pair_meta = {"pair_id": pair_id}
    (pair_dir / "index.html").write_text(
        _build_viewer_html(pair_meta, viewer_cases), encoding="utf-8"
    )

    return {"cases": len(viewer_cases), "preserved": preserved, "flipped": flipped}


def do_human_eval(
    cases_fname: str,
    model_a: str,
    model_b: str,
    model_a_take: int,
    model_b_take: int,
    out_dir: str,
    seed: int = 0,
    shuffle: bool = True,
    force: bool = False,
) -> None:
    cases_path = Path(cases_fname)
    with cases_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    test_cases = TypeAdapter(list[TestCase]).validate_python(data)

    if model_a == model_b and model_a_take == model_b_take:
        click.echo(
            "Warning: sides A and B are the same model and take — "
            "both columns will show the identical answer."
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

    summary = f"Done: {stats['cases']} case(s) in {pair_dir / ANNOTATIONS_FILE}"
    if stats["preserved"]:
        summary += f", {stats['preserved']} existing annotation(s) preserved"
    if stats["flipped"]:
        summary += f" ({stats['flipped']} verdict(s) remapped to a changed layout)"
    if skipped:
        summary += f", {len(skipped)} case(s) skipped"
    click.echo(summary + ".")
    click.echo(f"Viewer: {pair_dir / 'index.html'}")
