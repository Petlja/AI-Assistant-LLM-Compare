"""Human-eval annotation artifacts: combined YAML file + a side-by-side HTML viewer.

Reads pre-generated inference answers for a (model_a, take_a) / (model_b, take_b)
pair and produces, per pair, three artifacts in one directory:

- annotations.yml — ONE combined file the human annotator reads and fills in
  (human_verdict / human_notes per case). It contains no model/take/swap
  metadata, so the annotator stays blind to which side is which.
- assignment.yml — the answer key: per-case blind-shuffle assignment mapping
  displayed A/B back to the true (model, take) sources. Not for annotators.
- index.html — a self-contained side-by-side viewer as a reading aid.

Which answer is displayed as "A" is blind-shuffled per case (seeded,
deterministic) so human annotators are position-debiased like the judge.
Regeneration carries existing human input over instead of destroying it.
"""

import json
import random
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

ANNOTATIONS_FILE = "annotations.yml"
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


def _dump_yaml(data: dict) -> str:
    return yaml.dump(
        data,
        Dumper=_AnnotationDumper,
        allow_unicode=True,
        sort_keys=False,
        width=10000,
    )


def _pair_dir_name(model_a: str, take_a: int, model_b: str, take_b: int) -> str:
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


def _load_existing_annotations(
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
        VIEWER_TEMPLATE.replace("__TITLE__", f"Human eval: {pair_meta['pair_dir']}")
        .replace("__PAIR_META__", embed(pair_meta))
        .replace("__CASE_DATA__", embed(cases))
    )


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

    model_a_safe = safe_model_name(model_a)
    model_b_safe = safe_model_name(model_b)
    pair_dir_name = _pair_dir_name(model_a, model_a_take, model_b, model_b_take)
    pair_dir = Path(out_dir) / pair_dir_name
    pair_dir.mkdir(parents=True, exist_ok=True)

    if force:
        annotated, previous_swapped = {}, {}
    else:
        annotated, previous_swapped = _load_existing_annotations(pair_dir)

    md = MarkdownIt()
    annotations: list[HumanEvalAnnotation] = []
    assignments: list[HumanEvalCaseAssignment] = []
    viewer_cases: list[dict] = []
    skipped: list[str] = []
    preserved = 0
    flipped = 0

    click.echo(f"Loaded {len(test_cases)} test cases:")
    for tc in test_cases:
        answer_a_file = cases_path.parent / f"{tc.case_key}_{model_a_take}_{model_a_safe}.txt"
        answer_b_file = cases_path.parent / f"{tc.case_key}_{model_b_take}_{model_b_safe}.txt"
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
        displayed_a, displayed_b = (answer_b, answer_a) if swapped else (answer_a, answer_b)
        if swapped:
            side_a, side_b = (model_b, model_b_take), (model_a, model_a_take)
        else:
            side_a, side_b = (model_a, model_a_take), (model_b, model_b_take)

        verdict, notes = "", ""
        if tc.case_key in annotated:
            verdict = annotated[tc.case_key].human_verdict
            notes = annotated[tc.case_key].human_notes
            # Verdicts refer to the displayed order; if the layout changed
            # (e.g. a different --seed), remap them so they keep their meaning.
            if previous_swapped.get(tc.case_key, False) != swapped:
                verdict = _flip_verdict(verdict)
                flipped += 1
            preserved += 1

        annotations.append(
            HumanEvalAnnotation(
                id=tc.case_key,
                prompt=_yaml_text(tc.prompt),
                answer_a=_yaml_text(displayed_a),
                answer_b=_yaml_text(displayed_b),
                human_verdict=verdict,
                human_notes=notes,
                activity_url=tc.activity_url,
                activity_desc=tc.activity_desc,
            )
        )
        assignments.append(
            HumanEvalCaseAssignment(
                id=tc.case_key,
                swapped=swapped,
                model_a=side_a[0],
                take_a=side_a[1],
                model_b=side_b[0],
                take_b=side_b[1],
            )
        )
        # The viewer renders the unnormalized text so markdown hard breaks
        # (trailing double spaces) survive.
        viewer_cases.append(
            {
                "id": tc.case_key,
                "prompt_html": md.render(tc.prompt),
                "answer_a_html": md.render(displayed_a),
                "answer_b_html": md.render(displayed_b),
                "activity_url": tc.activity_url,
                "activity_desc": tc.activity_desc,
            }
        )

    if not viewer_cases:
        click.echo("No cases with both answer files present — nothing generated.")
        return

    annotations_file = pair_dir / ANNOTATIONS_FILE
    annotations_model = HumanEvalAnnotationsFile(pair_dir=pair_dir_name, cases=annotations)
    annotations_file.write_text(
        ANNOTATIONS_HEADER + _dump_yaml(annotations_model.model_dump()),
        encoding="utf-8",
    )

    assignment_model = HumanEvalAssignmentsFile(
        model_a=model_a,
        take_a=model_a_take,
        model_b=model_b,
        take_b=model_b_take,
        shuffled=shuffle,
        seed=seed,
        cases=assignments,
    )
    (pair_dir / ASSIGNMENT_FILE).write_text(
        ASSIGNMENT_HEADER + _dump_yaml(assignment_model.model_dump()),
        encoding="utf-8",
    )

    pair_meta = {
        "pair_dir": pair_dir_name,
        "model_a": model_a,
        "take_a": model_a_take,
        "model_b": model_b,
        "take_b": model_b_take,
        "shuffled": shuffle,
        "seed": seed,
    }
    index_file = pair_dir / "index.html"
    index_file.write_text(_build_viewer_html(pair_meta, viewer_cases), encoding="utf-8")

    summary = f"Done: {len(viewer_cases)} case(s) in {annotations_file}"
    if preserved:
        summary += f", {preserved} existing annotation(s) preserved"
    if flipped:
        summary += f" ({flipped} verdict(s) remapped to a changed layout)"
    if skipped:
        summary += f", {len(skipped)} case(s) skipped"
    click.echo(summary + ".")
    click.echo(f"Viewer: {index_file}")
