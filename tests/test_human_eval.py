"""Tests for the human-eval annotation artifacts (combined YAML + viewer)."""

import json
from pathlib import Path

import yaml

from plct_llm_compare.human_eval import (
    ANNOTATIONS_FILE,
    ASSIGNMENT_FILE,
    pair_dir_name,
    do_human_eval,
)
from plct_llm_compare.models import (
    HumanEvalAnnotationsFile,
    HumanEvalAssignmentsFile,
)

MODEL = "m1"
PAIR_DIR = pair_dir_name(MODEL, 1, MODEL, 2)


def _setup_cases(tmp_path: Path, n: int = 3) -> Path:
    cases = []
    for i in range(1, n + 1):
        key = f"T{i}"
        cases.append(
            {
                "case_key": key,
                "course_key": "course",
                "activity_key": "activity",
                "activity_url": f"https://example.com/{key}",
                "activity_desc": f"Opis {key}",
                "prompt": f"Pitanje {key}?",
                "system_message": "sys",
            }
        )
        # Trailing double space = markdown hard break; must not force the YAML
        # dump out of literal block style.
        (tmp_path / f"{key}_1_{MODEL}.txt").write_text(
            f"Take-one answer for {key}  \n\nwith **markdown** and </script> inside",
            encoding="utf-8",
        )
        (tmp_path / f"{key}_2_{MODEL}.txt").write_text(
            f"Take-two answer for {key}", encoding="utf-8"
        )
    cases_file = tmp_path / "cases.json"
    cases_file.write_text(json.dumps(cases), encoding="utf-8")
    return cases_file


def _run(cases_file: Path, out_dir: Path, **kwargs) -> Path:
    do_human_eval(str(cases_file), MODEL, MODEL, 1, 2, str(out_dir), **kwargs)
    return out_dir / PAIR_DIR


def _load_annotations(pair_dir: Path) -> HumanEvalAnnotationsFile:
    return HumanEvalAnnotationsFile.model_validate(
        yaml.safe_load((pair_dir / ANNOTATIONS_FILE).read_text(encoding="utf-8"))
    )


def _load_assignment(pair_dir: Path) -> HumanEvalAssignmentsFile:
    return HumanEvalAssignmentsFile.model_validate(
        yaml.safe_load((pair_dir / ASSIGNMENT_FILE).read_text(encoding="utf-8"))
    )


def test_combined_yaml_round_trip(tmp_path):
    pair_dir = _run(_setup_cases(tmp_path), tmp_path / "he")
    annotations = _load_annotations(pair_dir)
    assert annotations.pair_dir == PAIR_DIR
    assert [c.id for c in annotations.cases] == ["T1", "T2", "T3"]

    case = annotations.cases[0]
    assert case.human_verdict == ""
    assert case.human_notes == ""
    # Regardless of shuffle, both answers survive the YAML round trip
    # (modulo per-line trailing-whitespace normalization).
    assert {case.answer_a, case.answer_b} == {
        "Take-one answer for T1\n\nwith **markdown** and </script> inside",
        "Take-two answer for T1",
    }

    # Multiline answers must be dumped in readable literal block style.
    yaml_text = (pair_dir / ANNOTATIONS_FILE).read_text(encoding="utf-8")
    assert "|-" in yaml_text
    assert "\\n" not in yaml_text


def test_annotations_file_is_blind(tmp_path):
    pair_dir = _run(_setup_cases(tmp_path), tmp_path / "he")
    yaml_text = (pair_dir / ANNOTATIONS_FILE).read_text(encoding="utf-8")
    # No swap/model/take metadata may leak into the annotator-facing file.
    for leak in ("swapped", "take_a", "take_b", "model_a", "model_b", "seed"):
        assert leak not in yaml_text


def test_assignment_answer_key(tmp_path):
    pair_dir = _run(_setup_cases(tmp_path), tmp_path / "he")
    assignment = _load_assignment(pair_dir)
    assert (assignment.model_a, assignment.take_a) == (MODEL, 1)
    assert (assignment.model_b, assignment.take_b) == (MODEL, 2)
    for case in assignment.cases:
        # Displayed side A must be take 2 exactly when swapped.
        assert case.take_a == (2 if case.swapped else 1)
        assert case.take_b == 3 - case.take_a


def test_shuffle_deterministic_per_seed(tmp_path):
    cases_file = _setup_cases(tmp_path)
    pair_dir = _run(cases_file, tmp_path / "he", seed=7)
    first = {c.id: c.swapped for c in _load_assignment(pair_dir).cases}
    pair_dir = _run(cases_file, tmp_path / "he", seed=7, force=True)
    second = {c.id: c.swapped for c in _load_assignment(pair_dir).cases}
    assert first == second


def test_no_shuffle(tmp_path):
    pair_dir = _run(_setup_cases(tmp_path), tmp_path / "he", shuffle=False)
    assignment = _load_assignment(pair_dir)
    assert assignment.shuffled is False
    assert all(c.swapped is False for c in assignment.cases)


def _annotate(pair_dir: Path, case_id: str, verdict: str, notes: str) -> None:
    path = pair_dir / ANNOTATIONS_FILE
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    for case in content["cases"]:
        if case["id"] == case_id:
            case["human_verdict"] = verdict
            case["human_notes"] = notes
    path.write_text(
        yaml.safe_dump(content, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def test_annotations_preserved_on_regeneration(tmp_path):
    cases_file = _setup_cases(tmp_path)
    pair_dir = _run(cases_file, tmp_path / "he")
    _annotate(pair_dir, "T2", "B", "bolji primeri")

    _run(cases_file, tmp_path / "he")
    kept = {c.id: c for c in _load_annotations(pair_dir).cases}
    assert kept["T2"].human_verdict == "B"
    assert kept["T2"].human_notes == "bolji primeri"
    assert kept["T1"].human_verdict == ""

    _run(cases_file, tmp_path / "he", force=True)
    cleared = {c.id: c for c in _load_annotations(pair_dir).cases}
    assert cleared["T2"].human_verdict == ""


def test_verdicts_remapped_when_layout_changes(tmp_path):
    cases_file = _setup_cases(tmp_path)
    pair_dir = _run(cases_file, tmp_path / "he", seed=0)
    before = {c.id: c.swapped for c in _load_assignment(pair_dir).cases}
    for case_id in before:
        _annotate(pair_dir, case_id, "A", "")

    # Find a seed that flips at least one case's layout relative to seed 0.
    new_seed = None
    import random as _random

    for candidate in range(1, 50):
        layout = {
            cid: _random.Random(f"{candidate}:{cid}").random() < 0.5 for cid in before
        }
        if layout != before:
            new_seed = candidate
            break
    assert new_seed is not None

    _run(cases_file, tmp_path / "he", seed=new_seed)
    after = {c.id: c.swapped for c in _load_assignment(pair_dir).cases}
    verdicts = {c.id: c.human_verdict for c in _load_annotations(pair_dir).cases}
    for case_id in before:
        expected = "A" if before[case_id] == after[case_id] else "B"
        assert verdicts[case_id] == expected


def test_missing_answer_file_skips_case(tmp_path):
    cases_file = _setup_cases(tmp_path)
    (tmp_path / f"T3_2_{MODEL}.txt").unlink()
    pair_dir = _run(cases_file, tmp_path / "he")
    assert [c.id for c in _load_annotations(pair_dir).cases] == ["T1", "T2"]
    assert [c.id for c in _load_assignment(pair_dir).cases] == ["T1", "T2"]


def test_viewer_html(tmp_path):
    pair_dir = _run(_setup_cases(tmp_path), tmp_path / "he")
    html = (pair_dir / "index.html").read_text(encoding="utf-8")
    assert '"id": "T1"' in html
    # Closing tags inside the embedded JSON must be escaped so an answer
    # containing </script> cannot terminate the data block.
    assert "<\\/p>" in html
    assert html.count("</script>") == 3  # pair-meta, case-data, app script
    # Pure reading aid: no annotation input in the viewer.
    assert "human_verdict" not in html
    assert "localStorage" not in html
