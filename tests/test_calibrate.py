"""Tests for `plcmp calibrate`.

No network and no dependency on the sibling fine-tuning checkout: the scorer is
stubbed. What matters here is the thing a human will rely on when doing the
comparison by hand — that eval_answers.yml lines up row for row with
human_feedback.yml, in the same frame. A silent frame or ordering mismatch would
invert conclusions rather than produce an obvious error.
"""

import json
import types
from pathlib import Path

import pytest
import yaml

from plct_llm_compare import calibrate as calibrate_mod
from plct_llm_compare.calibrate import _derive_verdict, do_calibrate
from plct_llm_compare.human_eval import ANNOTATIONS_FILE, pair_dir_name
from plct_llm_compare.models import (
    CalibrationResultsFile,
    HumanEvalAnnotationsFile,
    HumanEvalAssignmentsFile,
)

MODEL = "m1"
PAIR_DIR = pair_dir_name(MODEL, 1, MODEL, 2)

# Score per answer text. Take 1 always beats take 2 by 20 points, except T3
# where they sit 3 apart so a tie band of 5 collapses it.
SCORES = {1: 80, 2: 60}
CLOSE_CASE = "T3"


class _StubConfig:
    name = "v1-baseline"
    model = "stub-model"
    temperature = 0.3
    scale_max = 100


def _stub_scoring(calls: list):
    """A stand-in for ai_assistant_fine_tuning.scoring."""

    def score_answer(client, *, system_message, question, answer, config):
        calls.append(answer)
        take = 1 if "Take-one" in answer else 2
        score = SCORES[take]
        if CLOSE_CASE in answer:
            # Deliberately close: 80 vs 77 is inside a tie band of 5.
            score = 80 if take == 1 else 77
        return types.SimpleNamespace(
            score=score,
            scale_max=100,
            improved_answer=f"improved({answer[:12]})",
            scorer=config.name,
        )

    def get_scorer(name="v1-baseline", *, model=None):
        if name != "v1-baseline":
            raise ValueError(f"Unknown scorer variant {name!r}. Known variants: v1-baseline")
        return _StubConfig()

    return types.SimpleNamespace(score_answer=score_answer, get_scorer=get_scorer)


@pytest.fixture
def env(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setattr(calibrate_mod, "load_scoring", lambda: _stub_scoring(calls))
    monkeypatch.setattr(calibrate_mod, "OpenAI", lambda **kw: object())

    cases = []
    for i in range(1, 4):
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
        (tmp_path / f"{key}_1_{MODEL}.txt").write_text(
            f"Take-one answer for {key}", encoding="utf-8"
        )
        (tmp_path / f"{key}_2_{MODEL}.txt").write_text(
            f"Take-two answer for {key}", encoding="utf-8"
        )
    cases_file = tmp_path / "cases.json"
    cases_file.write_text(json.dumps(cases), encoding="utf-8")
    return types.SimpleNamespace(
        cases_file=cases_file, out_dir=tmp_path / "out", calls=calls
    )


def _run(env, **kwargs) -> Path:
    kwargs.setdefault("scorer_name", "v1-baseline")
    kwargs.setdefault("tie_band", 5)
    do_calibrate(
        str(env.cases_file), MODEL, MODEL, 1, 2, str(env.out_dir), **kwargs
    )
    return env.out_dir / PAIR_DIR


def _results(pair_dir: Path) -> CalibrationResultsFile:
    return CalibrationResultsFile.model_validate(
        yaml.safe_load((pair_dir / "eval_answers.yml").read_text(encoding="utf-8"))
    )


def _annotations(pair_dir: Path) -> HumanEvalAnnotationsFile:
    return HumanEvalAnnotationsFile.model_validate(
        yaml.safe_load((pair_dir / ANNOTATIONS_FILE).read_text(encoding="utf-8"))
    )


# --------------------------------------------------------------------------
# Verdict derivation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b,band,expected",
    [
        (80, 60, 5, "A"),
        (60, 80, 5, "B"),
        (80, 77, 5, "Tie"),
        (80, 75, 5, "Tie"),   # exactly on the band -> Tie
        (80, 74, 5, "A"),     # just outside
        (50, 50, 0, "Tie"),
        (51, 50, 0, "A"),
    ],
)
def test_derive_verdict(a, b, band, expected):
    assert _derive_verdict(a, b, band) == expected


# --------------------------------------------------------------------------
# The alignment guarantee
# --------------------------------------------------------------------------


def test_all_four_files_are_written(env):
    pair_dir = _run(env)
    for name in ("eval_answers.yml", ANNOTATIONS_FILE, "assignment.yml", "index.html"):
        assert (pair_dir / name).exists(), name


def test_results_rows_align_with_annotation_rows(env):
    pair_dir = _run(env)
    # Same cases, same order — this is what makes a by-hand comparison possible.
    assert [c.id for c in _results(pair_dir).cases] == [
        c.id for c in _annotations(pair_dir).cases
    ]


def test_results_are_in_displayed_frame(env):
    """score_a must describe the answer shown as A, not the canonical A."""
    pair_dir = _run(env)
    results = {c.id: c for c in _results(pair_dir).cases}
    annotations = {c.id: c for c in _annotations(pair_dir).cases}
    assignment = HumanEvalAssignmentsFile.model_validate(
        yaml.safe_load((pair_dir / "assignment.yml").read_text(encoding="utf-8"))
    )

    for case in assignment.cases:
        shown_a = annotations[case.id].answer_a
        expected_take = 1 if "Take-one" in shown_a else 2
        expected = 80 if expected_take == 1 else 60
        if case.id == CLOSE_CASE:
            expected = 80 if expected_take == 1 else 77
        assert results[case.id].score_a == expected, (
            f"{case.id}: score_a should describe the displayed A answer"
        )
        # And the row's own swapped flag agrees with the answer key.
        assert results[case.id].swapped == case.swapped


def test_declared_frame_is_displayed(env):
    assert _results(_run(env)).frame == "displayed"


def test_tie_band_collapses_the_close_case(env):
    verdicts = {c.id: c.verdict for c in _results(_run(env)).cases}
    assert verdicts[CLOSE_CASE] == "Tie"
    assert set(verdicts) - {CLOSE_CASE}
    for case_id, verdict in verdicts.items():
        if case_id != CLOSE_CASE:
            assert verdict in {"A", "B"}


def test_raw_scores_allow_rederiving_the_band_by_hand(env):
    results = _results(_run(env))
    # A wider band must be reachable from the file alone, no re-run.
    rederived = {
        c.id: _derive_verdict(c.score_a, c.score_b, 25) for c in results.cases
    }
    assert set(rederived.values()) == {"Tie"}


# --------------------------------------------------------------------------
# Cost control and side artifacts
# --------------------------------------------------------------------------


def test_second_run_is_served_entirely_from_cache(env):
    _run(env)
    first = len(env.calls)
    assert first == 6  # 3 cases x 2 sides
    _run(env)
    assert len(env.calls) == first, "re-run should not call the scorer again"


def test_changed_answer_invalidates_the_cached_score(env):
    _run(env)
    before = len(env.calls)
    # Re-running inference replaces the text while model and take stay the same.
    (env.cases_file.parent / f"T1_1_{MODEL}.txt").write_text(
        "Take-one answer for T1 REGENERATED", encoding="utf-8"
    )
    _run(env)
    assert len(env.calls) == before + 1


def test_improved_answers_are_written_per_side(env):
    pair_dir = _run(env)
    for case_id in ("T1", "T2", "T3"):
        for side in ("a", "b"):
            path = pair_dir / "improved" / f"{case_id}_{side}.md"
            assert path.exists()
            assert path.read_text(encoding="utf-8").startswith("improved(")


def test_annotator_facing_file_leaks_no_scores(env):
    """The annotator must not be anchored by the scorer's opinion."""
    text = (_run(env) / ANNOTATIONS_FILE).read_text(encoding="utf-8")
    assert "score" not in text.lower()
    assert "swapped" not in text.lower()


# --------------------------------------------------------------------------
# Guard rails
# --------------------------------------------------------------------------


def test_unknown_scorer_is_a_clean_error(env):
    with pytest.raises(Exception, match="v1-baseline"):
        _run(env, scorer_name="nope")


def test_tie_band_swallowing_the_whole_scale_is_refused(env):
    with pytest.raises(Exception, match="Tie"):
        _run(env, tie_band=100)


def test_existing_human_input_survives_recalibration(env):
    pair_dir = _run(env)
    path = pair_dir / ANNOTATIONS_FILE
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["cases"][0]["human_verdict"] = "B"
    raw["cases"][0]["human_notes"] = "clearer"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    _run(env)
    after = _annotations(pair_dir).cases[0]
    assert after.human_verdict == "B"
    assert after.human_notes == "clearer"
