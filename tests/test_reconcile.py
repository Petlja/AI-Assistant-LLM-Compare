"""Tests for AB/BA reconciliation in JudgeCompareReconciledResult."""

from plct_llm_compare.models import (
    JudgeCompareReconciledResult,
    JudgeCompareStructuredResult,
)


def _result(winner: str, score_a: int = 50, score_b: int = 50) -> JudgeCompareStructuredResult:
    return JudgeCompareStructuredResult(
        analysis="x", score_a=score_a, score_b=score_b, winner=winner
    )


def test_agreement_on_a():
    # BA run saying "B" means model_a won in the swapped frame — agreement.
    result, agreed = JudgeCompareReconciledResult.reconcile(_result("A"), _result("B"))
    assert result.winner == "A"
    assert agreed is True


def test_agreement_on_tie():
    result, agreed = JudgeCompareReconciledResult.reconcile(_result("Tie"), _result("Tie"))
    assert result.winner == "Tie"
    assert agreed is True


def test_hard_disagreement_is_tie():
    # Both runs pick position A, i.e. each run prefers whichever answer came
    # first. The judge has expressed no preference, so the verdict is a Tie —
    # but position_agreed stays False, which is what makes the order bias
    # visible downstream.
    result, agreed = JudgeCompareReconciledResult.reconcile(_result("A"), _result("A"))
    assert result.winner == "Tie"
    assert agreed is False


def test_soft_disagreement_is_tie():
    result, agreed = JudgeCompareReconciledResult.reconcile(_result("A"), _result("Tie"))
    assert result.winner == "Tie"
    assert agreed is False


def test_order_flip_tie_is_distinguishable_from_a_real_tie():
    # Both are Tie verdicts; only position_agreed separates "no preference"
    # from "no consistency". Downstream analysis depends on this.
    real, real_agreed = JudgeCompareReconciledResult.reconcile(_result("Tie"), _result("Tie"))
    flip, flip_agreed = JudgeCompareReconciledResult.reconcile(_result("A"), _result("A"))
    assert real.winner == flip.winner == "Tie"
    assert real_agreed is True
    assert flip_agreed is False


def test_score_flip_and_half_up_rounding():
    ab = _result("A", score_a=82, score_b=60)
    # BA frame: model_a sits in slot B.
    ba = _result("B", score_a=59, score_b=83)
    result, agreed = JudgeCompareReconciledResult.reconcile(ab, ba)
    assert agreed is True
    # (82 + 83) / 2 = 82.5 must round half-up to 83, not banker's 82.
    assert result.score_a == 83
    # (60 + 59) / 2 = 59.5 -> 60
    assert result.score_b == 60
