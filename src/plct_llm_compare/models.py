"""Data models for LLM comparison."""

from typing import Literal

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    """A single test case for LLM comparison."""

    case_key: str
    course_key: str
    activity_key: str
    activity_url: str
    activity_desc: str
    prompt: str
    system_message: str | None = None


class TestCaseResponce(BaseModel):
    """Metadata for an inference run, saved alongside the HTML output."""

    case_key: str
    activity_url: str
    activity_desc: str
    prompt: str
    model: str
    take: int


class TestCaseJudgeCompareResult(BaseModel):
    """Metadata for a judge-compare run, saved alongside the judge HTML output."""

    case_key: str
    activity_url: str
    activity_desc: str
    prompt: str
    model_a: str
    model_b: str
    judge_model: str
    take: int
    answer_a: str
    answer_b: str
    judge_result: "JudgeCompareStructuredResult"
    judge_result_ab: "JudgeCompareStructuredResult"
    judge_result_ba: "JudgeCompareStructuredResult"
    position_agreed: bool
    judge_response: str


class JudgeCompareCumulativeScores(BaseModel):
    score_a_sum: float = 0.0
    score_b_sum: float = 0.0
    winner_a_count: int = 0
    winner_b_count: int = 0
    no_winner_count: int = 0
    position_agree_count: int = 0
    count: int = 0

    def update(
        self,
        judge_result: "JudgeCompareStructuredResult",
        position_agreed: bool,
    ) -> None:
        """Update cumulative scores and winner counts based on a new judge result."""
        self.score_a_sum += judge_result.score_a
        self.score_b_sum += judge_result.score_b

        if judge_result.winner == "A":
            self.winner_a_count += 1
        elif judge_result.winner == "B":
            self.winner_b_count += 1
        else:
            self.no_winner_count += 1

        if position_agreed:
            self.position_agree_count += 1

        self.count += 1


class JudgeCompareStructuredResult(BaseModel):
    """Structured judge evaluation for comparing two model answers."""

    analysis: str = Field(
        description=(
            "Free-form comparative reasoning. For each answer, list specific "
            "strengths and weaknesses covering: correctness, relevance to the "
            "prompt and system message, completeness/depth, clarity, and "
            "educational usefulness. Then state which answer is better overall "
            "and why."
        ),
    )
    score_a: int = Field(
        ge=1,
        le=100,
        description="Overall quality score for Answer A on a 1-100 scale.",
    )
    score_b: int = Field(
        ge=1,
        le=100,
        description="Overall quality score for Answer B on a 1-100 scale.",
    )
    winner: Literal["A", "B", "Tie"] = Field(
        description=(
            "The better answer based on your analysis. "
            "A, B, or Tie (only when genuinely indistinguishable)."
        ),
    )

    @staticmethod
    def reconcile(
        ab: "JudgeCompareStructuredResult",
        ba: "JudgeCompareStructuredResult",
    ) -> tuple["JudgeCompareStructuredResult", bool]:
        """Reconcile AB and BA runs into a single result.

        BA scores/winner are flipped so that A always refers to model_a.
        Returns (reconciled_result, position_agreed).
        """
        ba_score_a = ba.score_b
        ba_score_b = ba.score_a
        ba_winner_flipped: Literal["A", "B", "Tie"]
        if ba.winner == "A":
            ba_winner_flipped = "B"
        elif ba.winner == "B":
            ba_winner_flipped = "A"
        else:
            ba_winner_flipped = "Tie"

        avg_score_a = round((ab.score_a + ba_score_a) / 2)
        avg_score_b = round((ab.score_b + ba_score_b) / 2)

        ab_winner = ab.winner
        position_agreed = ab_winner == ba_winner_flipped

        if ab_winner == ba_winner_flipped:
            final_winner = ab_winner
        else:
            # Any position-swap disagreement is treated as unstable, therefore Tie.
            final_winner = "Tie"
            position_agreed = False

        combined_analysis = (
            f"=== Run AB ===\n{ab.analysis}\n\n"
            f"=== Run BA (swapped) ===\n{ba.analysis}\n\n"
            f"=== Reconciliation ===\n"
            f"AB winner: {ab.winner} (A={ab.score_a}, B={ab.score_b}), "
            f"BA winner (flipped): {ba_winner_flipped} (A={ba_score_a}, B={ba_score_b})\n"
            f"Averaged scores: A={avg_score_a}, B={avg_score_b}\n"
            f"Position agreement: {'Yes' if position_agreed else 'No'}\n"
            f"Final winner: {final_winner}"
        )

        return (
            JudgeCompareStructuredResult(
                analysis=combined_analysis,
                score_a=avg_score_a,
                score_b=avg_score_b,
                winner=final_winner,
            ),
            position_agreed,
        )
