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


class JudgeCategoryScores(BaseModel):
    """Scores assigned by the judge for a single answer."""

    correctness: int = Field(
        ge=0,
        le=10,
        description="Correctness score from 0 to 10 based on factual and instructional accuracy.",
    )
    relevance: int = Field(
        ge=0,
        le=10,
        description="Relevance score from 0 to 10 based on how directly the answer addresses the prompt.",
    )
    clarity: int = Field(
        ge=0,
        le=10,
        description="Clarity score from 0 to 10 based on organization, readability, and precision.",
    )
    educational_usefulness: int = Field(
        ge=0,
        le=10,
        description="Educational usefulness score from 0 to 10 based on how helpful the answer is for teaching or learning.",
    )

    @staticmethod
    def zeros() -> "JudgeCategoryScores":
        return JudgeCategoryScores(correctness=0, relevance=0, clarity=0, educational_usefulness=0)

    def __iadd__(self, other: "JudgeCategoryScores") -> "JudgeCategoryScores":
        self.correctness += other.correctness
        self.relevance += other.relevance
        self.clarity += other.clarity
        self.educational_usefulness += other.educational_usefulness
        return self

class JudgeCompareCumulativeScores(BaseModel):
    scores_a: JudgeCategoryScores = Field(default_factory=JudgeCategoryScores.zeros)
    scores_b: JudgeCategoryScores = Field(default_factory=JudgeCategoryScores.zeros)
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
        self.scores_a += judge_result.scores_a
        self.scores_b += judge_result.scores_b

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
        description="Free-form comparative reasoning: evaluate both answers against the prompt and system message, note strengths, weaknesses, and key differences.",
    )
    scores_a: JudgeCategoryScores
    scores_b: JudgeCategoryScores
    winner: Literal["A", "B", "Tie"] = Field(
        description="Final winner chosen only after comparing both answers and assigning scores. Use A, B, or Tie.",
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
        ba_scores_a = ba.scores_b
        ba_scores_b = ba.scores_a
        ba_winner_flipped: Literal["A", "B", "Tie"]
        if ba.winner == "A":
            ba_winner_flipped = "B"
        elif ba.winner == "B":
            ba_winner_flipped = "A"
        else:
            ba_winner_flipped = "Tie"

        avg_scores_a = JudgeCategoryScores(
            correctness=round((ab.scores_a.correctness + ba_scores_a.correctness) / 2),
            relevance=round((ab.scores_a.relevance + ba_scores_a.relevance) / 2),
            clarity=round((ab.scores_a.clarity + ba_scores_a.clarity) / 2),
            educational_usefulness=round(
                (ab.scores_a.educational_usefulness + ba_scores_a.educational_usefulness) / 2
            ),
        )
        avg_scores_b = JudgeCategoryScores(
            correctness=round((ab.scores_b.correctness + ba_scores_b.correctness) / 2),
            relevance=round((ab.scores_b.relevance + ba_scores_b.relevance) / 2),
            clarity=round((ab.scores_b.clarity + ba_scores_b.clarity) / 2),
            educational_usefulness=round(
                (ab.scores_b.educational_usefulness + ba_scores_b.educational_usefulness) / 2
            ),
        )

        ab_winner = ab.winner
        position_agreed = ab_winner == ba_winner_flipped

        if ab_winner == ba_winner_flipped:
            final_winner = ab_winner
        elif ab_winner == "Tie" or ba_winner_flipped == "Tie":
            final_winner = ab_winner if ab_winner != "Tie" else ba_winner_flipped
            position_agreed = False
        else:
            final_winner = "Tie"
            position_agreed = False

        combined_analysis = (
            f"=== Run AB ===\n{ab.analysis}\n\n"
            f"=== Run BA (swapped) ===\n{ba.analysis}\n\n"
            f"=== Reconciliation ===\n"
            f"AB winner: {ab.winner}, BA winner (flipped): {ba_winner_flipped}\n"
            f"Position agreement: {'Yes' if position_agreed else 'No'}\n"
            f"Final winner: {final_winner}"
        )

        return (
            JudgeCompareStructuredResult(
                analysis=combined_analysis,
                scores_a=avg_scores_a,
                scores_b=avg_scores_b,
                winner=final_winner,
            ),
            position_agreed,
        )
