"""Data models for LLM comparison."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


def safe_model_name(model: str) -> str:
    """Make a model name safe for use in file names (e.g. Qwen/Qwen3-14B -> Qwen--Qwen3-14B)."""
    return model.replace("/", "--")


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
    temperature: float | None = None


class TestCaseJudgeCompareResult(BaseModel):
    """Metadata for a judge-compare run, saved alongside the judge HTML output."""

    case_key: str
    activity_url: str
    activity_desc: str
    prompt: str
    model_a: str
    model_b: str
    judge_model: str
    take_a: int
    take_b: int
    answer_a: str
    answer_b: str
    judge_result: "JudgeCompareReconciledResult"
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
    inconsistent_count: int = 0  # legacy result files only
    order_flip_tie_count: int = 0
    position_agree_count: int = 0
    count: int = 0

    def update(
        self,
        judge_result: "JudgeCompareReconciledResult",
        position_agreed: bool,
    ) -> None:
        """Update cumulative scores and winner counts based on a new judge result."""
        self.score_a_sum += judge_result.score_a
        self.score_b_sum += judge_result.score_b

        if judge_result.winner == "A":
            self.winner_a_count += 1
        elif judge_result.winner == "B":
            self.winner_b_count += 1
        elif judge_result.winner == "Inconsistent":
            # Legacy files only; current runs never take this branch.
            self.inconsistent_count += 1
        else:
            self.no_winner_count += 1

        if position_agreed:
            self.position_agree_count += 1
        elif judge_result.winner == "Tie":
            # A tie the judge only reached by contradicting itself across the
            # two orderings. Counted separately so a summary can distinguish
            # "no preference" from "no consistency".
            self.order_flip_tie_count += 1

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


class JudgeCompareReconciledResult(JudgeCompareStructuredResult):
    """Reconciled AB+BA verdict.

    Never used as an LLM response format — only the base class's schema is sent
    to the judge, so the judge itself never produces this class's verdict.

    "Inconsistent" is retained in the Literal for one reason only: judge result
    files written before 2026-09-10 contain it, and they must still parse. No
    new run produces it — an order-flip is now a Tie. See `reconcile`.
    """

    winner: Literal["A", "B", "Tie", "Inconsistent"] = Field(
        description=(
            "Final verdict across both orderings: A, B, or Tie. Tie covers both "
            "a genuine draw and the two runs disagreeing after un-swapping; "
            "`position_agreed` tells the two apart. "
            '"Inconsistent" is legacy, read-only.'
        ),
    )

    @staticmethod
    def reconcile(
        ab: JudgeCompareStructuredResult,
        ba: JudgeCompareStructuredResult,
    ) -> tuple["JudgeCompareReconciledResult", bool]:
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

        # Half-up integer average; round() would round half to even (banker's
        # rounding) and systematically bias .5 averages downward.
        avg_score_a = (ab.score_a + ba_score_a + 1) // 2
        avg_score_b = (ab.score_b + ba_score_b + 1) // 2

        position_agreed = ab.winner == ba_winner_flipped

        if position_agreed:
            final_winner = ab.winner
        else:
            # A judge that says A one way and B the other has expressed no
            # preference, so the verdict is a Tie — and a Tie can be compared
            # against a human's Tie, which "Inconsistent" never could.
            #
            # The order-bias signal is NOT lost: `position_agreed` is returned
            # here and written per case to judge_results.yml, so the rate is
            # still computable. Read it — a judge that ties this way often is
            # reacting to order, not judging.
            final_winner = "Tie"

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
            JudgeCompareReconciledResult(
                analysis=combined_analysis,
                score_a=avg_score_a,
                score_b=avg_score_b,
                winner=final_winner,
            ),
            position_agreed,
        )


class HumanEvalAnnotation(BaseModel):
    """One case in the annotator-facing combined YAML file.

    Deliberately carries no model/take/swap metadata — the annotator must stay
    blind to which side is which; that mapping lives in the assignment file.
    """

    id: str
    prompt: str
    answer_a: str
    answer_b: str
    # Plain str, not Literal: hand-edited YAML must degrade to a warning at
    # read time, never a validation crash.
    human_verdict: str = ""
    human_notes: str = ""
    activity_url: str
    activity_desc: str

    @field_validator("human_verdict", "human_notes", mode="before")
    @classmethod
    def _tolerate_hand_edits(cls, value):
        # An annotator may clear a value (YAML null) or type an unquoted
        # scalar YAML parses as bool/int — never crash on that.
        return "" if value is None else str(value)


class HumanEvalAnnotationsFile(BaseModel):
    """Schema of the combined human_feedback.yml handed to human annotators.

    `pair_id` is an opaque digest, not the pair directory name: the directory
    names both models, and an annotator who knows the two models can start
    recognising one by style instead of judging the answer. assignment.yml
    records the same id against the real names.
    """

    # Defaulted so a file written before this field existed (it carried
    # `pair_dir`, which pydantic now ignores) still parses — those files may
    # hold real annotations and must never fail to load.
    pair_id: str = ""
    cases: list[HumanEvalAnnotation]


class HumanEvalCaseAssignment(BaseModel):
    """Answer key for one case: what the displayed A/B actually were.

    model_a/take_a describe the answer DISPLAYED as "A" after the blind
    shuffle; swapped=True means displayed A is the pair's canonical B as
    given on the command line.
    """

    id: str
    swapped: bool
    model_a: str
    take_a: int
    model_b: str
    take_b: int


class HumanEvalAssignmentsFile(BaseModel):
    """Schema of assignment.yml — the answer key kept away from annotators."""

    pair_id: str
    model_a: str
    take_a: int
    model_b: str
    take_b: int
    shuffled: bool
    seed: int
    cases: list[HumanEvalCaseAssignment]


class CalibrationCaseResult(BaseModel):
    """One case in eval_answers.yml: what the pointwise scorer said.

    In DISPLAYED frame — `score_a` is the score of whatever the annotator saw
    as answer A. That is what makes this file readable side by side with
    human_feedback.yml, which is the whole point: the comparison is done by
    hand. `swapped` is carried inline so the row is self-describing without
    cross-referencing assignment.yml.
    """

    id: str
    swapped: bool
    score_a: int
    score_b: int
    verdict: Literal["A", "B", "Tie"]


class CalibrationResultsFile(BaseModel):
    """Schema of eval_answers.yml — the scorer's side of the comparison.

    Raw scores are always written, so the tie band can be re-derived by hand at
    a different epsilon without spending another run.
    """

    frame: Literal["displayed"] = "displayed"
    scorer: str
    scale_max: int
    tie_band: int
    model_a: str
    take_a: int
    model_b: str
    take_b: int
    cases: list[CalibrationCaseResult]
