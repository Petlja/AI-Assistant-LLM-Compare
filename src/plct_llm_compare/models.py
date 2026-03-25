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


class JudgeCompareStructuredResult(BaseModel):
    """Structured judge evaluation for comparing two model answers."""

    rationale: list[str] = Field(
        min_length=3,
        max_length=6,
        description="Three to six concise comparison bullets describing important strengths and weaknesses across both answers.",
    )
    improvement_suggestions_a: list[str] = Field(
        min_length=1,
        max_length=3,
        description="One to three concrete suggestions that would improve Answer A.",
    )
    improvement_suggestions_b: list[str] = Field(
        min_length=1,
        max_length=3,
        description="One to three concrete suggestions that would improve Answer B.",
    )
    scores_a: JudgeCategoryScores
    scores_b: JudgeCategoryScores
    winner: Literal["A", "B", "Tie"] = Field(
        description="Final winner chosen only after comparing both answers and assigning scores. Use A, B, or Tie.",
    )
