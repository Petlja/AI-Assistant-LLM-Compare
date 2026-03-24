"""Data models for LLM comparison."""

from pydantic import BaseModel


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
    judge_response: str
