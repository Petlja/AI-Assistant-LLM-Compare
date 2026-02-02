"""Data models for LLM comparison."""

from pydantic import BaseModel


class TestCase(BaseModel):
    """A single test case for LLM comparison."""

    case_key: str
    course_key: str
    activity_key: str
    prompt: str
    system_message: str = None
