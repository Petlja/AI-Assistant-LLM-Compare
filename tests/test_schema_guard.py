"""Guard: the judge's response-format schema must never accept "Inconsistent".

"Inconsistent" exists only on the reconciled result; the LLM judge itself must
be constrained to A/B/Tie.
"""

from plct_llm_compare.judge_compare import _judge_response_format


def test_judge_winner_enum_unchanged():
    response_format = _judge_response_format()
    schema = response_format["json_schema"]["schema"]
    assert schema["properties"]["winner"]["enum"] == ["A", "B", "Tie"]
