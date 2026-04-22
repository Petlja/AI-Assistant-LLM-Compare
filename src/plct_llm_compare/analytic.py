"""Reusable analytics helpers for survey-response notebooks.

This module collects the generic data-loading, flattening, summarization
and GPT-based open-question analysis logic used by the
`eval/survey-response-analysis-notebook.py` notebook so the notebook itself
stays small and focused on ad-hoc exploration.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

__all__ = [
    "LABEL_TRANSLATIONS",
    "DEFAULT_OPEN_QUESTION_ANALYSIS_MODEL",
    "SurveyMetadata",
    "OpenQuestionTheme",
    "OpenQuestionAnswerClassification",
    "OpenQuestionAnalysisResult",
    "OpenQuestionPerModelEntry",
    "OpenQuestionPerModelAnswerClassification",
    "OpenQuestionPerModelAnalysisResult",
    "find_repo_root",
    "translate_label",
    "parse_question_key",
    "is_empty_value",
    "normalize_choice_label",
    "build_survey_metadata",
    "apply_survey_labels",
    "flatten_answers",
    "summarize_numeric",
    "summarize_choices",
    "collect_open_text",
    "get_question_choice_labels",
    "build_model_alias_map",
    "make_strict_schema",
    "open_question_analysis_response_format",
    "open_question_per_model_analysis_response_format",
    "get_openai_client",
    "safe_file_token",
    "build_open_question_analysis_markdown",
    "build_open_question_theme_frame",
    "analyze_open_question_group",
    "analyze_open_question_group_per_model",
]


# ---------------------------------------------------------------------------
# Constants and translation tables
# ---------------------------------------------------------------------------

LABEL_TRANSLATIONS: dict[str, str] = {
    "Koji LLM je dao bolji odgovor u pogledu sledećeg?": "Which LLM gave the better answer for the following?",
    "U kojoj meri se slažeš sa sledećim tvrđenjima": "To what extent do you agree with the following statements?",
    "Vaš ukupan utisak o jeziku odgovora": "Your overall impression of the response language",
    "Šta smatrate da je bolje ili lošije u odgovorima jednog ili drugog LLM-a?": "What do you consider better or worse in one LLM's answers versus the other's?",
    "Šta uočavate da bi trebalo ispravnije jezički formulisati?": "What do you notice should be phrased more correctly linguistically?",
    "Bez izmena ili nakon menjih korekcija, jezik odgovora je dovoljno dobar": "With no changes or after minor corrections, the response language is good enough",
    "Izbor termina": "Choice of terms",
    "Izbor termina u odgovoru je adekvatan": "The choice of terms in the response is appropriate",
    "Korisnost za nastavnu praksu": "Usefulness for teaching practice",
    "Odgovor je koristan za nastavnu praksu": "The response is useful for teaching practice",
    "Prirodnost srpskog jezika": "Naturalness of Serbian language",
    "Srpski jezik u odgovoru zvuči prirodno": "The Serbian language in the response sounds natural",
    "Ukupan utisak": "Overall impression",
    "Bolji je Qwen3-14B": "Qwen3-14B is better",
    "Bolji je gpt-5.2": "gpt-5.2 is better",
    "Ne slažem se": "Disagree",
    "Nema velike razlike": "No major difference",
    "Neodlučan sam": "Undecided",
    "Potpuno se slažem": "Strongly agree",
    "Slažem se": "Agree",
    "Uopšte se ne slažem": "Strongly disagree",
}


DEFAULT_OPEN_QUESTION_ANALYSIS_MODEL = "gpt-5.4"


# ---------------------------------------------------------------------------
# Survey metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurveyMetadata:
    """Lookup tables derived from a SurveyJS survey definition."""

    question_titles: dict[str, str]
    row_labels: dict[tuple[str, str], str]
    choice_labels: dict[tuple[str, str], str]


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward from ``start`` looking for a ``pyproject.toml`` file."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return current


def translate_label(text: object) -> object:
    if text is None:
        return text
    return LABEL_TRANSLATIONS.get(str(text), str(text))


def parse_question_key(question_key: str) -> dict[str, object]:
    prefix, question_id = question_key.rsplit("__", 1)
    case_key, take_text, model = prefix.split("__", 2)
    return {
        "case_key": case_key,
        "take": int(take_text),
        "model": model,
        "question_id": question_id,
    }


def is_empty_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, dict):
        return not value
    return False


def normalize_choice_label(choice_value: str, choice_text: str) -> str:
    if choice_text in {"Bolji je LLM A", "Bolji je LLM B"}:
        return str(translate_label(f"Bolji je {choice_value}"))
    return str(translate_label(choice_text))


def build_survey_metadata(survey: dict) -> SurveyMetadata:
    question_titles: dict[str, str] = {}
    row_labels: dict[tuple[str, str], str] = {}
    choice_labels: dict[tuple[str, str], str] = {}

    for page in survey.get("pages", []):
        for element in page.get("elements", []):
            element_name = element.get("name")
            if not element_name or "__q" not in element_name:
                continue

            parsed = parse_question_key(element_name)
            question_id = parsed["question_id"]
            question_titles.setdefault(
                question_id,
                str(translate_label(element.get("title", question_id))),
            )

            for row in element.get("rows", []):
                row_value = str(row.get("value", ""))
                row_labels.setdefault(
                    (question_id, row_value),
                    str(translate_label(row.get("text", row_value))),
                )

            for column in element.get("columns", []):
                column_value = str(column.get("value", ""))
                column_text = str(column.get("text", column_value))
                choice_labels.setdefault(
                    (question_id, column_value),
                    normalize_choice_label(column_value, column_text),
                )

    return SurveyMetadata(question_titles, row_labels, choice_labels)


def apply_survey_labels(frame: pd.DataFrame, metadata: SurveyMetadata) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    labeled = frame.copy()
    labeled["question_title"] = (
        labeled["question_id"]
        .map(metadata.question_titles)
        .fillna(labeled["question_id"].map(translate_label))
        .fillna(labeled["question_id"])
    )

    if "row_key" in labeled.columns:
        labeled["row_label"] = labeled.apply(
            lambda row: metadata.row_labels.get(
                (row["question_id"], str(row["row_key"])),
                translate_label(row.get("row_key")),
            ),
            axis=1,
        )

    if "choice" in labeled.columns:
        labeled["choice_label"] = labeled.apply(
            lambda row: metadata.choice_labels.get(
                (row["question_id"], str(row["choice"])),
                translate_label(row.get("choice")),
            ),
            axis=1,
        )

    return labeled


def get_question_choice_labels(survey: dict, question_id: str) -> list[str]:
    """Return ordered, translated choice labels for ``question_id``."""
    labels: list[str] = []
    for page in survey.get("pages", []):
        for element in page.get("elements", []):
            element_name = element.get("name")
            if not element_name or "__q" not in element_name:
                continue
            parsed = parse_question_key(element_name)
            if parsed["question_id"] != question_id:
                continue
            for column in element.get("columns", []):
                column_value = str(column.get("value", ""))
                column_text = str(column.get("text", column_value))
                normalized = normalize_choice_label(column_value, column_text)
                if normalized not in labels:
                    labels.append(normalized)
            if labels:
                return labels
    return labels


def build_model_alias_map(survey: dict) -> dict[str, str]:
    """Build an alias -> model-name map from the q4 comparison columns.

    The survey uses rating columns like ``{"value": "<model_short>", "text": "Bolji je LLM A"}``.
    Returns a mapping like ``{"A": "Qwen--Qwen3-14B", "B": "gpt-5.2"}``. Aliases from the
    column text are taken as-is; model values are the short names stored on the column.
    """
    import re

    pattern = re.compile(r"Bolji\s+je\s+LLM\s+([A-Z])", re.IGNORECASE)
    mapping: dict[str, str] = {}
    for page in survey.get("pages", []):
        for element in page.get("elements", []):
            element_name = element.get("name")
            if not element_name or "__q" not in element_name:
                continue
            if parse_question_key(element_name)["question_id"] != "q4":
                continue
            for column in element.get("columns", []):
                column_text = str(column.get("text", ""))
                column_value = str(column.get("value", ""))
                match = pattern.search(column_text)
                if match and column_value:
                    alias = match.group(1).upper()
                    mapping.setdefault(alias, column_value)
    return mapping


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def flatten_answers(response_entries: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for entry in response_entries:
        label = entry.get("label")
        submitted_at = entry.get("submitted_at")
        for question_key, value in entry.get("answers", {}).items():
            parsed = parse_question_key(question_key)
            if is_empty_value(value):
                continue

            if isinstance(value, dict):
                for row_key, row_value in value.items():
                    if is_empty_value(row_value):
                        continue
                    rows.append(
                        {
                            **parsed,
                            "question_key": question_key,
                            "label": label,
                            "submitted_at": submitted_at,
                            "row_key": row_key,
                            "value": row_value,
                        }
                    )
            else:
                rows.append(
                    {
                        **parsed,
                        "question_key": question_key,
                        "label": label,
                        "submitted_at": submitted_at,
                        "row_key": None,
                        "value": value,
                    }
                )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame["numeric_value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["is_numeric"] = frame["numeric_value"].notna()
    return frame


def summarize_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame[frame["is_numeric"]].copy()
    if numeric.empty:
        return pd.DataFrame(
            columns=[
                "question_id",
                "take",
                "model",
                "row_key",
                "response_count",
                "case_count",
                "mean",
                "min",
                "max",
            ]
        )

    summary = (
        numeric.groupby(["question_id", "take", "model", "row_key"], dropna=False)
        .agg(
            response_count=("numeric_value", "count"),
            case_count=("case_key", "nunique"),
            mean=("numeric_value", "mean"),
            min=("numeric_value", "min"),
            max=("numeric_value", "max"),
        )
        .reset_index()
        .sort_values(["question_id", "take", "model", "row_key"], na_position="last")
    )
    summary["mean"] = summary["mean"].round(3)
    return summary


def summarize_choices(frame: pd.DataFrame) -> pd.DataFrame:
    choices = frame[
        (~frame["is_numeric"])
        & frame["row_key"].notna()
        & frame["value"].map(lambda value: isinstance(value, str))
    ].copy()
    if choices.empty:
        return pd.DataFrame(
            columns=[
                "question_id",
                "take",
                "model",
                "row_key",
                "choice",
                "count",
                "share",
                "case_count",
            ]
        )

    choices["choice"] = choices["value"].str.strip()
    summary = (
        choices.groupby(["question_id", "take", "model", "row_key", "choice"], dropna=False)
        .agg(
            count=("label", "count"),
            case_count=("case_key", "nunique"),
        )
        .reset_index()
        .sort_values(["question_id", "take", "model", "row_key", "choice"], na_position="last")
    )
    totals = summary.groupby(["question_id", "take", "model", "row_key"], dropna=False)["count"].transform("sum")
    summary["share"] = (summary["count"] / totals).round(3)
    return summary


def collect_open_text(frame: pd.DataFrame) -> pd.DataFrame:
    text = frame[
        (~frame["is_numeric"])
        & frame["row_key"].isna()
        & frame["value"].map(lambda value: isinstance(value, str))
    ].copy()
    if text.empty:
        return pd.DataFrame(
            columns=[
                "question_id",
                "take",
                "model",
                "case_key",
                "label",
                "submitted_at",
                "text",
            ]
        )

    text["text"] = text["value"].str.strip()
    text = text[text["text"] != ""]
    return text[
        ["question_id", "take", "model", "case_key", "label", "submitted_at", "text"]
    ].sort_values(
        ["question_id", "take", "model", "case_key", "label"],
        na_position="last",
    )


# ---------------------------------------------------------------------------
# Open-question GPT analysis
# ---------------------------------------------------------------------------


class OpenQuestionTheme(BaseModel):
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)


class OpenQuestionAnswerClassification(BaseModel):
    answer_id: str = Field(min_length=1)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


class OpenQuestionAnalysisResult(BaseModel):
    overall_summary: str = Field(min_length=1)
    key_strengths: list[OpenQuestionTheme] = Field(default_factory=list)
    key_weaknesses: list[OpenQuestionTheme] = Field(default_factory=list)
    classifications: list[OpenQuestionAnswerClassification] = Field(default_factory=list)


def make_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    schema = dict(schema)
    if "$ref" in schema:
        return {"$ref": schema["$ref"]}
    schema.pop("default", None)
    if schema.get("type") == "object":
        schema["additionalProperties"] = False
    for key in ("properties", "definitions", "$defs"):
        if key in schema:
            schema[key] = {name: make_strict_schema(value) for name, value in schema[key].items()}
    if schema.get("type") == "object" and "properties" in schema:
        schema["required"] = list(schema["properties"].keys())
    if "items" in schema:
        schema["items"] = make_strict_schema(schema["items"])
    return schema


def open_question_analysis_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "open_question_analysis_result",
            "strict": True,
            "schema": make_strict_schema(OpenQuestionAnalysisResult.model_json_schema()),
        },
    }


def get_openai_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def safe_file_token(value: object) -> str:
    return str(value).strip().replace("/", "--").replace(" ", "-")


def _open_question_cache_path(
    *,
    cache_dir: Path,
    question_id: str,
    take: int,
    analysis_model: str,
    system_prompt: str,
    user_prompt: str,
) -> Path:
    """Compute a deterministic cache file path for an open-question analysis call.

    The file name embeds question_id, take and analysis model for readability, plus
    a short hash of the prompts so changes to inputs invalidate the cache naturally.
    Each call gets its own file, so deleting a single file forces re-run of just
    that one analysis.
    """
    payload = json.dumps(
        {
            "model": analysis_model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:12]
    model_safe = safe_file_token(analysis_model)
    filename = f"gpt-cache-{question_id}-take-{take}-{model_safe}-{digest}.json"
    return cache_dir / filename


def build_open_question_analysis_markdown(
    question_id: str,
    question_title: str,
    take: int,
    analysis_result: OpenQuestionAnalysisResult,
    analysis_model: str = DEFAULT_OPEN_QUESTION_ANALYSIS_MODEL,
) -> str:
    lines = [
        f"# {question_id} | {question_title} | take {take}",
        "",
        f"Analysis model: {analysis_model}",
        "",
        "## Overall summary",
        analysis_result.overall_summary,
        "",
        "## Key strengths",
    ]
    if analysis_result.key_strengths:
        lines.extend(f"- {item.label}: {item.description}" for item in analysis_result.key_strengths)
    else:
        lines.append("- None identified")

    lines.extend(["", "## Key weaknesses"])
    if analysis_result.key_weaknesses:
        lines.extend(f"- {item.label}: {item.description}" for item in analysis_result.key_weaknesses)
    else:
        lines.append("- None identified")

    return "\n".join(lines)


def build_open_question_theme_frame(analysis_result: OpenQuestionAnalysisResult) -> pd.DataFrame:
    rows = [
        {"theme_type": "strength", "label": item.label, "description": item.description}
        for item in analysis_result.key_strengths
    ]
    rows.extend(
        {"theme_type": "weakness", "label": item.label, "description": item.description}
        for item in analysis_result.key_weaknesses
    )
    return pd.DataFrame(rows)


def analyze_open_question_group(
    client: OpenAI,
    *,
    question_id: str,
    question_title: str,
    take: int,
    export_frame: pd.DataFrame,
    analysis_model: str = DEFAULT_OPEN_QUESTION_ANALYSIS_MODEL,
    cache_dir: Path | str | None = None,
) -> tuple[OpenQuestionAnalysisResult, pd.DataFrame]:
    prompt_frame = export_frame.reset_index(drop=True).copy()
    prompt_frame.insert(0, "answer_id", [f"a{index + 1:03d}" for index in range(len(prompt_frame))])
    response_payload = prompt_frame[
        ["answer_id", "case_key", "model", "respondent", "answer_text"]
    ].to_dict(orient="records")

    system_prompt = (
        "You analyze open-ended survey feedback about LLM answers. "
        "Identify recurring strong and weak points across the full answer set, then classify each answer using only those recurring labels. "
        "Keep labels and descriptions in English even if the original answers are in Serbian."
    )
    user_prompt = (
        f"Survey question ID: {question_id}\n"
        f"Survey question title: {question_title}\n"
        f"Take: {take}\n\n"
        "Task:\n"
        "1. Read all answers together.\n"
        "2. Derive 3 to 7 recurring strengths and 3 to 7 recurring weaknesses that appear across the answer set.\n"
        "3. For each answer, assign zero or more labels from the derived strengths and weaknesses lists.\n"
        "4. Add a short rationale grounded in that answer text.\n\n"
        "Rules:\n"
        "- Do not invent answer-specific labels outside the shared key strengths and key weaknesses lists.\n"
        "- Use only evidence present in the answer text.\n"
        "- If an answer does not support a strength or weakness, leave that list empty.\n"
        "- Keep labels short noun phrases.\n"
        "- Keep rationales concise and specific.\n\n"
        "Answers:\n"
        f"{json.dumps(response_payload, ensure_ascii=False, indent=2)}"
    )

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = _open_question_cache_path(
            cache_dir=Path(cache_dir),
            question_id=question_id,
            take=take,
            analysis_model=analysis_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    content: str | None = None
    if cache_path is not None and cache_path.exists():
        content = cache_path.read_text(encoding="utf-8").strip()
        if not content:
            content = None

    if content is None:
        completion = client.chat.completions.create(
            model=analysis_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=5000,
            temperature=0.0,
            response_format=open_question_analysis_response_format(),
        )

        message = completion.choices[0].message
        content = (message.content or "").strip()
        if not content:
            refusal = getattr(message, "refusal", None)
            raise RuntimeError(f"GPT analysis returned empty content (refusal={refusal!r})")

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(content, encoding="utf-8")

    analysis_result = OpenQuestionAnalysisResult.model_validate_json(content)
    classifications_by_answer = {item.answer_id: item for item in analysis_result.classifications}
    expected_ids = set(prompt_frame["answer_id"])
    returned_ids = set(classifications_by_answer)
    if expected_ids != returned_ids:
        missing_ids = sorted(expected_ids - returned_ids)
        extra_ids = sorted(returned_ids - expected_ids)
        raise RuntimeError(
            "GPT analysis classification IDs do not match answers. "
            f"Missing: {missing_ids}; extra: {extra_ids}"
        )

    classified_frame = prompt_frame.copy()
    classified_frame["strength_labels"] = classified_frame["answer_id"].map(
        lambda answer_id: "; ".join(classifications_by_answer[answer_id].strengths)
    )
    classified_frame["weakness_labels"] = classified_frame["answer_id"].map(
        lambda answer_id: "; ".join(classifications_by_answer[answer_id].weaknesses)
    )
    classified_frame["classification_rationale"] = classified_frame["answer_id"].map(
        lambda answer_id: classifications_by_answer[answer_id].rationale
    )
    return analysis_result, classified_frame


# ---------------------------------------------------------------------------
# Per-model variant (for questions like q5 comparing multiple LLMs in one text)
# ---------------------------------------------------------------------------


class OpenQuestionPerModelEntry(BaseModel):
    model: str = Field(min_length=1)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


class OpenQuestionPerModelAnswerClassification(BaseModel):
    answer_id: str = Field(min_length=1)
    per_model: list[OpenQuestionPerModelEntry] = Field(default_factory=list)


class OpenQuestionPerModelAnalysisResult(BaseModel):
    overall_summary: str = Field(min_length=1)
    key_strengths: list[OpenQuestionTheme] = Field(default_factory=list)
    key_weaknesses: list[OpenQuestionTheme] = Field(default_factory=list)
    classifications: list[OpenQuestionPerModelAnswerClassification] = Field(default_factory=list)


def open_question_per_model_analysis_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "open_question_per_model_analysis_result",
            "strict": True,
            "schema": make_strict_schema(OpenQuestionPerModelAnalysisResult.model_json_schema()),
        },
    }


def analyze_open_question_group_per_model(
    client: OpenAI,
    *,
    question_id: str,
    question_title: str,
    take: int,
    export_frame: pd.DataFrame,
    known_models: list[str] | None = None,
    model_alias_map: dict[str, str] | None = None,
    analysis_model: str = DEFAULT_OPEN_QUESTION_ANALYSIS_MODEL,
    cache_dir: Path | str | None = None,
) -> tuple[OpenQuestionPerModelAnalysisResult, pd.DataFrame]:
    """Analyze an open question that compares multiple models in one free-form text.

    Unlike ``analyze_open_question_group``, each answer can map to multiple per-model
    entries. The returned ``classified_frame`` has one row per (answer, model
    discussed), so strengths/weaknesses are attributed to a specific model.

    ``model_alias_map`` maps free-form aliases used by respondents (e.g. ``"A"``,
    ``"B"``) to canonical model names. When provided, the alias map is included in
    the prompt and the resulting ``model_discussed`` column is normalized to the
    canonical model names.
    """
    prompt_frame = export_frame.reset_index(drop=True).copy()
    prompt_frame.insert(0, "answer_id", [f"a{index + 1:03d}" for index in range(len(prompt_frame))])
    response_payload = prompt_frame[
        ["answer_id", "case_key", "model", "respondent", "answer_text"]
    ].to_dict(orient="records")

    system_prompt = (
        "You analyze open-ended survey feedback that compares multiple LLM answers in a single free-form text. "
        "Identify recurring strong and weak points across the full answer set, then for each answer attribute "
        "strengths and weaknesses to the specific model(s) being discussed. "
        "Keep labels and descriptions in English even if the original answers are in Serbian."
    )

    known_models_section = ""
    if known_models:
        known_models_section = (
            "Use only these canonical model names when attributing strengths/weaknesses to a model:\n"
            f"{json.dumps(known_models, ensure_ascii=False)}\n\n"
        )

    alias_section = ""
    if model_alias_map:
        alias_lines = [f"- {alias} -> {name}" for alias, name in sorted(model_alias_map.items())]
        alias_section = (
            "Respondents refer to models using these aliases. Resolve each alias to its canonical "
            "model name and use the canonical name in the ``model`` field of per-model entries:\n"
            + "\n".join(alias_lines)
            + "\n\n"
        )

    user_prompt = (
        f"Survey question ID: {question_id}\n"
        f"Survey question title: {question_title}\n"
        f"Take: {take}\n\n"
        f"{known_models_section}"
        f"{alias_section}"
        "Task:\n"
        "1. Read all answers together.\n"
        "2. Derive 3 to 7 recurring strengths and 3 to 7 recurring weaknesses that appear across the answer set.\n"
        "3. For each answer, produce a list of per-model entries. Each entry covers one model the answer discusses, "
        "with strengths and weaknesses from the derived lists attributed to that specific model, plus a short rationale.\n"
        "4. If an answer discusses only one model, produce a single entry. If it discusses several, produce one entry per model.\n\n"
        "Rules:\n"
        "- Do not invent labels outside the shared key strengths and key weaknesses lists.\n"
        "- Use only evidence present in the answer text.\n"
        "- If the answer does not attribute strengths or weaknesses to a given model, leave those lists empty for that entry.\n"
        "- Keep labels short noun phrases.\n"
        "- Keep rationales concise and specific.\n"
        "- If no model can be identified in an answer, use \"unknown\" as the model name.\n\n"
        "Answers:\n"
        f"{json.dumps(response_payload, ensure_ascii=False, indent=2)}"
    )

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = _open_question_cache_path(
            cache_dir=Path(cache_dir),
            question_id=f"{question_id}-per-model",
            take=take,
            analysis_model=analysis_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    content: str | None = None
    if cache_path is not None and cache_path.exists():
        content = cache_path.read_text(encoding="utf-8").strip()
        if not content:
            content = None

    if content is None:
        completion = client.chat.completions.create(
            model=analysis_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=6000,
            temperature=0.0,
            response_format=open_question_per_model_analysis_response_format(),
        )

        message = completion.choices[0].message
        content = (message.content or "").strip()
        if not content:
            refusal = getattr(message, "refusal", None)
            raise RuntimeError(f"GPT analysis returned empty content (refusal={refusal!r})")

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(content, encoding="utf-8")

    analysis_result = OpenQuestionPerModelAnalysisResult.model_validate_json(content)
    classifications_by_answer = {item.answer_id: item for item in analysis_result.classifications}
    expected_ids = set(prompt_frame["answer_id"])
    returned_ids = set(classifications_by_answer)
    if expected_ids != returned_ids:
        missing_ids = sorted(expected_ids - returned_ids)
        extra_ids = sorted(returned_ids - expected_ids)
        raise RuntimeError(
            "GPT per-model analysis classification IDs do not match answers. "
            f"Missing: {missing_ids}; extra: {extra_ids}"
        )

    expanded_rows: list[dict[str, Any]] = []
    base_columns = [col for col in prompt_frame.columns]

    def _resolve_model(raw_model: str) -> str:
        if not model_alias_map:
            return raw_model
        stripped = raw_model.strip()
        if stripped in model_alias_map:
            return model_alias_map[stripped]
        upper = stripped.upper()
        if upper in model_alias_map:
            return model_alias_map[upper]
        return raw_model

    for row in prompt_frame.to_dict(orient="records"):
        answer_id = row["answer_id"]
        entries = classifications_by_answer[answer_id].per_model
        if not entries:
            expanded_rows.append(
                {
                    **{col: row[col] for col in base_columns},
                    "model_discussed": "",
                    "strength_labels": "",
                    "weakness_labels": "",
                    "classification_rationale": "",
                }
            )
            continue
        for entry in entries:
            expanded_rows.append(
                {
                    **{col: row[col] for col in base_columns},
                    "model_discussed": _resolve_model(entry.model),
                    "strength_labels": "; ".join(entry.strengths),
                    "weakness_labels": "; ".join(entry.weaknesses),
                    "classification_rationale": entry.rationale,
                }
            )

    classified_frame = pd.DataFrame(expanded_rows)
    return analysis_result, classified_frame
