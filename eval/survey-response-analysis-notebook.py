# %% [markdown]
# # Survey Response Analysis
#
# This notebook loads the saved survey responses, aggregates results across all cases, and presents them as tables and charts.
#
# It focuses on three output views:
#
# - numeric ratings such as `q1`
# - categorical comparison answers such as `q4`
# - open-ended responses such as `q2`, `q3`, and `q5`
#
# Generic helpers (parsing, aggregation, GPT-based open-question analysis) live in
# `plct_llm_compare.analytic` so this notebook stays small and easy to tweak ad-hoc.

# %% [markdown]
# ## 1. Load Input Data
#
# Import the required libraries and locate the response file in a way that works whether the notebook is started from the repository root or the `eval` directory.

# %%
import json
from pathlib import Path

import pandas as pd
from IPython.display import display
from pydantic import ValidationError

from plct_llm_compare.analytic import (
    DEFAULT_OPEN_QUESTION_ANALYSIS_MODEL,
    analyze_open_question_group,
    analyze_open_question_group_per_model,
    apply_survey_labels,
    build_model_alias_map,
    build_open_question_analysis_markdown,
    build_open_question_theme_frame,
    build_survey_metadata,
    collect_open_text,
    find_repo_root,
    flatten_answers,
    get_openai_client,
    get_question_choice_labels,
    safe_file_token,
    summarize_choices,
    summarize_numeric,
)

OPEN_QUESTION_ANALYSIS_MODEL = DEFAULT_OPEN_QUESTION_ANALYSIS_MODEL

repo_root = find_repo_root()
response_candidates = [
    repo_root / "eval" / "survey-responces.json",
    repo_root / "survey-responces.json",
    Path.cwd() / "survey-responces.json",
]
responses_path = next((path for path in response_candidates if path.exists()), None)
if responses_path is None:
    raise FileNotFoundError("Could not find eval/survey-responces.json")

survey_candidates = [
    repo_root / "eval" / "output" / "survey.json",
    repo_root / "output" / "survey.json",
    Path.cwd() / "output" / "survey.json",
]
survey_path = next((path for path in survey_candidates if path.exists()), None)
if survey_path is None:
    raise FileNotFoundError("Could not find eval/output/survey.json")

with responses_path.open(encoding="utf-8") as handle:
    responses = json.load(handle)

with survey_path.open(encoding="utf-8") as handle:
    survey_definition = json.load(handle)

print(f"Loaded {len(responses)} respondent entries from {responses_path}")
print(f"Loaded survey definition from {survey_path}")
pd.set_option("display.max_colwidth", 160)
pd.set_option("display.max_rows", 200)

survey_metadata = build_survey_metadata(survey_definition)
model_alias_map = build_model_alias_map(survey_definition)
print(f"Model alias map: {model_alias_map}")

# %% [markdown]
# ## 2. Run Core Calculations
#
# Create the flattened answer table and derive the main cross-case result tables.

# %%
answers_df = apply_survey_labels(flatten_answers(responses), survey_metadata)
answers_df = answers_df[answers_df["label"].str.match(r"^s\d+$", na=False)]
numeric_stats = apply_survey_labels(summarize_numeric(answers_df), survey_metadata)
choice_stats = apply_survey_labels(summarize_choices(answers_df), survey_metadata)
open_text = apply_survey_labels(collect_open_text(answers_df), survey_metadata)

question_completion = (
    answers_df.groupby(["question_id", "take", "model"], dropna=False)
    .agg(
        answer_rows=("value", "count"),
        respondent_count=("label", "nunique"),
        case_count=("case_key", "nunique"),
    )
    .reset_index()
    .sort_values(["question_id", "take", "model"], na_position="last")
)
question_completion["question_title"] = (
    question_completion["question_id"]
    .map(survey_metadata.question_titles)
    .fillna(question_completion["question_id"])
)

print(f"Flattened rows: {len(answers_df)}")
print(f"Numeric summary rows: {len(numeric_stats)}")
print(f"Choice summary rows: {len(choice_stats)}")
print(f"Open-text rows: {len(open_text)}")

# %% [markdown]
# ## 3. Format Tabular Results
#
# Display the key tables and build a pivoted view for the main numeric ratings so models can be compared side by side.

# %%
respondent_info = (
    answers_df.loc[answers_df["label"].notna(), ["label", "submitted_at"]]
    .drop_duplicates(subset="label")
    .sort_values("submitted_at")
    .reset_index(drop=True)
)
print("Answer labels")
display(respondent_info)
# display(respondent_info.sort_values("label"))

display(question_completion[["question_id", "question_title", "take", "model", "answer_rows", "respondent_count", "case_count"]])

numeric_stats_wide = pd.pivot_table(
    numeric_stats[["question_id", "row_label", "model", "response_count", "case_count", "mean", "min", "max"]],
    index=["question_id", "row_label"],
    columns="model",
    values=["response_count", "case_count", "mean", "min", "max"],
    aggfunc="first",
)

print("Open-ended question responses")
if open_text.empty:
    print("No open-ended responses found.")
else:
    results_dir = repo_root / "eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    open_text_tables = open_text.rename(columns={"label": "respondent", "text": "answer_text"})
    open_question_exports: list[tuple[str, str, int, pd.DataFrame]] = []
    for (question_id, question_title, take), question_frame in open_text_tables.groupby(
        ["question_id", "question_title", "take"],
        sort=False,
        dropna=False,
    ):
        export_frame = (
            question_frame[["case_key", "model", "respondent", "answer_text"]]
            .sort_values(["case_key", "respondent", "model"], na_position="last")
            .reset_index(drop=True)
        )
        export_path = results_dir / f"open-question-{question_id}-take-{take}.csv"
        export_frame.to_csv(export_path, index=False, encoding="utf-8")
        open_question_exports.append((question_id, question_title, take, export_frame))
        print(f"{question_id} | {question_title} | take {take}")
        print(f"Exported {export_path}")
        display(export_frame)

    print(f"Open-question GPT analysis model: {OPEN_QUESTION_ANALYSIS_MODEL}")
    openai_client = get_openai_client()
    classified_by_question: dict[tuple[str, str, int], tuple[bool, pd.DataFrame]] = {}
    if openai_client is None:
        print("Skipping GPT-5.4 analysis because OPENAI_API_KEY is not set.")
    else:
        analysis_model_safe = safe_file_token(OPEN_QUESTION_ANALYSIS_MODEL)
        gpt_cache_dir = results_dir / "gpt-cache"
        gpt_cache_dir.mkdir(parents=True, exist_ok=True)
        for question_id, question_title, take, export_frame in open_question_exports:
            print(f"Analyzing {question_id} | {question_title} | take {take}")
            per_model_question = question_id == "q5"
            try:
                if per_model_question:
                    analysis_result, classified_frame = analyze_open_question_group_per_model(
                        openai_client,
                        question_id=question_id,
                        question_title=question_title,
                        take=take,
                        export_frame=export_frame,
                        model_alias_map=model_alias_map,
                        known_models=sorted(set(model_alias_map.values())) or None,
                        analysis_model=OPEN_QUESTION_ANALYSIS_MODEL,
                        cache_dir=gpt_cache_dir,
                    )
                else:
                    analysis_result, classified_frame = analyze_open_question_group(
                        openai_client,
                        question_id=question_id,
                        question_title=question_title,
                        take=take,
                        export_frame=export_frame,
                        analysis_model=OPEN_QUESTION_ANALYSIS_MODEL,
                        cache_dir=gpt_cache_dir,
                    )
            except (RuntimeError, ValidationError) as exc:
                print(f"GPT-5.4 analysis failed for {question_id} take {take}: {exc}")
                continue

            summary_path = results_dir / (
                f"open-question-{question_id}-take-{take}-{analysis_model_safe}-analysis.md"
            )
            classification_path = results_dir / (
                f"open-question-{question_id}-take-{take}-{analysis_model_safe}-classification.csv"
            )
            summary_path.write_text(
                build_open_question_analysis_markdown(
                    question_id=question_id,
                    question_title=question_title,
                    take=take,
                    analysis_result=analysis_result,
                    analysis_model=OPEN_QUESTION_ANALYSIS_MODEL,
                ),
                encoding="utf-8",
            )
            classified_frame.to_csv(classification_path, index=False, encoding="utf-8")

            print(analysis_result.overall_summary)
            print(f"Saved GPT-5.4 summary to {summary_path}")
            print(f"Saved GPT-5.4 classifications to {classification_path}")
            display(build_open_question_theme_frame(analysis_result))
            display_columns = [
                "case_key",
                "model",
                "respondent",
                "answer_text",
                "strength_labels",
                "weakness_labels",
                "classification_rationale",
            ]
            if per_model_question:
                display_columns.insert(2, "model_discussed")
            display(classified_frame[display_columns])
            classified_by_question[(question_id, question_title, take)] = (
                per_model_question,
                classified_frame,
            )

# %% [markdown]
# ## 3b. Category Statistics for Open Questions
#
# For each open question (q2, q3, q5) tabulate how often each GPT-identified
# strength / weakness label appears across answers, and plot the distribution.
# For q5 the counts are split per model being discussed.

# %%
import matplotlib.pyplot as plt


def _explode_labels(frame: pd.DataFrame, column: str, extra_keys: list[str]) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return pd.DataFrame(columns=[*extra_keys, "label"])
    working = frame[[*extra_keys, column]].copy()
    working[column] = working[column].fillna("").astype(str)
    working["label"] = working[column].str.split(";")
    working = working.explode("label")
    working["label"] = working["label"].str.strip()
    working = working[working["label"] != ""]
    return working[[*extra_keys, "label"]]


if "classified_by_question" in globals() and classified_by_question:
    for (question_id, question_title, take), (is_per_model, classified_frame) in (
        classified_by_question.items()
    ):
        print(f"Category statistics for {question_id} | {question_title}")
        if is_per_model:
            model_column = "model_discussed"
        else:
            model_column = "model"
        group_keys = [model_column] if model_column in classified_frame.columns else []

        for theme_type, labels_column in (
            ("strength", "strength_labels"),
            ("weakness", "weakness_labels"),
        ):
            exploded = _explode_labels(classified_frame, labels_column, group_keys)
            if exploded.empty:
                print(f"  No {theme_type} labels for {question_id}.")
                continue

            if group_keys:
                stats = (
                    exploded.groupby([model_column, "label"], dropna=False)
                    .size()
                    .reset_index(name="count")
                )
                totals_per_model = (
                    stats.groupby(model_column)["count"].transform("sum")
                )
                stats["share"] = (stats["count"] / totals_per_model).round(3)
                stats = stats.sort_values(
                    [model_column, "count"], ascending=[True, False]
                )
                print(f"{theme_type.title()}s for {question_id} (per model)")
                display(stats)

                pivot = stats.pivot(
                    index="label", columns=model_column, values="count"
                ).fillna(0)
                pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
                ax = pivot.plot(
                    kind="bar",
                    figsize=(11, 5),
                    title=f"{question_id} — {theme_type} label counts per model",
                )
                ax.set_xlabel("")
                ax.set_ylabel("Count")
                ax.legend(title="Model")
                plt.xticks(rotation=25, ha="right")
                plt.tight_layout()
                plt.show()
            else:
                stats = (
                    exploded.groupby("label", dropna=False)
                    .size()
                    .reset_index(name="count")
                    .sort_values("count", ascending=False)
                )
                total = int(stats["count"].sum())
                stats["share"] = (stats["count"] / total).round(3) if total else 0.0
                print(f"{theme_type.title()}s for {question_id}")
                display(stats)

                ax = stats.set_index("label")["count"].plot(
                    kind="bar",
                    figsize=(10, 4),
                    title=f"{question_id} — {theme_type} label counts",
                )
                ax.set_xlabel("")
                ax.set_ylabel("Count")
                plt.xticks(rotation=25, ha="right")
                plt.tight_layout()
                plt.show()

# %% [markdown]
# ## 4. Create Result Visualizations
#
# Use simple bar charts to compare mean numeric scores and the distribution of categorical comparison answers across all cases.

# %%
import matplotlib.pyplot as plt

q1_means = numeric_stats[numeric_stats["question_id"] == "q1"].copy()
if not q1_means.empty:
    q1_title = q1_means["question_title"].iloc[0]
    q1_plot = q1_means.pivot(index="row_label", columns="model", values="mean")
    ax = q1_plot.plot(kind="bar", figsize=(11, 5), title=f"{q1_title}\nMean ratings across all cases")
    ax.set_xlabel("")
    ax.set_ylabel("Mean score")
    ax.legend(title="Model")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.show()

q4_shares = choice_stats[choice_stats["question_id"] == "q4"].copy()
if not q4_shares.empty:
    q4_title = q4_shares["question_title"].iloc[0]
    expected_q4_labels = get_question_choice_labels(survey_definition, "q4")
    for _, row_frame in q4_shares.groupby("row_label", sort=False):
        row_label = row_frame["row_label"].iloc[0]
        plot_data = row_frame.set_index("choice_label")["share"]
        plot_data = plot_data.reindex(expected_q4_labels, fill_value=0)
        ax = plot_data.plot(kind="bar", figsize=(9, 4), title=f"{q4_title}\n{row_label}")
        ax.set_xlabel("")
        ax.set_ylabel("Share")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.show()
