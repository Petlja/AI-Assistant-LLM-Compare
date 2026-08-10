# Comparing AI Assistant Responses Across LLMs

A CLI tool for evaluating and comparing LLM responses to educational prompts. It supports automated model-vs-model comparison judged by a third judge/tutor model, prompt / system-message A/B judging, generating human-annotation files with a side-by-side viewer, and generating human-evaluation surveys.

## Setup

Create and activate a Python 3.13+ virtual environment, then install dependencies:

```bash
uv sync
```

Run the tests with:

```bash
uv run pytest
```

## Usage

The CLI (`plcmp`) has the following commands:

### 0. Define test cases

Edit `eval/test-cases.yml` to define the test cases. Each entry specifies a case key, course/activity identifiers, activity URL, and the prompt to send to the model.

### 1. Prepare

Ensure your virtual environment is activated. Loads test cases from a YAML file, fetches lesson content for each activity, and writes an enriched JSON file with system messages to `eval/output/`.

```bash
plcmp prepare [-c eval/test-cases.yml]
```

### 2. Inference

Run inference for the chosen model. Saves answer text, HTML responses and JSON metadata to `eval/output/`.

```bash
plcmp inference [-c eval/output/test-cases-sysmsg.json] [-m gpt-4o] [--take 1] [-t 0.5]
```

Examples:

```bash
plcmp inference -m gpt-4o
plcmp inference -m gpt-4o --take 2
plcmp inference -m Qwen/Qwen3-14B
plcmp inference -m Qwen/Qwen3-32B
```

To generate an answer pair from the same model (for judge/human alignment), run two takes at a moderate temperature:

```bash
plcmp inference -m gpt-4o --take 1 -t 0.7
plcmp inference -m gpt-4o --take 2 -t 0.7
```

### 3. Judge Compare

Reads the pre-generated answer files for both models and sends them to a third model (judge) for comparison. Each pair is judged **twice**, with the A/B order swapped between runs, to cancel out position bias; if the two runs disagree after un-swapping, the final verdict is `Inconsistent` (the judge is order-biased on that pair) rather than a silent tie. Saves judge output as HTML and detailed JSON metadata to `eval/output/`.

```bash
plcmp judge_compare [-c eval/output/test-cases-sysmsg.json] [--model-a gpt-4o-mini] [--model-b gpt-4o] [--model-a-take 1] [--model-b-take 1] [--judge-model gpt-5.2]
```

Examples:

```bash
plcmp judge_compare --model-a gpt-4o-mini --model-b gpt-4o --model-a-take 1 --model-b-take 2 --judge-model gpt-5.2
```

### 3.1 System-Message Comparison

Use this command to compare two answer sets generated from different preparations of the same prompt set.
This is useful when you want to judge the effect of different system messages.

```bash
plcmp judge_compare_sysmsg \
  --cases-a eval/output/prepared-v1-sysmsg.json \
  --cases-b eval/output/prepared-v2-sysmsg.json \
  --model-a gpt-4o \
  --model-b gpt-4o \
  --model-a-take 1 \
  --model-b-take 2 \
  --judge-model gpt-5.2
```

### 4. Human Evaluation

Generates the artifacts for collecting human pairwise preferences on the same answer pairs the judge compares (used to measure judge/human alignment). Writes three files to `eval/output/human_eval/<pair>/`:

- `annotations.yml` — one combined file the annotator fills in by hand (`human_verdict`: `A` / `B` / `Tie`, plus optional `human_notes` per case). Contains no model/take metadata, so the annotator stays blind.
- `assignment.yml` — the answer key mapping the displayed A/B back to the true (model, take) sources. Do not share it with annotators.
- `index.html` — a self-contained side-by-side viewer (open locally in a browser) used purely as a reading aid.

```bash
plcmp human_eval [-c eval/output/test-cases-sysmsg.json] [--model-a gpt-4o] [--model-b gpt-4o] [--model-a-take 1] [--model-b-take 2] [-o eval/output/human_eval] [--seed 0] [--no-shuffle] [--force]
```

Which answer appears as "A" is blind-shuffled per case (seeded and deterministic, so re-runs keep the same layout; disable with `--no-shuffle`). Re-running the command preserves any `human_verdict`/`human_notes` already filled in — and if the layout changed (different `--seed`), verdicts are remapped so they keep their meaning; `--force` regenerates from scratch.

### 5. Survey

Scans `eval/output/` for HTML/JSON file pairs and generates a SurveyJS `survey.json` for evaluating the responses.

```bash
plcmp survey [-o eval/output]
```

### 6. Analyze Survey Responses in Notebook

Survey response analysis now lives in the notebook at `eval/survey-response-analysis.ipynb`. It loads `eval/survey-responces.json`, uses `eval/output/survey.json` for question labels, computes cross-case statistics with pandas, renders tables and charts, and exports the processed tables to `eval/notebook-output/`.

Open the notebook and run it top to bottom.

## Project Structure

```
eval/
  test-cases.yml          # Test case definitions
  output/                 # Generated outputs (HTML, JSON, survey.json)
    human_eval/           # Human-annotation artifacts (annotations.yml, assignment.yml, viewer)
src/plct_llm_compare/
  plcmp.py                # CLI entrypoint
  prepare.py              # Prepare command (fetch lesson content)
  inference.py            # Inference command (run LLM)
  survey.py               # Survey command (generate SurveyJS JSON)
  judge_compare.py        # Judge compare command (A vs B with judge model)
  human_eval.py           # Human eval command (annotation YAML + viewer)
  human_eval_template.py  # HTML template for the side-by-side viewer
  models.py               # Pydantic data models
  config.py               # Configuration (API keys)
tests/                    # Pytest suite
```
