# Comparing AI Assistant Responses Across LLMs

A CLI tool for comparing LLM responses to educational prompts and generating evaluation surveys.

## Setup

Create and activate a Python 3.13+ virtual environment, then install dependencies:

```bash
uv sync
```

## Usage

The CLI (`plcmp`) has four commands:

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
plcmp inference [-c eval/output/test-cases-sysmsg.json] [-m gpt-4o]
```

Examples:

```bash
plcmp inference -m gpt-4o
plcmp inference -m Qwen/Qwen3-14B
plcmp inference -m Qwen/Qwen3-32B
```

### 3. Judge Compare

Reads the pre-generated answer files for both models and sends them to a third model (judge) for comparison. Saves judge output as HTML and detailed JSON metadata to `eval/output/`.

```bash
plcmp judge_compare [-c eval/output/test-cases-sysmsg.json] [--model-a gpt-4o-mini] [--model-b gpt-4o] [--judge-model gpt-5.2]
```

### 4. Survey

Scans `eval/output/` for HTML/JSON file pairs and generates a SurveyJS `survey.json` for evaluating the responses.

```bash
plcmp survey [-o eval/output]
```

### 5. Analyze Survey Responses in Notebook

Survey response analysis now lives in the notebook at `eval/survey-response-analysis.ipynb`. It loads `eval/survey-responces.json`, uses `eval/output/survey.json` for question labels, computes cross-case statistics with pandas, renders tables and charts, and exports the processed tables to `eval/notebook-output/`.

Open the notebook and run it top to bottom.

## Project Structure

```
eval/
  test-cases.yml          # Test case definitions
  output/                 # Generated outputs (HTML, JSON, survey.json)
src/plct_llm_compare/
  plcmp.py                # CLI entrypoint
  prepare.py              # Prepare command (fetch lesson content)
  inference.py            # Inference command (run LLM)
  survey.py               # Survey command (generate SurveyJS JSON)
  judge_compare.py        # Judge compare command (A vs B with judge model)
  models.py               # Pydantic data models
  config.py               # Configuration (API keys)
```
