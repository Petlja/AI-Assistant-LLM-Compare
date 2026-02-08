# Comparing AI Assistant Responses Across LLMs

A CLI tool for comparing LLM responses to educational prompts and generating evaluation surveys.

## Setup

Create and activate a Python 3.13+ virtual environment, then install dependencies:

```bash
poetry install
```

## Usage

The CLI (`plcmp`) has three commands that form a pipeline:

### 0. Define test cases

Edit `eval/test-cases.yml` to define the test cases. Each entry specifies a case key, course/activity identifiers, activity URL, and the prompt to send to the model.

### 1. Prepare

Ensure your virtual environment is activated. Loads test cases from a YAML file, fetches lesson content for each activity, and writes an enriched JSON file with system messages to `eval/output/`.

```bash
plcmp prepare [-c eval/test-cases.yml]
```

### 2. Inference

Runs each test case against a specified model (3 takes per case), saves HTML responses and JSON metadata to `eval/output/`.

```bash
plcmp inference [-c eval/output/test-cases-sysmsg.json] [-m gpt-4o]
```

Examples:

```bash
plcmp inference -m gpt-4o
plcmp inference -m Qwen/Qwen3-14B
plcmp inference -m Qwen/Qwen3-32B
```

### 3. Survey

Scans `eval/output/` for HTML/JSON file pairs and generates a SurveyJS `survey.json` for evaluating the responses.

```bash
plcmp survey [-o eval/output]
```

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
  models.py               # Pydantic data models
  config.py               # Configuration (API keys)
```
