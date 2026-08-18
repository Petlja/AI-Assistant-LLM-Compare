# Plan: a validated scorer, shared by the fine-tuning pipeline and the compare harness

**Status:** in progress — Phase 1 done
**Created:** 2026-08-18 · **Revised:** 2026-08-18 (open questions answered)
**Repos:** `AI-Assistant-Fine-Tuning` (FT), `AI-Assistant-LLM-Compare` (CMP)

---

## 0. The idea in one page

### What exists today

**FT pipeline** (`astft`):

```
gen-questions  →  gen-answers  →  gen-td  →  sft-finetune / dpo-finetune
 teacher model     student model   teacher model scores 1-10
 reads lectures    (local vLLM)    + writes improved_answer
                                   score >= 7  → SFT record
                                   score <  7  → DPO record (chosen=improved, rejected=original)
```

The scorer lives in `ai_assistant_fine_tuning/core.py`: `EVALUATION_PROMPT`,
`EvaluationResult(score 1-10, improved_answer)`, `evaluate_answer()`.
**This one function decides the entire shape of the training data.** Nobody has
ever checked whether its scores mean anything.

**CMP harness** (`plcmp`): `prepare → inference → judge_compare → human_eval → survey`.
`judge_compare` shows the judge *both* answers side by side and asks "which is
better" (A / B / Tie), run twice with swapped order to cancel position bias.

### The judgment error

`judge_compare` is **pairwise**. The FT pipeline is **pointwise**. They are
different tasks, and a pairwise judge being good tells you nothing about whether
the pointwise scorer is good. So CMP currently validates a component the FT
pipeline does not use.

Pairwise comparison stays the right tool for the *end product* question — "is the
fine-tuned model better than the base model" — so that command survives. It is
just not what we are building right now.

### What we actually want

Take the **pointwise scorer**, and validate it against humans through pairwise
preferences (humans are far more reliable at "which of these two is better" than
at "give this a 70 out of 100"):

```
answer A ──► scorer ──► score_a  ┐
                                 ├──► derived verdict (A / B / Tie)  ┐
answer B ──► scorer ──► score_b  ┘                                   ├──► compared by hand
                                                                     │
human annotator sees A and B blind ──► human verdict (A / B / Tie) ──┘
```

The scorer never sees both answers at once — exactly as in `gen-td`. Iterate on
it (prompt, rubric, model, temperature) until the derived verdicts match human
verdicts as closely as possible. **Then that same, now-validated scorer is what
`gen-td` runs.** One implementation, one prompt, two callers.

### Success criteria

- One importable, dependency-light scorer module in FT, used by both repos.
- A repeatable CMP command that emits, for one answer pair set, the four files
  needed to judge scorer-vs-human agreement by hand.
- A measured baseline, and at least one iteration that beats it.
- `gen-td` running the validated scorer.

### Decisions — locked in, do not relitigate

| Decision | Choice |
|---|---|
| Where the scorer lives | FT repo. It is a fine-tuning pipeline component; CMP is its test harness. |
| How CMP gets it | `uv` path dependency, editable — same pattern CMP already uses for `plct-server`. |
| `improved_answer` | **Always generated.** Token cost is not a concern. It stays part of the scorer contract, in calibration runs as well as `gen-td`. |
| Score scale | **1-100.** Matches `judge_compare`, and 1-10 integers tie so often the agreement metric goes numb. |
| Tie band ε | **Default 5** on the 1-100 scale (`abs(score_a - score_b) <= 5 → Tie`). Sensible range 3-5; exposed as a CLI argument, and raw scores are always written out so ε can be re-derived by hand without re-running. |
| SFT/DPO threshold | Stays a CLI argument. **Default moves 7 → 70** with the scale. |
| Pairwise `judge_compare` | Keep exactly one such command, cleaned up. It answers the end-product question and doubles as the reference ceiling for the pointwise scorer. |
| `survey` + `analytic` | **Keep, do not touch.** Separate track, out of scope for this plan. See Step 2.2. |
| Analysis / agreement command | **Not built.** Comparison of scorer verdicts against human verdicts is done by hand or with outside tools. The CMP command's job ends at producing well-formed, row-aligned files. |

---

## Phase 1 — Clear the decks (FT repo) ✅ DONE

> Completed 2026-08-18 on branch `chore/remove-sample-eval`, commit `f410d2e`
> (FT repo). 691 lines removed. `--scores-output` gone from `gen-td`, verified
> through click; `gen_td.py` is byte-identical to `4177fc2`. Not yet merged to
> `main` — merge or PR when convenient.

**Goal:** remove the human-feedback experiment from FT. That work belongs in CMP,
where the human artifacts already live. Leaves FT as a clean fine-tuning pipeline.

**Scope:** the five commits dated 2026-08-03 — `736beef`, `6cf2775`, `906ffb6`,
`69ae3d3`, `7e655a8`.

⚠️ **Do not `git revert` the range.** `6cf2775` also carries two keepers: the
`requests>=2.34.0` pin (urllib3 compat fix) and the venv-setup docs in `AGENTS.md`.
Remove by hand.

### Step 1.1 — Delete `sample-eval` ✅

- Delete `ai_assistant_fine_tuning/sample_eval.py` (595 lines).
- `ai_assistant_fine_tuning/cli.py`: drop the import and `cli.add_command(sample_eval)`
  plus its `# Human evaluation sampling` comment.
- `pyproject.toml`: drop `markdown>=3.0` and `pyyaml>=6.0` under `# Sample evaluation`.
  Verified: nothing else in the package imports them.
- `.gitignore`: drop `/human_eval`.

**Done when:** `astft --help` runs and lists no `sample-eval`.

### Step 1.2 — Revert `gen-td` to pure routing ✅

- `ai_assistant_fine_tuning/gen_td.py`: remove `--scores-output`, the
  `scores_output` parameter, `score_records`, and the two `click.echo` lines
  that mention scores. Restores the file to its `4177fc2` shape.

**Done when:** `astft gen-td --help` shows no `--scores-output`; a small run
still writes `training_data.jsonl`, `training_data_dpo.jsonl`, `evaluation.txt`.

### Step 1.3 — Clean the docs ✅

- `AGENTS.md`: delete `### HUMAN EVALUATION` / `#### sample-eval` (~lines 174-225),
  the `scores.jsonl` bullets in Step 3 (~lines 85, 94), the `sample_eval.py` entry
  in Module Organization (~line 343), and the `sample-eval` line in the workflow
  (~line 388). **Keep** `### Virtual Environment Setup`.
- `README.md`: check for and remove any `sample-eval` mentions.

**Done when:** `grep -rn "sample.eval\|scores-output" --include=*.md --include=*.py .`
returns nothing outside `.git`.

### Step 1.4 — Commit ✅

One commit: `chore: remove human-eval sampling from fine-tuning pipeline`.
Body should say the work moved to AI-Assistant-LLM-Compare.

---

## Phase 2 — Clear the decks (CMP repo)

**Goal:** resolve the pending work and give CMP exactly one pairwise compare
command. Nothing is deleted.

### Step 2.1 — Land or drop the working tree

There are uncommitted changes on `main` right now: `judge_compare.py` (the two
prompt templates merged into one parameterised template + `judge_results_*.yml`
output), `plcmp.py`, `.gitignore`, and an untracked `test.yaml`.

Read the diff, decide, commit or `git restore`. **Everything after this step
assumes a clean tree.** Do not start Phase 3 with this hanging.

That diff already does most of Step 2.3. If it looks right, landing it is the
cheapest path.

### Step 2.2 — What stays (no deletions)

| Module | Lines | Disposition |
|---|---|---|
| `prepare.py` | 47 | Keep. Fetches lesson content, produces system messages. |
| `inference.py` | 83 | Keep. Produces the answers being scored. |
| `judge_compare.py` | 546 | Keep — one command, cleaned up in Step 2.3. |
| `human_eval.py` + `human_eval_template.py` | 306 + 165 | Keep. Its generators get reused by the new calibrate command. |
| `models.py` | 250 | Keep, will grow. |
| `survey.py` | 190 | **Keep, do not touch.** Separate track. |
| `analytic.py` | 854 | **Keep, do not touch.** Separate track. |

`survey` + `analytic` are their own line of work. They are not part of the
calibration pipeline and no step in this plan modifies them.

> ❓ **One thing to confirm at chunk time:** "last two are its own branch" — read
> here as *a separate track of functionality, left on `main`*. If you meant a
> literal git branch (move them off `main` to keep it lean), say so; it is a
> one-step change, but it is destructive to `main` so it is not assumed.

### Step 2.3 — Clean sweep of the one compare command

Keep the behaviour; tidy everything around it.

- One parameterised prompt template, not two near-duplicates.
- `judge_compare` and `judge_compare_sysmsg` collapse into one implementation
  (two CLI entry points over one code path is fine; two code paths is not).
- Review the CLI arguments: consistent names, sensible defaults, nothing vestigial.
- Review the judge prompt end to end — it is the most carefully tuned prose in
  either repo and Phase 6 borrows from it.
- Output file naming consistent and predictable.

**Done when:** `judge_compare.py` is meaningfully shorter, both CLI entry points
still run end to end on `eval/test-cases.yml`, and tests pass.

### Step 2.4 — Commit
Leave it to user to commit
---

## Phase 3 — The scorer module (FT repo)

**Goal:** one importable, dependency-light, stable-API scorer. This is the
centrepiece — everything downstream depends on its shape.

Note the ordering: this phase extracts the scorer **without changing its
behaviour** (still 1-10), so the extraction is verifiable on its own. The rescale
to 1-100 is Phase 4, as its own verifiable step. Do not merge the two.

### Step 3.1 — Make FT importable without torch

Blocker: FT's `pyproject.toml` pulls `torch`, `vllm`, `trl`, `peft`,
`bitsandbytes`, `transformers` as hard dependencies. CMP must not install those.

- Move all training/serving deps into an optional extra: `[project.optional-dependencies] train = [...]`.
- Base install becomes roughly `click`, `openai`, `pydantic`, `python-dotenv`,
  `tenacity`, `requests`, `plct-ai-data-unifier`.
- Guard the imports in `finetune_sft.py` / `finetune_dpo.py` so `astft --help`
  still works without the extra, failing with a clear message only when a
  training command actually runs.
- Second blocker: FT is `>=3.10,<3.14`, CMP is `>=3.13,<4.0`. Overlap is 3.13
  only — workable, but note it in both READMEs so nobody bumps CMP to 3.14.

**Done when:** `uv sync` without `--extra train` gives a working `astft --help`,
and `uv sync --extra train` still trains.

### Step 3.2 — Extract `scoring.py`

New module `ai_assistant_fine_tuning/scoring.py`. Move the scoring concern out of
`core.py` into a self-contained unit:

```python
class ScorerConfig(BaseModel):
    name: str            # identifies this variant in results, e.g. "v1-baseline"
    model: str
    temperature: float
    scale_max: int       # 10 now; becomes 100 in Phase 4
    prompt_template: str

class Score(BaseModel):
    score: int
    scale_max: int
    improved_answer: str   # always present — see Decisions
    raw: str               # keep for debugging

def score_answer(client, *, system_message, question, answer, config) -> Score: ...

SCORER_VARIANTS: dict[str, ScorerConfig]   # registry; "v1-baseline" is today's behaviour
```

Rules:
- `EVALUATION_PROMPT` moves across as the `v1-baseline` template — **byte for
  byte**, so Step 3.3 can prove nothing drifted.
- The scorer takes an injected client. No global `OpenAI()` inside it.
- No filesystem access, no `click`. Pure function of its inputs.
- Re-export from `ai_assistant_fine_tuning/__init__.py` so CMP imports stay short.

**Done when:** `from ai_assistant_fine_tuning.scoring import score_answer, SCORER_VARIANTS`
works from a bare venv with only base deps.

### Step 3.3 — Point `gen-td` at it

`gen_td.py` calls `score_answer(...)` instead of `core.evaluate_answer`.
**Zero behaviour change.**

**Done when:** the same input JSONL produces byte-identical SFT/DPO output before
and after, at temperature 0.

### Step 3.4 — Tests + commit

Unit tests with a stubbed client: parse a valid response, reject an out-of-range
score, handle a refusal. Do not hit the API in tests.

---

## Phase 4 — Rescale to 1-100 (FT repo)

**Goal:** the locked-in scale change, as its own step so the diff is readable and
the blast radius is obvious.

### Step 4.1 — Rewrite the scorer rubric for 1-100

- `EvaluationResult.score` / `Score.score`: `ge=1, le=100`.
- `ScorerConfig.scale_max = 100`.
- Rewrite the score anchors in the prompt. **Do not just multiply by ten** — the
  existing 1-10 bands are four coarse buckets. Port the five-band anchor scheme
  from `JUDGE_USER_PROMPT_TEMPLATE` in CMP's `judge_compare.py`; it is already
  written for 1-100 and already tuned.
- Keep the `improved_answer` instruction; update "would score 9-10" → "90-100".
- `transform_evaluate_batch` in `core.py` validates `1 <= score <= 10` — update
  it, or batch-path runs will start throwing.

### Step 4.2 — Move the threshold default

- `DEFAULT_THRESHOLD = 7` → `70`.
- `gen_td.py`: `click.IntRange(1, 10)` → `click.IntRange(1, 100)`, `show_default`
  text updated. It stays an input argument.
- Sweep `AGENTS.md` / `README.md` for every "1-10", "score >= 7", "7/10".

⚠️ **This changes every training dataset generated afterwards.** Old
`answers.jsonl` scored under the 1-10 scorer is not comparable to new output. Say
so in the commit message and in `AGENTS.md`.

### Step 4.3 — Sanity-check and commit

Run `gen-td` over a real `answers.jsonl`. Look at the score histogram: if
everything piles up in one 10-point band, the anchors are not doing their job and
the tie band will have nothing to bite on. Fix the anchors before moving on —
this is the foundation the whole calibration sits on.

---

## Phase 5 — The calibrate command (CMP repo)

**Goal:** one command that produces everything needed to compare scorer verdicts
against human verdicts by hand.

### Step 5.1 — Add the dependency

CMP `pyproject.toml`:

```toml
[tool.uv.sources]
ai-assistant-fine-tuning = { path = "../AI-Assistant-Fine-Tuning", editable = true }
```

Same pattern as the existing `plct-server` entry. Note in CMP's README that the
two repos must sit side by side.

**Done when:** `uv sync` succeeds and `python -c "from ai_assistant_fine_tuning.scoring import score_answer"`
works in CMP's venv — with no torch installed.

### Step 5.2 — New command `plcmp calibrate`

```
plcmp calibrate \
  -c eval/output/test-cases-sysmsg.json \
  --model-a gpt-4o --model-a-take 1 \
  --model-b gpt-4o --model-b-take 2 \
  --scorer v1-baseline \
  --tie-band 5 \
  -o eval/output/calibrate
```

Per case: read the two pre-generated `.txt` answers, run `score_answer` on A
**alone**, then on B **alone** — never both in one call — and derive
`A` / `B` / `Tie` from the two scores and ε.

Then emit one directory per pair containing four files:

```
eval/output/calibrate/<pair>/
  index.html          # blind side-by-side reading aid for the annotator
  human_feedback.yml  # the annotator fills this in by hand
  eval_answers.yml    # what the scorer said — row-aligned with human_feedback.yml
  assignment.yml      # answer key: displayed A/B → true (model, take)
```

*(Named `human_feedback.yml` per your `human_feedback.yaml`, with `.yml` to match
the repo's existing `assignment.yml`.)*

**The two comparison files must line up row for row.** Analysis is by hand, so
`eval_answers.yml` is written in **displayed order** — its `A` is whatever the
annotator saw as `A` — with the same case ids in the same sequence as
`human_feedback.yml`. Each row carries `swapped` inline so the file is
self-describing without cross-referencing the key:

```yaml
scorer: v1-baseline
tie_band: 5
frame: displayed        # A/B as shown to the annotator, NOT canonical
model_a: gpt-4o
take_a: 1
model_b: gpt-4o
take_b: 2
cases:
  - id: A1
    swapped: false      # displayed A is model_a/take_a
    score_a: 82
    score_b: 68
    verdict: A          # derived from the scores at tie_band 5
```

Raw `score_a` / `score_b` are always written, so ε can be re-derived by hand in a
spreadsheet without re-running anything.

Implementation notes:
- New file `src/plct_llm_compare/calibrate.py`. Do not bolt this onto `judge_compare.py`.
- **Reuse `human_eval.py`'s generators** for `index.html`, the blind shuffle, the
  assignment key, and the carry-over-existing-input behaviour. Do not reimplement
  them — one shuffle implementation, or the two files will silently misalign.
- Keep `plcmp human_eval` registered as a standalone command for when you want
  the human artifacts without paying for scoring. Same module, same layout.
- `improved_answer` is generated for every scored answer (locked decision). Write
  them to `improved/<case>_<side>.md` inside the pair directory rather than inline,
  so `eval_answers.yml` stays readable.
- Concurrency: 2N independent calls. `asyncio.gather` with a semaphore, as
  `judge_compare` does for its AB/BA pair.
- Cache scores on disk keyed by (case, model, take, scorer name). Phase 6 re-runs
  this repeatedly and you should not pay twice.

**Explicitly not in scope:** any agreement/κ/confusion-matrix command. The command
stops at producing the files.

**Done when:** the command runs over the existing `eval/output/` answers and
writes a complete, row-aligned pair directory.

### Step 5.3 — Commit

---

## Phase 6 — Calibrate

**Goal:** make scorer verdicts match human verdicts. Phases 1-5 exist to make
this phase cheap to repeat.

### Step 6.1 — Generate a pair set and collect human feedback

```bash
plcmp inference -m gpt-4o --take 1 -t 0.7
plcmp inference -m gpt-4o --take 2 -t 0.7
plcmp calibrate --model-a gpt-4o --model-b gpt-4o --model-a-take 1 --model-b-take 2
```

Hand `index.html` + `human_feedback.yml` to annotators. Keep `assignment.yml` and
`eval_answers.yml` away from them.

**Sizing:** with N cases, the standard error on an agreement rate near 70% is
about `sqrt(0.7*0.3/N)`. N=30 → ±8pp, which cannot tell a real 5pp improvement
from noise. **Aim for 100+ annotated pairs.** Prefer pairs that are genuinely
close in quality — pairs where one answer is obviously terrible are agreed on by
everything and carry no signal.

This is the long pole and it does not parallelise with code. It also does not
depend on any of Phases 1-5 — `plcmp human_eval` works *today*, so annotation can
start immediately and in parallel with all the refactoring.

### Step 6.2 — Baseline

Compare `eval_answers.yml` against the filled-in `human_feedback.yml` by hand or
in whatever tool you like. Record in `eval/alignment-results.md`: scorer name, tie
band, N, agreement, date, and **which cases disagreed**.

The disagreement list matters more than the aggregate. That is where the next
prompt improvement comes from.

Also run the pairwise `judge_compare` over the same pairs. **That is the reference
ceiling** — a pairwise judge should beat a pointwise scorer. If the pointwise
scorer already matches it, you are done early.

### Step 6.3 — Sweep the tie band

Free: re-derive verdicts from the `score_a`/`score_b` already in
`eval_answers.yml` at ε = 3, 4, 5 and see which lands best. **No new API calls.**
Do this before touching any prompt.

### Step 6.4 — Scorer variants

One variant per chat session. For each: add a `ScorerConfig` to `SCORER_VARIANTS`
in FT, run `plcmp calibrate --scorer <name>`, append a row to
`eval/alignment-results.md`.

Candidates, roughly in order of expected value:
1. `v2-criteria` — port the full criteria list (correctness, instruction
   following, completeness, relevance, clarity, educational usefulness) from
   `JUDGE_USER_PROMPT_TEMPLATE` into the pointwise prompt. That rubric was tuned;
   the pointwise prompt never got the same attention.
2. `v3-temp0` — temperature 0 instead of 0.3. Removes run-to-run noise, which
   otherwise shows up as fake disagreement.
3. `v4-model` — a stronger teacher model. Tells you how much of the gap is the
   prompt versus the model.
4. `v5-anchors` — targeted anchor rewrites driven by whatever the disagreement
   cases from 6.2 actually show.

**Discipline:** one change per variant. Always against the same frozen human set.
Do not tune on the disagreement cases and then report on those same cases — past
three or four iterations, hold out 20-30% of the pairs and only touch them at the
end.

**Done when:** `eval/alignment-results.md` has a table of variants and a clear winner.

---

## Phase 7 — Close the loop (FT repo)

### Step 7.1 — Promote the winner

Make the winning variant the default `ScorerConfig`. Keep the older variants in
`SCORER_VARIANTS` — they are the reproduction record for `eval/alignment-results.md`.

### Step 7.2 — Re-anchor the threshold

The default moved to 70 in Phase 4 as a straight translation. Now set it from
evidence: run the winning scorer over a real `answers.jsonl`, look at the score
distribution, and pick the split that gives the SFT/DPO ratio you actually want.

### Step 7.3 — Regenerate and sanity-check

Run `gen-td` with the winning scorer. Compare the SFT/DPO split and score
histogram against the previous scorer. Spot-check ten routed records by hand.

### Step 7.4 — Document

- FT `AGENTS.md` / `README.md`: the scorer is validated against human preferences;
  point at CMP's `eval/alignment-results.md`; explain how to add a variant.
- CMP `README.md`: `calibrate` is the primary path; `judge_compare` answers the
  end-product question and is the pairwise reference; `survey` / `analytic` are a
  separate track.

---

## Suggested chat-sized chunks

Each line is one session. Phases are ordered; within a phase, steps are ordered.

| # | Chunk | Repo | Size |
|---|---|---|---|
| ~~1~~ | ~~Steps 1.1-1.4 — remove `sample-eval`~~ **done** (`f410d2e`) | FT | small |
| 2 | Step 2.1 — resolve working tree | CMP | small |
| 3 | Steps 2.2-2.4 — clean sweep of the one compare command | CMP | medium |
| 4 | Step 3.1 — optional `train` extra | FT | medium |
| 5 | Steps 3.2-3.4 — extract `scoring.py`, repoint `gen-td`, tests | FT | large |
| 6 | Steps 4.1-4.3 — rescale to 1-100, threshold default 70 | FT | medium |
| 7 | Steps 5.1-5.3 — dependency + `plcmp calibrate` | CMP | large |
| 8 | Step 6.1 — generate pairs, hand out annotation files | CMP | mostly waiting on people |
| 9 | Steps 6.2-6.3 — baseline + tie-band sweep (by hand) | — | small |
| 10+ | Step 6.4 — one scorer variant per session | FT + CMP | small each |
| last | Phase 7 — promote, re-anchor, document | FT | medium |

**Chunk 8 gates chunks 9+**, but it does not have to wait for chunks 1-7:
`plcmp human_eval` already produces the annotation artifacts today. Start the
human annotation as early as you can.

## Context a fresh chat needs

Point it at this file, name the chunk, and add:

- The two repos sit side by side under `c:\Git\`.
- FT scorer today: `ai_assistant_fine_tuning/core.py` — `EVALUATION_PROMPT`,
  `EvaluationResult`, `evaluate_answer`, consumed by `gen_td.py`.
- CMP pairwise judge: `src/plct_llm_compare/judge_compare.py`; human artifacts:
  `src/plct_llm_compare/human_eval.py`.
- The point of the whole thing: **the pointwise scorer that routes SFT/DPO is
  unvalidated; make it match human pairwise preference, then ship it back.**
- Work in bite-size chunks. One chunk per session, verified before the next.
