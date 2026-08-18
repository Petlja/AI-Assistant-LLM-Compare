"""Import the fine-tuning repo's scorer without depending on its package.

`plcmp calibrate` has to run the *exact* scorer `astft gen-td` routes on —
a copy would drift and the calibration would measure the wrong thing. But this
repo cannot declare AI-Assistant-Fine-Tuning as a dependency:

- Its `pyproject.toml` pulls torch, vllm, trl, peft and bitsandbytes. vllm has
  no Windows wheels, so `uv sync` here would simply fail.
- The fine-tuning VM runs `uv run` (serve.sh, eval_*.sh), which re-syncs the
  environment from that `pyproject.toml` before executing, and its `uv.lock` is
  gitignored. Editing the dependency lists to carve out a light install would
  re-resolve the pinned training stack on the VM. That environment is not to be
  touched.

So instead: `ai_assistant_fine_tuning/scoring.py` is written as a leaf module
(pydantic only at runtime, never importing `core`), and we put its repo on
`sys.path` and import just that module. Nothing else from that package is
imported, and nothing heavy is reachable from it.

The two repos must sit side by side. Override with the
`AI_ASSISTANT_FINE_TUNING_PATH` environment variable if they do not.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

REPO_PATH_ENV = "AI_ASSISTANT_FINE_TUNING_PATH"
DEFAULT_REPO_DIR_NAME = "AI-Assistant-Fine-Tuning"
SCORING_MODULE = "ai_assistant_fine_tuning.scoring"


def fine_tuning_repo_path() -> Path:
    """Locate the sibling fine-tuning checkout."""
    override = os.environ.get(REPO_PATH_ENV)
    if override:
        return Path(override).expanduser().resolve()
    # .../AI-Assistant-LLM-Compare/src/plct_llm_compare/scorer_bridge.py
    #     parents[2] = repo root, parents[3] = the directory holding both repos
    return Path(__file__).resolve().parents[3] / DEFAULT_REPO_DIR_NAME


@lru_cache(maxsize=1)
def load_scoring() -> ModuleType:
    """Import and return the fine-tuning repo's `scoring` module.

    Deliberately called at command runtime rather than at import time, so a
    missing sibling checkout does not break every other `plcmp` command.
    """
    repo = fine_tuning_repo_path()
    module_file = repo / "ai_assistant_fine_tuning" / "scoring.py"
    if not module_file.exists():
        raise RuntimeError(
            f"Could not find the fine-tuning scorer at {module_file}.\n"
            f"The two repos are expected to sit side by side:\n"
            f"    <parent>/{DEFAULT_REPO_DIR_NAME}\n"
            f"    <parent>/{Path(__file__).resolve().parents[2].name}\n"
            f"Set {REPO_PATH_ENV} to the fine-tuning repo path if they do not."
        )

    repo_str = str(repo)
    if repo_str not in sys.path:
        # Appended, not prepended: this repo's own modules must keep priority.
        sys.path.append(repo_str)

    try:
        import importlib

        return importlib.import_module(SCORING_MODULE)
    except ImportError as exc:  # pragma: no cover - depends on external checkout
        raise RuntimeError(
            f"Found {module_file} but could not import {SCORING_MODULE}: {exc}\n"
            "scoring.py is meant to be a leaf module needing only pydantic. "
            "If this fails, something heavyweight was added to its imports."
        ) from exc
