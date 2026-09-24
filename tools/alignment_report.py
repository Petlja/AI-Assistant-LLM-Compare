"""Compare a filled human_feedback.yml against the pointwise scorer and/or judge.

`plcmp calibrate` and `plcmp judge_compare` both stop before computing agreement.
This is that missing step, kept out of the CLI so it stays an analysis tool
rather than a supported command.

What is under test is the POINTWISE SCORER — the `calibrate` side. It is imported
live from the AI-Assistant-Fine-Tuning repo through `scorer_bridge.py`, and
`astft gen-td` routes SFT vs DPO on the single number it returns. Run `--answers`
on its own for that.

`--judge` is optional context: a pairwise judge that sees both answers at once.
Its `Inconsistent` verdicts describe the judge's own position bias and say
nothing about the scorer, so don't let them set the frame.

Everything is normalised into the DISPLAYED frame — what the annotator saw, and
the frame human_feedback.yml is written in. eval_answers.yml is already
displayed; judge_results.yml is canonical and gets un-swapped here.

Usage:
    uv run python tools/alignment_report.py \
        --human   <pair>/human_feedback.yml \
        --answers <pair>/eval_answers.yml \
        --judge   judge_results_<run>.yml
"""

import argparse
import sys
from pathlib import Path

import yaml

FLIP = {"A": "B", "B": "A"}
CATEGORIES = ("A", "B", "Tie")

# Annotators work in Serbian on a Cyrillic keyboard layout, so a verdict often
# comes back as a homoglyph: Cyrillic А (U+0410) and В (U+0412) are visually the
# Latin A and B they meant to type. Fold them by shape, not by transliteration.
HOMOGLYPHS = str.maketrans(
    {
        "А": "A",  # CYRILLIC CAPITAL A
        "а": "a",  # CYRILLIC SMALL A
        "В": "B",  # CYRILLIC CAPITAL VE — looks like B
        "в": "b",  # CYRILLIC SMALL VE
        "Т": "T",  # CYRILLIC CAPITAL TE
        "е": "e",  # CYRILLIC SMALL IE
        "і": "i",  # CYRILLIC SMALL BYELORUSSIAN-UKRAINIAN I
    }
)


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def flip_if(verdict: str, swapped: bool) -> str:
    """Canonical -> displayed (the mapping is its own inverse)."""
    return FLIP.get(verdict, verdict) if swapped else verdict


def read_human(path: Path) -> dict[str, str]:
    """Verdicts by case id, after checking the file is actually filled in."""
    doc = load(path)
    verdicts: dict[str, str] = {}
    blank: list[str] = []
    bad: list[str] = []
    folded: list[str] = []
    for case in doc["cases"]:
        raw = (case.get("human_verdict") or "").strip()
        if not raw:
            blank.append(case["id"])
            continue
        key = raw.translate(HOMOGLYPHS)
        if key != raw:
            folded.append(f"{case['id']}={raw!r}")
        norm = {"a": "A", "b": "B", "tie": "Tie"}.get(key.lower())
        if norm is None:
            bad.append(f"{case['id']}={raw!r}")
            continue
        verdicts[case["id"]] = norm
    if blank:
        sys.exit(
            f"{path}: {len(blank)}/{len(doc['cases'])} cases have an empty "
            f"human_verdict ({', '.join(blank)}).\n"
            "This is the blank template — get the annotator's filled copy."
        )
    if bad:
        sys.exit(f"{path}: unrecognised verdicts (want A/B/Tie): {', '.join(bad)}")
    if folded:
        print(f"note: read {len(folded)} Cyrillic-homoglyph verdict(s): {', '.join(folded)}")
    return verdicts


def read_swaps(answers: dict | None, judge_dir: Path) -> dict[str, bool]:
    """Per-case displayed-vs-canonical swap flags, from whichever file has them."""
    if answers is not None:
        return {c["id"]: bool(c.get("swapped")) for c in answers["cases"]}
    assignment = judge_dir / "assignment.yml"
    if not assignment.exists():
        sys.exit(
            "need swap flags to un-swap the judge into displayed frame: pass "
            f"--answers, or put assignment.yml next to the judge file ({assignment})"
        )
    return {c["id"]: bool(c.get("swapped")) for c in load(assignment)["cases"]}


def derive(score_a: int, score_b: int, tie_band: int) -> str:
    """Same rule as calibrate.py: a difference inside the band is a Tie."""
    if abs(score_a - score_b) <= tie_band:
        return "Tie"
    return "A" if score_a > score_b else "B"


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa over A/B/Tie. None when it is undefined (pe == 1)."""
    n = len(pairs)
    if not n:
        return None
    po = sum(x == y for x, y in pairs) / n
    pe = sum(
        (sum(x == c for x, _ in pairs) / n) * (sum(y == c for _, y in pairs) / n)
        for c in CATEGORIES
    )
    return None if pe >= 1.0 else (po - pe) / (1 - pe)


def confusion(pairs: list[tuple[str, str]]) -> str:
    label = "human \\ other"
    rows = [f"{label:<14}" + "".join(f"{c:>6}" for c in CATEGORIES)]
    for h in CATEGORIES:
        counts = [sum(x == h and y == o for x, y in pairs) for o in CATEGORIES]
        rows.append(f"{h:<14}" + "".join(f"{n:>6}" for n in counts))
    return "\n".join("  " + r for r in rows)


def report(label: str, pairs: list[tuple[str, str]], total: int) -> None:
    print(f"\n=== human vs {label} ===")
    if not pairs:
        print("  no comparable cases")
        return
    agreed = sum(x == y for x, y in pairs)
    k = kappa(pairs)
    k_txt = "undefined (one rater is constant)" if k is None else f"{k:+.3f}"
    print(f"  Compared:   {len(pairs)}/{total} cases")
    print(f"  Agreement:  {agreed}/{len(pairs)} ({agreed / len(pairs):.0%})")
    print(f"  Cohen's k:  {k_txt}")
    print(confusion(pairs))


def scorer_diagnostics(answers: dict, human: dict[str, str], ids: list[str]) -> None:
    """Why the scorer agrees or not, independent of where the tie band sits.

    Agreement alone cannot separate "ranks badly" from "barely ranks at all", and
    for calibrating the fine-tuning repo's pointwise prompt that is the whole
    question. So: does the margin point the right way when there is one, and does
    the scorer even produce a spread of scores.
    """
    raw = {c["id"]: (c["score_a"], c["score_b"]) for c in answers["cases"]}

    print("\n=== scorer margins (displayed frame) ===")
    print(f"  {'id':<5} {'human':<6} {'score_a':>7} {'score_b':>7} {'margin':>7}  direction")
    right = wrong = flat = 0
    for cid in ids:
        sa, sb = raw[cid]
        margin = sa - sb
        if margin == 0:
            verdict = "no margin"
            if human[cid] != "Tie":
                flat += 1
        elif human[cid] == "Tie":
            verdict = "human tie"
        elif (margin > 0) == (human[cid] == "A"):
            verdict = "agrees"
            right += 1
        else:
            verdict = "OPPOSITE"
            wrong += 1
        print(f"  {cid:<5} {human[cid]:<6} {sa:>7} {sb:>7} {margin:>+7}  {verdict}")

    print("\n=== does the margin point the right way? ===")
    print("  (tie band ignored entirely — sign of score_a - score_b vs the human's pick)")
    decided = right + wrong
    if decided:
        print(f"  Directional:  {right}/{decided} ({right / decided:.0%}) — 50% is a coin flip")
    print(f"  No margin at all, where the human had a clear pick: {flat}/{len(ids)}")

    # A pointwise prompt that returns the same number for everything cannot rank
    # anything, however the band is tuned. Check the spread before blaming the band.
    scores = [s for cid in ids for s in raw[cid]]
    mode = max(set(scores), key=scores.count)
    print("\n=== scorer score distribution ===")
    print(f"  {len(scores)} scores, {len(set(scores))} distinct, range {min(scores)}-{max(scores)}")
    print(
        f"  Most common value: {mode} appears {scores.count(mode)}x "
        f"({scores.count(mode) / len(scores):.0%})"
    )

    # If the scorer tracked human preference these two would separate: clearly
    # positive where the human picked A, clearly negative where they picked B.
    # Median alongside mean because one outlier case can invent a separation.
    for pick in ("A", "B"):
        margins = sorted(raw[i][0] - raw[i][1] for i in ids if human[i] == pick)
        if not margins:
            continue
        mean = sum(margins) / len(margins)
        mid = len(margins) // 2
        median = margins[mid] if len(margins) % 2 else (margins[mid - 1] + margins[mid]) / 2
        print(
            f"  Margin where human picked {pick}: mean {mean:+.1f}, median {median:+.1f} "
            f"(n={len(margins)}, spread {margins[0]:+d}..{margins[-1]:+d})"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--human", required=True, type=Path, help="filled human_feedback.yml")
    ap.add_argument("--answers", type=Path, help="eval_answers.yml from `plcmp calibrate`")
    ap.add_argument(
        "--judge", type=Path, help="judge_results_<run>.yml from `plcmp judge_compare`"
    )
    ap.add_argument(
        "--tie-band",
        type=int,
        default=None,
        help="override the scorer tie band (default: whatever eval_answers.yml used)",
    )
    ap.add_argument(
        "--drop-inconsistent",
        action="store_true",
        help="exclude position-inconsistent judge cases instead of counting them as misses",
    )
    args = ap.parse_args()

    if not args.answers and not args.judge:
        sys.exit("nothing to compare against: pass --answers and/or --judge")

    human = read_human(args.human)
    answers = load(args.answers) if args.answers else None
    judge = load(args.judge) if args.judge else None

    # Scorer verdicts are written in displayed frame already.
    scorer: dict[str, str] = {}
    band = args.tie_band
    if answers is not None:
        band = band if band is not None else answers.get("tie_band", 5)
        for c in answers["cases"]:
            scorer[c["id"]] = (
                derive(c["score_a"], c["score_b"], band)
                if args.tie_band is not None
                else c["verdict"]
            )

    # Judge verdicts are canonical; un-swap them per case.
    judged: dict[str, str] = {}
    consistent: dict[str, bool] = {}
    if judge is not None:
        swaps = read_swaps(answers, args.judge.parent)
        for c in judge["cases"]:
            cid = c["id"]
            if cid not in swaps:
                sys.exit(f"case {cid} is in the judge file but has no swap flag")
            judged[cid] = flip_if(c["judge_verdict"], swaps[cid])
            consistent[cid] = bool(c.get("position_agreed", True))

    ids = [c["id"] for c in load(args.human)["cases"]]
    missing = [i for i in ids if (judge and i not in judged) or (answers and i not in scorer)]
    if missing:
        sys.exit(f"cases missing from the machine-side files: {', '.join(missing)}")

    print(f"Pair: {args.human.parent.name}")
    if answers is not None:
        print(f"Scorer: {answers.get('scorer')} @ tie_band={band}")
    if judge is not None:
        print(f"Judge:  {judge.get('judge_model')}")

    header = f"\n{'id':<5} {'human':<6}"
    if judged:
        header += f" {'judge':<13} {'pos_ok':<7}"
    if scorer:
        header += f" {'scorer':<7}"
    print(header)
    for cid in ids:
        line = f"{cid:<5} {human[cid]:<6}"
        if judged:
            line += f" {judged[cid]:<13} {str(consistent[cid]):<7}"
        if scorer:
            line += f" {scorer[cid]:<7}"
        print(line)

    if scorer:
        report("scorer", [(human[i], scorer[i]) for i in ids], len(ids))
        scorer_diagnostics(answers, human, ids)

    if judged:
        pairs = [
            (human[i], judged[i])
            for i in ids
            if not (args.drop_inconsistent and not consistent[i])
        ]
        note = " (position-inconsistent cases dropped)" if args.drop_inconsistent else ""
        report(f"judge{note}", pairs, len(ids))
        if not args.drop_inconsistent:
            n_inc = sum(1 for i in ids if not consistent[i])
            if n_inc:
                print(
                    f"  note: {n_inc} case(s) are Inconsistent and can never match a "
                    "human verdict; rerun with --drop-inconsistent to exclude them"
                )

    # The tie band is a free parameter; show how much it moves the result.
    if answers is not None:
        print("\n=== scorer tie-band sensitivity ===")
        raw = {c["id"]: (c["score_a"], c["score_b"]) for c in answers["cases"]}
        for eps in (0, 2, 5, 8, 10, 15):
            got = [(human[i], derive(raw[i][0], raw[i][1], eps)) for i in ids]
            agreed = sum(x == y for x, y in got)
            ties = sum(1 for _, v in got if v == "Tie")
            k = kappa(got)
            k_txt = "  n/a" if k is None else f"{k:+.3f}"
            print(
                f"  tie_band={eps:<3} agreement {agreed}/{len(got)} "
                f"({agreed / len(got):>3.0%})  k={k_txt}  scorer ties={ties}"
            )


if __name__ == "__main__":
    main()
