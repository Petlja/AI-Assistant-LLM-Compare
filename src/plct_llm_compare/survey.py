"""Generate a SurveyJS survey.json from inference outputs."""

import json
from collections import OrderedDict
from pathlib import Path

import click

from .models import TestCaseResponce

MATRIX_QUESTION = {
    "type": "matrix",
    "isRequired": True,
    "isAllRowRequired": True,
    "title": "U kojoj meri se slažeš sa sledećim tvrđenjima",
    "columns": [
        {"value": 5, "text": "Potpuno se slažem"},
        {"value": 4, "text": "Slažem se"},
        {"value": 3, "text": "Neodlučan sam"},
        {"value": 2, "text": "Ne slažem se"},
        {"value": 1, "text": "Uopšte se ne slažem"},
    ],
    "rows": [
        {"value": "dom", "text": "Odgovor je koristan za nastavnu praksu"},
        {"value": "edu", "text": "Izbor termina u odgovoru je adekvatan"},
        {"value": "gra", "text": "Srpski jezik u odgovoru zvuči prirodno"},
        {"value": "srp", "text": "Bez izmena ili nakon menjih korekcija, jezik odgovora je dovoljno dobar"},
    ],
}


def _build_page(page_index: int, meta: TestCaseResponce, model_alias: str, html_content: str) -> dict:
    """Build a single SurveyJS page from metadata and HTML content."""
    page_name = f"page{page_index}"
    title_html = (
        f'<h4>(LLM {model_alias}) U kontekstu lekcije '
        f'<a href="{meta.activity_url}" target="_blank">{meta.activity_desc}</a> '
        f'AI asistentu je zadat prompt:</h4>'
        f'<h3>{meta.prompt}</h3>'
    )
    question_name_prefix = f"{meta.case_key}__{meta.take}__{meta.model.split('/')[-1]}"
    return {
        "name": page_name,
        "elements": [
            {
                "type": "html",
                "name": f"{page_name}_title",
                "html": title_html,
            },
            {
                "type": "panel",
                "name": f"{page_name}_aiPanel",
                "title": "Odgovor AI asistenta",
                "elements": [
                    {
                        "type": "html",
                        "name": f"{page_name}_response",
                        "html": html_content,
                    }
                ],
            },
            {
                **MATRIX_QUESTION,
                "name": f"{question_name_prefix}__q1",
            },
            {
                "type": "comment",
                "name": f"{question_name_prefix}__q2",
                "title": "Šta uočavate da bi trebalo ispravnije jezički formulisati?"
            },
            {
                "type": "comment",
                "name": f"{question_name_prefix}__q3",
                "title": "Vaš ukupan utisak o jeziku odgovora"
            }
        ],
    }


def _build_rating_page(page_index: int, case_key: str, meta: TestCaseResponce,
                        model_aliases_in_group: list[tuple[str, str]]) -> dict:
    """Build a SurveyJS comparison/rating page for a case_key group."""
    page_name = f"page{page_index}"
    question_name_prefix = f"{case_key}__0__-"
    title_html = (
        f'<h4>Poređenje odgovora u kontekstu lekcije '
        f'<a href="{meta.activity_url}" target="_blank">{meta.activity_desc}</a></h4>'
        f'<h3>Prompt: {meta.prompt}</h3>'
    )
    rating_columns = [{"value": "N", "text": f"Nema velike razlike"}] +[
        {"value": model.split('/')[-1], "text": f"Bolji je LLM {alias}"}
        for alias, model in model_aliases_in_group
    ]
    return {
        "name": page_name,
        "elements": [
            {
                "type": "html",
                "name": f"{page_name}_title",
                "html": title_html,
            },
            {
                "type": "matrix",
                "name": f"{question_name_prefix}__q4",
                "isRequired": True,
                "isAllRowRequired": True,
                "title": "Koji LLM je dao bolji odgovor u pogledu sledećeg?",
                "description": "Možete se vratiti da pogledate odgovore na prethodnim stranicama da se podsetite.",
                "columns": rating_columns,
                "rows": [
                    {"value": "dom", "text": "Korisnost za nastavnu praksu"},
                    {"value": "edu", "text": "Izbor termina"},
                    {"value": "gra", "text": "Prirodnost srpskog jezika"},
                    {"value": "overall", "text": "Ukupan utisak"},
                ],
            },
            {
                "type": "comment",
                "name": f"{question_name_prefix}__q5",
                "title": "Šta smatrate da je bolje ili lošije u odgovorima jednog ili drugog LLM-a?",
            },
        ],
    }


def do_survey(output_dir: str) -> None:
    """Scan output_dir for *.html files and generate survey.json."""
    output_path = Path(output_dir)
    html_files = sorted(output_path.glob("*.html"))

    if not html_files:
        click.echo("No HTML files found in the output directory.")
        return

    # Collect entries grouped by case_key (preserving insertion order)
    case_groups: OrderedDict[str, list[tuple[TestCaseResponce, str, str]]] = OrderedDict()
    model_aliases: dict[str, str] = {}
    next_model_alias = "A"

    for html_file in html_files:
        json_file = html_file.with_suffix(".json")
        if not json_file.exists():
            click.echo(f"  Skipping {html_file.name}: no matching JSON metadata.")
            continue

        meta = TestCaseResponce.model_validate_json(json_file.read_text(encoding="utf-8"))
        if meta.take > 1:
            click.echo(f"  Skipping {html_file.name}: take {meta.take} > 1.")
            continue

        model_alias = model_aliases.get(meta.model)
        if not model_alias:
            model_alias = next_model_alias
            model_aliases[meta.model] = model_alias
            next_model_alias = chr(ord(next_model_alias) + 1)

        html_content = html_file.read_text(encoding="utf-8")
        case_groups.setdefault(meta.case_key, []).append((meta, model_alias, html_content))
        click.echo(f"  Added {html_file.name} to group {meta.case_key}")

    # Build pages: for each case_key, emit model response pages then a rating page
    pages = []
    page_idx = 1
    for case_key, entries in case_groups.items():
        aliases_in_group: list[tuple[str, str]] = []
        for meta, model_alias, html_content in entries:
            pages.append(_build_page(page_idx, meta, model_alias, html_content))
            aliases_in_group.append((model_alias, meta.model))
            page_idx += 1

        # Add a comparison/rating page if there are multiple models
        if len(aliases_in_group) > 1:
            first_meta = entries[0][0]
            pages.append(_build_rating_page(page_idx, case_key, first_meta, aliases_in_group))
            page_idx += 1

    survey = {
        "title": "Upitnik o odgovorima AI Asistenta",
        "description": (
            "U ovom upitniku će se nalaziti primeri raznih odgovora AI asistenta "
            "na razne prompt-ove i od vas se očekuje da ocenite svaki od odgovora"
        ),
        "pages": pages,
        "showProgressBar": "top",
        "completedHtml": "<h3>Hvala vam!</h3>",
    }

    survey_file = output_path / "survey.json"
    survey_file.write_text(json.dumps(survey, indent=2, ensure_ascii=False), encoding="utf-8")
    click.echo(f"Survey saved to {survey_file} ({len(pages)} pages)")
