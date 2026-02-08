"""Generate a SurveyJS survey.json from inference outputs."""

import json
from pathlib import Path

import click

from .models import TestCaseResponce

MATRIX_QUESTION = {
    "type": "matrix",
    "isRequired": True,
    "title": "U kojoj meri se slažeš sa sledećim tvrđenjima",
    "columns": [
        {"value": 5, "text": "Potpuno se slažem"},
        {"value": 4, "text": "Slažem se"},
        {"value": 3, "text": "Neodlučan sam"},
        {"value": 2, "text": "Ne slažem se"},
        {"value": 1, "text": "Uopšte se ne slažem"},
    ],
    "rows": [
        {"value": "dom", "text": "Odgovor je stručno precizan"},
        {"value": "edu", "text": "Odgovor je nastavno-metodološki dobar"},
        {"value": "use", "text": "Odgovor je upotrebljiv nastavniku"},
        {"value": "gra", "text": "Odgovor je pravopisno i gramatički ispravan"},
        {"value": "srp", "text": "Odgovor je jezično-stilski korektan"},
    ],
}


def _build_page(page_index: int, meta: TestCaseResponce, model_alias: str, html_content: str) -> dict:
    """Build a single SurveyJS page from metadata and HTML content."""
    page_name = f"page{page_index}"
    title_html = (
        f'<h4>(LLM {model_alias}) AI asistentu je u kontekstu '
        f'<a href="{meta.activity_url}" target="_blank">ove lekcije</a> '
        f'zadat prompt:</h4>'
        f'<h3>{meta.prompt}</h3>'
    )
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
                
                "name": f"{meta.case_key}__{meta.take}__{meta.model.split('/')[-1]}__q1",
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

    pages = []
    model_aliases = dict()
    next_model_alias = "A"
    for idx, html_file in enumerate(html_files, start=1):
        json_file = html_file.with_suffix(".json")
        if not json_file.exists():
            click.echo(f"  Skipping {html_file.name}: no matching JSON metadata.")
            continue

        meta = TestCaseResponce.model_validate_json(json_file.read_text(encoding="utf-8"))
        if meta.take > 1:
            click.echo(f"  Skipping {html_file.name}: take {meta.take} > 1.")
        else:
            model_alias = model_aliases.get(meta.model)
            if not model_alias:
                model_alias = next_model_alias
                model_aliases[meta.model] = model_alias
                next_model_alias = chr(ord(next_model_alias) + 1)
            html_content = html_file.read_text(encoding="utf-8")
            pages.append(_build_page(idx, meta, model_alias, html_content))
            click.echo(f"  Added page for {html_file.name}")

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
