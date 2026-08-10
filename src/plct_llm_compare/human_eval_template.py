"""Self-contained HTML template for the human-eval side-by-side viewer.

Pure reading aid: the annotator reads answers here and fills in
annotations.yml by hand. Placeholders __TITLE__, __PAIR_META__ and
__CASE_DATA__ are substituted with str.replace (not str.format — the
template is full of CSS/JS braces).
"""

VIEWER_TEMPLATE = r"""<!DOCTYPE html>
<html lang="sr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
    color: #1c2733;
    background: #f2f4f7;
    line-height: 1.55;
  }
  a { color: #1d6fd1; }
  #app { max-width: 1400px; margin: 0 auto; padding: 1rem 1.5rem 5rem; }

  header.page h1 { margin: 0.5rem 0 0.25rem; font-size: 1.4rem; }
  header.page .meta { color: #5b6b7b; margin: 0 0 1rem; font-size: 0.9rem; }

  table.index { width: 100%; border-collapse: collapse; background: #fff;
                border: 1px solid #d8dee6; border-radius: 8px; overflow: hidden; }
  table.index th, table.index td { text-align: left; padding: 0.6rem 0.9rem;
                                   border-bottom: 1px solid #e6eaef; }
  table.index th { background: #eef1f5; font-size: 0.85rem; color: #46566a;
                   text-transform: uppercase; letter-spacing: 0.03em; }
  table.index tbody tr { cursor: pointer; }
  table.index tbody tr:hover { background: #f5f9ff; }

  button { font: inherit; cursor: pointer; border-radius: 6px;
           border: 1px solid #c3ccd6; background: #fff; padding: 0.45rem 1rem; }
  button:hover { background: #f0f4f9; }

  header.case { display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap; }
  header.case h1 { margin: 0.5rem 0; font-size: 1.3rem; }
  header.case .back { font-size: 0.9rem; }

  .prompt-box { background: #fff8e6; border: 1px solid #ecd9a0; border-radius: 8px;
                padding: 0.75rem 1rem; margin-bottom: 1rem; }
  .prompt-box .label { font-size: 0.75rem; text-transform: uppercase;
                       letter-spacing: 0.05em; color: #8a6d1a; margin-bottom: 0.25rem; }
  .prompt-box .activity { font-size: 0.85rem; color: #5b6b7b; margin-top: 0.4rem; }

  .answers { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; align-items: start; }
  @media (max-width: 900px) { .answers { grid-template-columns: 1fr; } }
  .answer { background: #fff; border: 1px solid #d8dee6; border-radius: 8px; }
  .answer h2 { margin: 0; padding: 0.5rem 1rem; font-size: 1rem;
               background: #eef1f5; border-bottom: 1px solid #d8dee6;
               border-radius: 8px 8px 0 0; }
  .answer .body { padding: 0.25rem 1rem 0.75rem; overflow-wrap: break-word; }
  .answer .body pre { background: #f4f6f8; border: 1px solid #e2e7ec;
                      border-radius: 6px; padding: 0.6rem; overflow-x: auto; }
  .answer .body code { background: #f4f6f8; padding: 0.1rem 0.3rem; border-radius: 4px; }
  .answer .body pre code { background: none; padding: 0; }
  .answer .body table { border-collapse: collapse; }
  .answer .body th, .answer .body td { border: 1px solid #d8dee6; padding: 0.3rem 0.6rem; }
  .answer .body img { max-width: 100%; }

  footer.bar { position: fixed; left: 0; right: 0; bottom: 0; background: #fff;
               border-top: 1px solid #d8dee6; box-shadow: 0 -2px 8px rgba(0,0,0,0.06);
               padding: 0.6rem 1.5rem; display: flex; align-items: center;
               justify-content: space-between; gap: 0.75rem; }
</style>
</head>
<body>
<div id="app"></div>
<script id="pair-meta" type="application/json">__PAIR_META__</script>
<script id="case-data" type="application/json">__CASE_DATA__</script>
<script>
(function () {
  "use strict";
  var pairMeta = JSON.parse(document.getElementById("pair-meta").textContent);
  var cases = JSON.parse(document.getElementById("case-data").textContent);
  var app = document.getElementById("app");

  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function caseIndexFromHash() {
    var m = location.hash.match(/^#c(\d+)$/);
    if (!m) { return -1; }
    var i = parseInt(m[1], 10);
    return (i >= 0 && i < cases.length) ? i : -1;
  }
  function goTo(i) {
    if (i < 0 || i >= cases.length) { location.hash = ""; }
    else { location.hash = "#c" + i; }
    render();
  }

  function renderIndex() {
    var rows = cases.map(function (c, i) {
      return '<tr data-i="' + i + '">' +
        "<td><strong>" + esc(c.id) + "</strong></td>" +
        "<td>" + esc(c.activity_desc) + "</td>" +
        "</tr>";
    }).join("");
    app.innerHTML =
      '<header class="page">' +
      "<h1>Poređenje odgovora — pregled</h1>" +
      '<p class="meta">' + esc(pairMeta.pair_dir) +
      " — ocene se upisuju u annotations.yml</p>" +
      "</header>" +
      '<table class="index"><thead><tr><th>Slučaj</th><th>Opis</th></tr></thead>' +
      "<tbody>" + rows + "</tbody></table>";
    Array.prototype.forEach.call(app.querySelectorAll("tbody tr"), function (tr) {
      tr.addEventListener("click", function () { goTo(parseInt(tr.getAttribute("data-i"), 10)); });
    });
  }

  function renderCase(i) {
    var c = cases[i];
    app.innerHTML =
      '<header class="case">' +
      '<a href="#" class="back">← Lista</a>' +
      "<h1>Slučaj " + esc(c.id) + " (" + (i + 1) + "/" + cases.length + ")</h1>" +
      "</header>" +
      '<section class="prompt-box">' +
      '<div class="label">Pitanje</div>' +
      c.prompt_html +
      '<div class="activity">' + esc(c.activity_desc) +
      ' — <a href="' + esc(c.activity_url) + '" target="_blank" rel="noopener">lekcija</a></div>' +
      "</section>" +
      '<div class="answers">' +
      '<section class="answer"><h2>Odgovor A</h2><div class="body">' + c.answer_a_html + "</div></section>" +
      '<section class="answer"><h2>Odgovor B</h2><div class="body">' + c.answer_b_html + "</div></section>" +
      "</div>" +
      '<footer class="bar">' +
      '<button id="prev">← Prethodni</button>' +
      '<button id="next">Sledeći →</button>' +
      "</footer>";

    app.querySelector(".back").addEventListener("click", function (e) {
      e.preventDefault();
      goTo(-1);
    });
    document.getElementById("prev").addEventListener("click", function () { goTo(i - 1); });
    document.getElementById("next").addEventListener("click", function () {
      goTo(i + 1 < cases.length ? i + 1 : -1);
    });
  }

  window.addEventListener("hashchange", render);
  function render() {
    var i = caseIndexFromHash();
    if (i < 0) { renderIndex(); } else { renderCase(i); }
    window.scrollTo(0, 0);
  }
  render();
})();
</script>
</body>
</html>
"""
