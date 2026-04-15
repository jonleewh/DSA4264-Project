# ./venv/bin/python src/llm_bot/app.py

from __future__ import annotations

import html
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    from .engine import LocalJobCourseBot
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from llm_bot.engine import LocalJobCourseBot


HOST = "127.0.0.1"
PORT = 8000

BOT = LocalJobCourseBot()


def render_page(query: str = "", result: dict | None = None) -> str:
    query = html.escape(query)
    result_html = ""
    if result:
        result_html = render_results(result)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MOE Job-Module Copilot</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: #fffaf3;
      --ink: #1f2937;
      --muted: #6b7280;
      --accent: #0f766e;
      --accent-2: #c2410c;
      --line: #eadfce;
      --chip: #efe4d3;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #f9dfc5 0, transparent 24%),
        radial-gradient(circle at bottom right, #d9efe8 0, transparent 26%),
        var(--bg);
    }}
    .shell {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(15,118,110,.95), rgba(194,65,12,.82));
      color: white;
      border-radius: 24px;
      padding: 28px;
      box-shadow: 0 18px 50px rgba(31,41,55,.12);
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: clamp(2rem, 4vw, 3.4rem);
      line-height: 1;
      letter-spacing: -.02em;
    }}
    .hero p {{
      margin: 0;
      max-width: 740px;
      font-size: 1.05rem;
    }}
    .panel {{
      margin-top: 24px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 22px;
      box-shadow: 0 10px 35px rgba(31,41,55,.06);
    }}
    form {{
      display: grid;
      gap: 14px;
    }}
    textarea {{
      width: 100%;
      min-height: 120px;
      padding: 16px;
      border-radius: 18px;
      border: 1px solid #d7c8b3;
      resize: vertical;
      font: inherit;
      background: white;
    }}
    button, .example {{
      cursor: pointer;
      border: 0;
      border-radius: 999px;
      font: inherit;
    }}
    .primary {{
      width: fit-content;
      padding: 12px 20px;
      background: var(--accent);
      color: white;
      font-weight: 700;
    }}
    .summary {{
      font-size: 1.02rem;
      line-height: 1.6;
      margin-bottom: 18px;
    }}
    .grid {{
      display: grid;
      gap: 18px;
      grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
    }}
    .card {{
      background: white;
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
    }}
    .card h2 {{
      margin: 0 0 12px;
      font-size: 1.2rem;
    }}
    .item {{
      padding: 14px 0;
      border-top: 1px solid #f0e5d7;
    }}
    .item:first-of-type {{
      border-top: 0;
      padding-top: 0;
    }}
    .meta {{
      color: var(--muted);
      font-size: .94rem;
      margin: 6px 0;
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }}
    .chip {{
      padding: 6px 10px;
      border-radius: 999px;
      background: var(--chip);
      font-size: .88rem;
    }}
    .stat-strip {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 14px 0 2px;
    }}
    .stat {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 10px 12px;
      min-width: 150px;
    }}
    .stat strong {{
      display: block;
      font-size: 1.1rem;
    }}
    code {{
      background: #f3ede3;
      padding: 2px 6px;
      border-radius: 8px;
    }}
    @media (max-width: 700px) {{
      .shell {{ padding: 18px 12px 36px; }}
      .panel {{ padding: 16px; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <h1>MOE Job-Module Copilot</h1>
      <p>
        Search job ads in natural language, surface the most relevant listings, and explain which
        university modules best match the skills employers are asking for based on the baseline canonical pipeline.
      </p>
    </section>

    <section class="panel">
      <form method="post" action="/">
        <label for="query"><strong>Ask in natural language</strong></label>
        <textarea id="query" name="query" placeholder="Example: Find entry-level data analyst jobs that use Python and SQL">{query}</textarea>
        <button class="primary" type="submit">Search Jobs And Recommend University Modules</button>
      </form>
    </section>

    {result_html}
  </main>
</body>
</html>"""


def render_chip_list(items: list[str], empty_text: str = "None") -> str:
    if not items:
        return f'<div class="chips"><span class="chip">{html.escape(empty_text)}</span></div>'
    chips = "".join(f'<span class="chip">{html.escape(item)}</span>' for item in items)
    return f'<div class="chips">{chips}</div>'


def render_results(result: dict) -> str:
    stats = result["stats"]
    interpreted = result["interpreted_query"]

    stat_strip = f"""
    <div class="stat-strip">
      <div class="stat"><span>Matched job ads</span><strong>{stats['matched_job_count']}</strong></div>
      <div class="stat"><span>Top jobs returned</span><strong>{stats['returned_job_count']}</strong></div>
      <div class="stat"><span>Top modules</span><strong>{stats['returned_module_count']}</strong></div>
    </div>
    """

    top_skills = render_chip_list(
        [item["skill"] for item in stats["top_skill_profile"]],
        empty_text="No dominant skill cluster",
    )
    interpreted_skills = render_chip_list(interpreted["skills"], empty_text="No exact skill phrases detected")

    recommendation_status = result["recommendation_status"]
    jobs_html = "".join(render_job(job) for job in result["jobs"]) or "<p>No job matches yet.</p>"
    if result["modules"]:
        modules_html = "".join(render_module(module) for module in result["modules"])
    elif not recommendation_status.get("available"):
        missing_list = ""
        if recommendation_status.get("missing_paths"):
            items = "".join(
                f"<li><code>{html.escape(path)}</code></li>" for path in recommendation_status["missing_paths"]
            )
            missing_list = f"<ul>{items}</ul>"
        modules_html = (
            f"<p>{html.escape(recommendation_status['message'])}</p>"
            "<p class=\"meta\">Run the baseline pipeline first, then reload the app.</p>"
            f"{missing_list}"
        )
    else:
        modules_html = "<p>No university module matches were found for this query.</p>"
    missing_html = ""
    if recommendation_status.get("missing_paths"):
        items = "".join(
            f"<li><code>{html.escape(path)}</code></li>"
            for path in recommendation_status["missing_paths"]
        )
        missing_html = f"<ul>{items}</ul>"

    return f"""
    <section class="panel">
      <div class="summary">{html.escape(result['summary'])}</div>
      {stat_strip}
      <div class="grid" style="margin-top: 18px;">
        <div class="card">
          <h2>Interpreted Query</h2>
          <p class="meta">Detected filters</p>
          {interpreted_skills}
          <p class="meta">School filter: <code>{html.escape(str(interpreted.get('school_filter') or 'Any'))}</code></p>
          <p class="meta">Work type: <code>{html.escape(str(interpreted['work_type'] or 'Any'))}</code></p>
          <p class="meta">Salary min: <code>{html.escape(str(interpreted['salary_min'] or 'Any'))}</code></p>
          <p class="meta">Salary max: <code>{html.escape(str(interpreted['salary_max'] or 'Any'))}</code></p>
          <p class="meta">Experience cap: <code>{html.escape(str(interpreted['experience_max'] or 'Any'))}</code></p>
        </div>
        <div class="card">
          <h2>Shared Employer Skills</h2>
          <p class="meta">Top signals from the best-ranked job ads</p>
          {top_skills}
        </div>
        <div class="card">
          <h2>Baseline Pipeline Status</h2>
          <p class="meta">{html.escape(recommendation_status['message'])}</p>
          <p class="meta">Canonical modules loaded: <code>{html.escape(str(recommendation_status.get('module_row_count', 0)))}</code></p>
          <p class="meta">Canonical jobs loaded: <code>{html.escape(str(recommendation_status.get('job_row_count', 0)))}</code></p>
          <p class="meta">Canonical matched jobs: <code>{html.escape(str(recommendation_status.get('canonical_job_match_count', 0)))}</code></p>
          <p class="meta">Active school filter: <code>{html.escape(str(recommendation_status.get('school_filter') or 'Any'))}</code></p>
          {missing_html}
        </div>
      </div>
    </section>

    <section class="grid" style="margin-top: 18px;">
      <div class="card">
        <h2>Top Job Ads</h2>
        {jobs_html}
      </div>
      <div class="card">
        <h2>Relevant University Modules</h2>
        {modules_html}
      </div>
    </section>
    """


def render_job(job: dict) -> str:
    matched_skills = render_chip_list(job["matched_skills"], empty_text="No exact query skill overlap")
    categories = render_chip_list(job["categories"], empty_text="Uncategorized")
    salary_text = "Not stated"
    if job["salary_minimum"] and job["salary_maximum"]:
        salary_text = f"SGD {int(job['salary_minimum']):,} to {int(job['salary_maximum']):,}"

    return f"""
    <article class="item">
      <strong>{html.escape(job['title'])}</strong>
      <p class="meta">Score: <code>{job['search_score']}</code> | {html.escape(job['work_type'] or 'Unknown')} | {salary_text}</p>
      <p class="meta">Matched query skills</p>
      {matched_skills}
      <p class="meta">Job categories</p>
      {categories}
    </article>
    """


def render_module(module: dict) -> str:
    matched = render_chip_list(module["matched_skills"], empty_text="No overlap")
    query_skills = render_chip_list(
        module.get("supported_query_skills", []),
        empty_text="No direct query skill support",
    )
    title_terms = render_chip_list(
        module.get("matched_title_terms", []),
        empty_text="No title match to the requested role",
    )
    missing = render_chip_list(module["missing_skills"], empty_text="No major missing skills")
    return f"""
    <article class="item">
      <strong>{html.escape(module['id'])} {html.escape(module['title'])}</strong>
      <p class="meta">{html.escape(module['source'])} | Recommendation score <code>{module.get('recommendation_score', module['alignment_score'])}</code> | Alignment score <code>{module['alignment_score']}</code> | Coverage <code>{module['coverage_top_k']}</code></p>
      <p class="meta">Supports your query skill(s)</p>
      {query_skills}
      <p class="meta">Role terms matched in the module title</p>
      {title_terms}
      <p class="meta">Matched employer skills</p>
      {matched}
      <p class="meta">Still missing from this module</p>
      {missing}
    </article>
    """


class CopilotHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/search":
            params = parse_qs(parsed.query)
            query = (params.get("q") or [""])[0]
            self._respond_json(query)
            return

        if parsed.path != "/":
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return

        self._respond_html(render_page())

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/":
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8")
        form = parse_qs(body)
        query = (form.get("query") or [""])[0].strip()

        result = BOT.run_query(query) if query else None
        self._respond_html(render_page(query=query, result=result))

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _respond_html(self, payload: str) -> None:
        encoded = payload.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _respond_json(self, query: str) -> None:
        payload = BOT.run_query(query) if query else {"error": "Missing q query parameter"}
        status = HTTPStatus.OK if query else HTTPStatus.BAD_REQUEST
        encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main(host: str = HOST, port: int = PORT) -> None:
    server = ThreadingHTTPServer((host, port), CopilotHandler)
    print(f"Serving MOE Job-Course Copilot on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
