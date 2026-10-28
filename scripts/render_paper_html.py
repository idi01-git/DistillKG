"""
Standalone Publication-Grade HTML Manuscript Renderer
Author: Feynman Research Agent
Project: distillation-kg-fdd (Phase 8)

Converts papers/distillation-kg-fdd.md to papers/distillation-kg-fdd.html:
- MathJax CDN integration for dynamic LaTeX equation rendering ($e(t)$, Theorem 1 proof, etc.)
- Mermaid.js integration for architecture flowcharts
- IEEE Transactions styling with responsive typography, clean tables, and print-to-PDF formatting
"""

import os
import re
import html

PAPER_MD_PATH = os.path.join(os.path.dirname(__file__), "..", "papers", "distillation-kg-fdd.md")
PAPER_HTML_PATH = os.path.join(os.path.dirname(__file__), "..", "papers", "distillation-kg-fdd.html")


def render_html():
    if not os.path.exists(PAPER_MD_PATH):
        raise FileNotFoundError(f"Manuscript not found at {PAPER_MD_PATH}")

    with open(PAPER_MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Extract YAML frontmatter
    yaml_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", md_text, re.DOTALL)
    title = "Explainable Machine Learning and Dynamic Knowledge-Graph Assisted Fault Diagnosis for Continuous Distillation Columns"
    abstract_text = ""
    body_md = md_text

    if yaml_match:
        yaml_content = yaml_match.group(1)
        body_md = md_text[yaml_match.end():]
        t_m = re.search(r'title:\s*"([^"]+)"', yaml_content)
        if t_m:
            title = t_m.group(1)
        abs_m = re.search(r'abstract:\s*\|\s*\n(.*?)(?=\n[a-z]+:|\Z)', yaml_content, re.DOTALL)
        if abs_m:
            abstract_text = abs_m.group(1).strip()

    # Simple Markdown Parser for Headings, Tables, Lists, and Mermaid blocks
    lines = body_md.split("\n")
    html_lines = []
    in_mermaid = False
    mermaid_content = []
    in_table = False
    table_lines = []

    for line in lines:
        stripped = line.strip()

        # Handle Mermaid code blocks
        if stripped.startswith("```mermaid"):
            in_mermaid = True
            mermaid_content = []
            continue
        elif in_mermaid and stripped.startswith("```"):
            in_mermaid = False
            html_lines.append('<div class="mermaid">')
            html_lines.append("\n".join(mermaid_content))
            html_lines.append('</div>')
            continue
        elif in_mermaid:
            mermaid_content.append(line)
            continue

        # Handle Markdown Tables
        if stripped.startswith("|") and stripped.endswith("|"):
            if not in_table:
                in_table = True
                table_lines = [stripped]
            else:
                table_lines.append(stripped)
            continue
        elif in_table:
            in_table = False
            # Render table
            html_lines.append('<div class="table-responsive"><table class="table-ieee">')
            for i, tl in enumerate(table_lines):
                cells = [c.strip() for c in tl.split("|")[1:-1]]
                if i == 0:
                    html_lines.append("<thead><tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr></thead><tbody>")
                elif i == 1:
                    continue  # Separator line
                else:
                    html_lines.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            html_lines.append("</tbody></table></div>")

        # Headings
        if stripped.startswith("# "):
            html_lines.append(f'<h1 class="paper-section">{stripped[2:]}</h1>')
        elif stripped.startswith("## "):
            html_lines.append(f'<h2 class="paper-subsection">{stripped[3:]}</h2>')
        elif stripped.startswith("### "):
            html_lines.append(f'<h3 class="paper-subsubsection">{stripped[4:]}</h3>')
        elif stripped.startswith("* ") or stripped.startswith("- "):
            html_lines.append(f'<li class="paper-list-item">{stripped[2:]}</li>')
        elif stripped == "---":
            html_lines.append('<hr class="paper-divider" />')
        elif stripped:
            html_lines.append(f'<p class="paper-paragraph">{stripped}</p>')
        else:
            html_lines.append("")

    rendered_body = "\n".join(html_lines)
    # Wrap in HTML template with MathJax, Mermaid, and IEEE CSS
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <!-- MathJax for Crisp LaTeX Rendering -->
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <!-- Mermaid.js for Flowchart Rendering -->
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({{ startOnLoad: true, theme: 'neutral' }});</script>
    <style>
        body {{
            font-family: 'Times New Roman', Times, serif;
            color: #1a1a1a;
            line-height: 1.6;
            margin: 0;
            padding: 40px 20px;
            background-color: #f7f9fa;
        }}
        .paper-container {{
            max-width: 960px;
            margin: 0 auto;
            background: #ffffff;
            padding: 50px 70px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);
            border-radius: 4px;
        }}
        .paper-title {{
            font-size: 24pt;
            font-weight: bold;
            text-align: center;
            margin-bottom: 12px;
            line-height: 1.25;
        }}
        .paper-author {{
            font-size: 11pt;
            text-align: center;
            color: #444;
            margin-bottom: 24px;
            font-style: italic;
        }}
        .abstract-box {{
            border-top: 1px solid #222;
            border-bottom: 1px solid #222;
            padding: 16px 0;
            margin-bottom: 30px;
            font-size: 10pt;
            text-align: justify;
        }}
        .abstract-heading {{
            font-weight: bold;
            display: inline;
        }}
        .paper-section {{
            font-size: 15pt;
            font-weight: bold;
            border-bottom: 1.5px solid #003366;
            padding-bottom: 4px;
            margin-top: 36px;
            color: #003366;
            text-transform: uppercase;
        }}
        .paper-subsection {{
            font-size: 12pt;
            font-weight: bold;
            margin-top: 22px;
            color: #222;
        }}
        .paper-subsubsection {{
            font-size: 10.5pt;
            font-weight: bold;
            font-style: italic;
            margin-top: 14px;
        }}
        .paper-paragraph {{
            font-size: 10.5pt;
            text-align: justify;
            margin-bottom: 12px;
            text-indent: 1.5em;
        }}
        .table-responsive {{
            overflow-x: auto;
            margin: 24px 0;
        }}
        .table-ieee {{
            width: 100%;
            border-collapse: collapse;
            font-size: 9pt;
            text-align: left;
        }}
        .table-ieee th {{
            border-top: 1.5px solid #222;
            border-bottom: 1px solid #222;
            padding: 8px 6px;
            font-weight: bold;
            background: #fafafa;
        }}
        .table-ieee td {{
            padding: 6px 6px;
            border-bottom: 0.5px solid #ddd;
        }}
        .table-ieee tr:last-child td {{
            border-bottom: 1.5px solid #222;
        }}
        .mermaid {{
            text-align: center;
            margin: 24px 0;
        }}
        .paper-divider {{
            border: 0;
            border-top: 0.5px solid #ccc;
            margin: 28px 0;
        }}
        @media print {{
            body {{ background: #fff; padding: 0; }}
            .paper-container {{ box-shadow: none; padding: 0; max-width: 100%; }}
        }}
    </style>
</head>
<body>
    <div class="paper-container">
        <div class="paper-title">{html.escape(title)}</div>
        <div class="paper-author">Feynman Research Laboratory &bull; Advanced Industrial Informatics Group &bull; October 2026</div>
        
        <div class="abstract-box">
            <span class="abstract-heading">Abstract&mdash;</span>{html.escape(abstract_text)}
        </div>

        {rendered_body}
    </div>
</body>
</html>
"""

    with open(PAPER_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_doc)

    print(f"Publication-grade HTML manuscript successfully rendered to {PAPER_HTML_PATH}")
    return PAPER_HTML_PATH


if __name__ == "__main__":
    render_html()
