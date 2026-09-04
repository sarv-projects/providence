import os

OUTPUT_DIR = "reports"


def _safe_filename(filename: str, ext: str) -> str:
    """Strip directories / unsafe chars so callers cannot traverse out of OUTPUT_DIR."""
    import re
    base = os.path.basename(filename or "report")
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base).strip("._") or "report"
    return f"{base}.{ext}"


def save_markdown(report: str, filename: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, _safe_filename(filename, "md"))
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    return os.path.abspath(path)


def save_html(report: str, filename: str, title: str = "Research Report") -> str:
    """Save report as HTML with MathJax rendering."""
    from src.render.math import markdown_to_html
    html_content = markdown_to_html(report, title=title)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, _safe_filename(filename, "html"))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return os.path.abspath(path)
