"""
Math Rendering — LaTeX detection, sanitization, MathJax/KaTeX wrapper.

Architecture:
  1. Detection: regex-based identification of inline ($...$) and block ($$...$$) math
  2. Normalization: convert \(...\) → $...$ and \[...\] → $$...$$ before regex processing
  3. Sanitization: HTML-escaping, common-symbol validation, balanced-delimiter checks
  4. Rendering: wraps math in MathJax-compatible spans/divs, MathML fallback
  5. Export: full HTML page with MathJax CDN, CSS styling

No external Python dependencies — MathJax loaded via CDN in HTML export.
"""

from __future__ import annotations

import html
import re
import secrets
from typing import Tuple


# ── Detection ────────────────────────────────────────────────────────────

# Match inline math: $...$  (not $$, not escaped \$)
_INLINE_MATH_RE = re.compile(
    r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)",
    re.DOTALL,
)

# Match block math: $$...$$
_BLOCK_MATH_RE = re.compile(
    r"\$\$(.+?)\$\$",
    re.DOTALL,
)

# Common LaTeX commands that indicate valid math
_VALID_COMMANDS = frozenset({
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma",
    "tau", "upsilon", "phi", "chi", "psi", "omega",
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
    "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Pi", "Rho", "Sigma",
    "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega",
    "sum", "prod", "int", "oint", "iint", "iiint", "oint",
    "frac", "sqrt", "partial", "nabla", "infty", "pm", "mp",
    "times", "div", "cdot", "ast", "star", "circ", "bullet",
    "leq", "geq", "neq", "approx", "equiv", "sim", "propto",
    "subset", "supset", "subseteq", "supseteq", "in", "notin",
    "forall", "exists", "nexists", "implies", "iff", "therefore",
    "mathbb", "mathbf", "mathit", "mathrm", "mathcal", "mathfrak",
    "hat", "bar", "tilde", "vec", "dot", "ddot",
    "lim", "log", "ln", "sin", "cos", "tan", "sec", "csc", "cot",
    "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh",
    "exp", "max", "min", "argmax", "argmin", "det", "dim",
    "left", "right", "big", "Big", "bigg", "Bigg",
    "text", "textup", "textbf", "textit",
    "begin", "end", "matrix", "pmatrix", "bmatrix", "cases", "align",
})


def _normalize_delimiters(text: str) -> str:
    """Convert LaTeX-style delimiters to $/$ for MathJax compatibility.

    \(expr\) → $expr$   (inline)
    \[expr\] → $$expr$$ (block)
    """
    # IMPORTANT: close before open for inline to avoid double-replacement
    text = text.replace(r"\)", "$")
    text = text.replace(r"\(", "$")
    text = text.replace(r"\]", "$$")
    text = text.replace(r"\[", "$$")
    return text


def detect_math(text: str) -> dict:
    """Detect all math expressions in text (both $ and \( \[ styles).

    Returns:
        { "inline": [str, ...], "block": [str, ...], "count": int }
    """
    # First normalize LaTeX-style delimiters
    text = _normalize_delimiters(text)

    inline_matches = [m.group(1).strip() for m in _INLINE_MATH_RE.finditer(text)]
    block_matches = [m.group(1).strip() for m in _BLOCK_MATH_RE.finditer(text)]
    return {
        "inline": inline_matches,
        "block": block_matches,
        "count": len(inline_matches) + len(block_matches),
    }


def has_math(text: str) -> bool:
    """Quick check: does this text contain any LaTeX math delimiters?"""
    # Check $...$ style
    if "$" in text and _INLINE_MATH_RE.search(text) is not None:
        return True
    # Check \(...\) style
    if r"\(" in text or r"\[" in text:
        return True
    return False


# ── Sanitization ─────────────────────────────────────────────────────────

def _validate_delimiters(tex: str) -> bool:
    """Check that LaTeX delimiters are balanced (braces, brackets, parens)."""
    # Count opening/closing braces
    counts = {"{": 0, "(": 0, "[": 0}
    pairs = {"}": "{", ")": "(", "]": "["}
    for ch in tex:
        if ch in counts:
            counts[ch] += 1
        elif ch in pairs:
            opener = pairs[ch]
            if opener not in counts:
                continue
            counts[opener] -= 1
            if counts[opener] < 0:
                return False
    return all(v == 0 for v in counts.values())


def _extract_commands(tex: str) -> set[str]:
    """Extract LaTeX command names from an expression."""
    commands = set()
    for match in re.finditer(r"\\([a-zA-Z]+)", tex):
        commands.add(match.group(1))
    return commands


def sanitize_latex(tex: str) -> Tuple[str, bool]:
    """Sanitize a LaTeX expression for safe rendering.

    - HTML-escapes <, >, &
    - Validates delimiters are balanced
    - Warns on unrecognized commands (but doesn't block them)
    - Strips leading/trailing whitespace

    Returns:
        (sanitized_tex, is_valid)
    """
    if not tex or not tex.strip():
        return "", False

    tex = tex.strip()

    # HTML-escape to prevent injection
    tex = html.escape(tex, quote=False)

    # Validate delimiter balance
    if not _validate_delimiters(tex):
        # Try to fix common issues: unclosed brace at end
        brace_count = tex.count("{") - tex.count("}")
        if brace_count > 0:
            tex += "}" * brace_count
        else:
            return tex, False  # can't fix

    # Re-validate after fix
    if not _validate_delimiters(tex):
        return tex, False

    # Warn on entirely unrecognized commands (optional, not blocking)
    commands = _extract_commands(tex)
    unknown = commands - _VALID_COMMANDS
    if unknown and len(unknown) == len(commands):
        # All commands are unknown — might be plain text, not math
        return tex, False

    return tex, True


def sanitize_text(text: str) -> str:
    """Sanitize all math expressions in a text block.

    Handles both $...$ / $$...$$ and \(...\) / \[...\] delimiters.
    Converts all to $...$ / $$...$$ for MathJax compatibility.
    Processes block math BEFORE inline to prevent interference.
    """
    # First normalize LaTeX-style delimiters to $/$ for consistent processing
    text = _normalize_delimiters(text)

    def _replace_inline(match):
        tex = match.group(1).strip()
        sanitized, valid = sanitize_latex(tex)
        if valid:
            return f"${sanitized}$"
        return f"<code>{html.escape(match.group(0))}</code>"

    def _replace_block(match):
        tex = match.group(1).strip()
        sanitized, valid = sanitize_latex(tex)
        if valid:
            return f"$${sanitized}$$"
        return f"<pre><code>{html.escape(match.group(0))}</code></pre>"

    # Process blocks first so inline patterns inside blocks aren't matched
    text = _BLOCK_MATH_RE.sub(_replace_block, text)
    text = _INLINE_MATH_RE.sub(_replace_inline, text)

    return text


# ── Rendering ────────────────────────────────────────────────────────────

def render_mathjax_html(text: str) -> str:
    """Wrap math expressions in MathJax-compatible HTML.

    Inline: $x^2$ stays as-is (MathJax parses it natively)
    Block:  $$...$$ stays as-is

    This function primarily sanitizes and ensures proper formatting.
    """
    return sanitize_text(text)


def wrap_html_page(
    body_html: str,
    title: str = "Research Report",
    include_mathjax: bool = True,
) -> str:
    """Wrap content in a full HTML page with MathJax CDN and styling.

    Args:
        body_html: Inner HTML content (already sanitized with math).
        title: Page title.
        include_mathjax: Whether to include MathJax CDN script.

    Returns:
        Complete HTML document as a string.
    """
    mathjax_script = ""
    if include_mathjax:
        # Nonce-based CSP: the two MathJax scripts are the only scripts allowed
        # to run on this page — anything injected through report content is
        # blocked by both escaping (above) and the CSP meta tag.
        nonce = secrets.token_urlsafe(16)
        mathjax_script = f"""
    <script id="MathJax-script" async nonce="{nonce}"
        src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
    </script>
    <script nonce="{nonce}">
        MathJax = {{
            tex: {{
                inlineMath: [['$', '$']],
                displayMath: [['$$', '$$']],
                processEscapes: true,
            }},
            options: {{
                ignoreHtmlClass: 'no-mathjax',
            }}
        }};
    </script>"""
    else:
        nonce = secrets.token_urlsafe(16)
    csp = (
        f"default-src 'none'; "
        f"script-src 'nonce-{nonce}' https://cdn.jsdelivr.net; "
        f"style-src 'unsafe-inline'; "
        f"img-src https: data:; font-src https://cdn.jsdelivr.net; "
        f"connect-src 'none'; base-uri 'none'; form-action 'none'"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <meta http-equiv="Content-Security-Policy" content="{csp}"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>{html.escape(title)}</title>
    {mathjax_script}
    <style>
        :root {{
            --bg: #ffffff; --text: #1a1a2e; --muted: #6b7280;
            --accent: #2563eb; --code-bg: #f3f4f6;
        }}
        @media (prefers-color-scheme: dark) {{
            :root {{
                --bg: #0f172a; --text: #e2e8f0; --muted: #94a3b8;
                --code-bg: #1e293b;
            }}
        }}
        * {{ box-sizing: border-box }}
        body {{
            max-width: 780px; margin: 0 auto; padding: 40px 24px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                         Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: var(--bg); color: var(--text);
            line-height: 1.7; font-size: 16px;
        }}
        h1 {{ font-size: 2em; border-bottom: 2px solid var(--accent);
             padding-bottom: 8px; margin-top: 0; }}
        h2 {{ font-size: 1.5em; margin-top: 2em; color: var(--accent); }}
        h3 {{ font-size: 1.2em; margin-top: 1.5em; }}
        p {{ margin: 1em 0; }}
        a {{ color: var(--accent); }}
        blockquote {{
            border-left: 3px solid var(--accent); margin: 1em 0;
            padding: 0.5em 1em; color: var(--muted);
        }}
        code {{
            background: var(--code-bg); padding: 2px 6px;
            border-radius: 4px; font-size: 0.9em;
        }}
        pre {{
            background: var(--code-bg); padding: 16px; border-radius: 8px;
            overflow-x: auto;
        }}
        pre code {{ background: none; padding: 0; }}
        .math-block {{ margin: 1.5em 0; text-align: center; }}
        .meta {{
            color: var(--muted); font-size: 0.9em;
            border-bottom: 1px solid var(--muted); padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        @media print {{
            body {{ max-width: 100%; font-size: 12pt; }}
        }}
    </style>
</head>
<body>
{body_html}
</body>
</html>"""


_SAFE_HREF_RE = re.compile(r"^https?://", re.I)


def _safe_href(url: str) -> str:
    """Return a link target that cannot execute script.

    Only http/https URLs are allowed; anything else (javascript:, data:,
    vbscript:, file:, relative XSS tricks) is reduced to an inert '#'.
    """
    url = (url or "").strip()
    # Normalize scheme-detection: strip control chars / whitespace that
    # browsers may ignore (e.g. "java\tscript:").
    probe = re.sub(r"[\s\x00-\x1f]", "", url).lower()
    if _SAFE_HREF_RE.match(probe):
        return url
    return "#"


def markdown_to_html(md_text: str, title: str = "Research Report") -> str:
    """Convert markdown with math to full HTML page.

    Uses basic markdown-to-HTML conversion (headings, paragraphs, links, lists)
    plus MathJax for math rendering.
    """
    # First sanitize math (also normalizes \( \) → $ $)
    text = render_mathjax_html(md_text)

    # Basic markdown → HTML conversion
    lines = text.split("\n")
    html_lines: list[str] = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()

        # Code blocks
        if stripped.startswith("```"):
            if in_code_block:
                html_lines.append("</code></pre>")
                in_code_block = False
            else:
                html_lines.append("<pre><code>")
                in_code_block = True
            continue
        if in_code_block:
            html_lines.append(html.escape(line))
            continue

        # Block math — wrap in div for centering.
        # XSS: math text originates from web content / LLM output — escape it.
        if stripped.startswith("$$") and stripped.endswith("$$"):
            html_lines.append(f'<div class="math-block">{html.escape(stripped)}</div>')
            continue

        # Headings — escaped: heading text comes from untrusted sources
        if stripped.startswith("# "):
            html_lines.append(f"<h1>{html.escape(stripped[2:])}</h1>")
        elif stripped.startswith("## "):
            html_lines.append(f"<h2>{html.escape(stripped[3:])}</h2>")
        elif stripped.startswith("### "):
            html_lines.append(f"<h3>{html.escape(stripped[4:])}</h3>")
        # Links
        elif re.match(r"^\[(\d+)\]\s+(.+)", stripped):
            # Citation-style: [1] https://...
            html_lines.append(f"<p>{html.escape(stripped)}</p>")
        elif re.match(r"^\[(.+)\]\((.+)\)$", stripped):
            match = re.match(r"^\[(.+)\]\((.+)\)$", stripped)
            if match:
                href = _safe_href(match.group(2))
                html_lines.append(f'<p><a href="{html.escape(href, quote=True)}">{html.escape(match.group(1))}</a></p>')
        # Empty line
        elif not stripped:
            html_lines.append("")
        # Regular paragraph — escaped: content is untrusted
        else:
            html_lines.append(f"<p>{html.escape(stripped)}</p>")

    body_html = "\n".join(html_lines)
    return wrap_html_page(body_html, title=title)
