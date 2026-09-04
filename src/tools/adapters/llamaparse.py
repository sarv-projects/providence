"""
LlamaParse / Docling Document Parser Adapter — layout-aware PDF → Markdown.

Addresses the "thousands of company PDFs" scenario: raw PyPDF extraction
destroys tables, columns, and reading order. This adapter prefers:

  1. LlamaParse (cloud, agentic OCR + tables + charts → clean Markdown)
     Requires: LLAMA_CLOUD_API_KEY + `llama-cloud` SDK
  2. Docling (local, layout-aware, tables → Markdown)
     Requires: `docling` package installed
  3. MinerU adapter (existing CLI/PyPDF fallback chain)

Any unavailable path degrades gracefully to the next.
"""

from __future__ import annotations

import os
import urllib.request
from typing import List, Dict

# Only actual PDF artifacts ("arxiv.org/pdf/...", ".pdf"); arxiv /abs/ pages
# are HTML and would be mis-detected as PDFs, failing the whole chain.
_PDF_HINTS = (".pdf", "/pdf/")


def _is_pdf_url(url: str) -> bool:
    return any(h in url.lower() for h in _PDF_HINTS)


def _download(url: str) -> str:
    """Download a URL to a temp file; returns local path (caller cleans up)."""
    import tempfile
    try:
        from ..urlguard import bounded_read, is_safe_url
    except ImportError:  # pragma: no cover - guard module always present
        bounded_read = None  # type: ignore
        def is_safe_url(u: str) -> bool:  # type: ignore
            return u.startswith(("http://", "https://"))
    if not is_safe_url(url):
        raise ValueError(f"Blocked URL (SSRF guard): {url}")
    req = urllib.request.Request(
        url, headers={"User-Agent": "AutonomousResearchAgent/1.0"}
    )
    with urllib.request.urlopen(req, timeout=30.0) as resp:
        data = bounded_read(resp) if bounded_read else resp.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(data)
    tmp.close()
    return tmp.name


def _parse_llamaparse(file_or_url: str) -> Dict[str, str] | None:
    """LlamaParse cloud parse. Returns None if SDK/key unavailable or fails.

    Primary path uses the official `llama-parse` SDK (LlamaParse.load_data).
    If that package is missing, a best-effort `llama_cloud` direct call is
    attempted before giving up. Both are optional deps — absence degrades to
    Docling → MinerU without crashing.
    """
    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        return None
    if file_or_url.startswith("http://") or file_or_url.startswith("https://"):
        local = _download(file_or_url)
    else:
        local = file_or_url
    try:
        # ── Official SDK: llama-parse ──
        try:
            from llama_parse import LlamaParse
            parser = LlamaParse(result_type="markdown")
            docs = parser.load_data(local)
            md = "\n\n".join(
                (getattr(d, "text", "") or "") for d in (docs or [])
            )
            if md.strip():
                title = os.path.basename(file_or_url)
                return {
                    "url": file_or_url,
                    "content": md,
                    "title": f"PDF (LlamaParse): {title}",
                    "source": "llamaparse",
                }
        except Exception as e:
            print(f"  [llamaparse] llama-parse SDK failed ({e}) — trying llama_cloud")

        # ── Fallback: raw llama_cloud client ──
        from llama_cloud import LlamaCloud
        client = LlamaCloud()  # reads LLAMA_CLOUD_API_KEY from env
        with open(local, "rb") as f:
            file_obj = client.files.create(file=f, purpose="parse")
        result = client.parsing.parse(
            file_id=file_obj.id,
            tier="agentic",
            version="latest",
            expand=["markdown"],
        )
        md = ""
        try:
            for page in result.markdown.pages:
                md += (page.markdown or "") + "\n\n"
        except Exception:
            md = getattr(result.markdown, "markdown", "") or ""
        if md.strip():
            title = os.path.basename(file_or_url)
            return {
                "url": file_or_url,
                "content": md,
                "title": f"PDF (LlamaParse): {title}",
                "source": "llamaparse",
            }
        return None
    except Exception as e:
        print(f"  [llamaparse] parse failed ({e}) — trying next parser")
        return None
    finally:
        if local != file_or_url and os.path.exists(local):
            try:
                os.remove(local)
            except OSError:
                pass


def _parse_docling(file_or_url: str) -> Dict[str, str] | None:
    """Docling local parse. Returns None if package unavailable or fails."""
    try:
        from docling.document_converter import DocumentConverter
    except Exception:
        return None
    try:
        converter = DocumentConverter()
        if file_or_url.startswith("http://") or file_or_url.startswith("https://"):
            source = file_or_url
        else:
            source = file_or_url
        result = converter.convert(source)
        md = result.document.export_to_markdown()
        title = os.path.basename(file_or_url)
        return {
            "url": file_or_url,
            "content": md,
            "title": f"PDF (Docling): {title}",
            "source": "docling",
        }
    except Exception as e:
        print(f"  [docling] parse failed ({e}) — falling back to MinerU")
        return None


def llamaparse_parse_pdf(file_or_url: str) -> Dict[str, str]:
    """Parse a PDF (local path or URL) with LlamaParse → Docling → MinerU."""
    if not file_or_url:
        return {}

    parsed = _parse_llamaparse(file_or_url)
    if parsed and parsed.get("content"):
        return parsed

    parsed = _parse_docling(file_or_url)
    if parsed and parsed.get("content"):
        return parsed

    # Final fallback: existing MinerU adapter (CLI → PyPDF/PyMuPDF)
    from .mineru import mineru_parse_pdf
    return mineru_parse_pdf(file_or_url)


def llamaparse_extract(urls: List[str]) -> List[Dict]:
    """Extract content from PDF URLs using the preferred parser chain."""
    results = []
    for url in urls[:10]:
        if _is_pdf_url(url):
            parsed = llamaparse_parse_pdf(url)
            if parsed and parsed.get("content"):
                results.append(parsed)
    return results


def llamaparse_search(query: str, max_results: int = 5) -> List[Dict]:
    """Search arXiv for PDFs to run through the layout-aware parser.

    Note: this adapter owns the PDF parse chain (LlamaParse → Docling →
    MinerU), so its internal mineru fallback intentionally runs before the
    standalone mineru/nougat adapters in registry.extract priority order.
    """
    from .mineru import mineru_search
    results = mineru_search(query, max_results=max_results)
    for r in results:
        r["source"] = "arxiv-llamaparse"
        # Preserve the original relevance score (don't clobber with a constant
        # so merged registry results keep true ordering).
        r.setdefault("score", 0.76)
    return results
