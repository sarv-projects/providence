"""
Nougat Neural OCR Adapter — converts academic PDFs with LaTeX equations into Markdown.

Uses:
  - Nougat CLI (`nougat`) if available locally
  - Nougat REST API if NOUGA_API_URL is configured
  - Fallback to MinerU / PyPDF parser for mathematical papers
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import urllib.request
import json
from typing import List, Dict


def nougat_parse_pdf(file_or_url: str) -> Dict[str, str]:
    """Parse academic paper PDF into LaTeX-formatted Markdown using Nougat neural OCR."""
    if not file_or_url:
        return {}

    api_url = os.getenv("NOUGAT_API_URL", "") or os.getenv("NOUGA_API_URL", "")
    temp_file = None
    pdf_path = file_or_url

    # Download if URL
    if file_or_url.startswith("http://") or file_or_url.startswith("https://"):
        try:
            from ..urlguard import bounded_read, is_safe_url
            if not is_safe_url(file_or_url):
                return {"url": file_or_url, "content": "", "error": "Blocked URL (SSRF guard)"}
            req = urllib.request.Request(file_or_url, headers={"User-Agent": "AutonomousResearchAgent/1.0"})
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                data = bounded_read(resp)
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp.write(data)
            temp.close()
            temp_file = temp.name
            pdf_path = temp_file
        except Exception as e:
            return {"url": file_or_url, "content": "", "error": f"Download failed: {e}"}

    try:
        # Check Nougat REST API
        if api_url:
            try:
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                req = urllib.request.Request(
                    f"{api_url.rstrip('/')}/predict/",
                    data=pdf_bytes,
                    headers={"Content-Type": "application/pdf"}
                )
                with urllib.request.urlopen(req, timeout=60.0) as resp:
                    output = resp.read().decode("utf-8")
                    return {
                        "url": file_or_url,
                        "content": output,
                        "title": f"Academic Paper (Nougat): {os.path.basename(file_or_url)}",
                        "source": "nougat_api"
                    }
            except Exception as e:
                print(f"  [nougat] API call failed ({e})")

        # Check Nougat CLI
        nougat_bin = shutil.which("nougat")
        if nougat_bin:
            out_dir = tempfile.mkdtemp()
            try:
                cmd = [nougat_bin, pdf_path, "-o", out_dir]
                subprocess.run(cmd, capture_output=True, timeout=120, check=True)
                for f in os.listdir(out_dir):
                    if f.endswith(".mmd") or f.endswith(".md"):
                        with open(os.path.join(out_dir, f), "r", encoding="utf-8") as mmd_f:
                            content = mmd_f.read()
                        return {
                            "url": file_or_url,
                            "content": content,
                            "title": f"Academic Paper (Nougat): {os.path.basename(file_or_url)}",
                            "source": "nougat_cli"
                        }
            except Exception as e:
                print(f"  [nougat] CLI execution failed ({e})")
            finally:
                shutil.rmtree(out_dir, ignore_errors=True)

        # Fallback to MinerU parser
        from .mineru import mineru_parse_pdf
        fallback = mineru_parse_pdf(file_or_url)
        if fallback:
            fallback["source"] = "nougat_fallback"
            return fallback

        return {"url": file_or_url, "content": "", "title": os.path.basename(file_or_url)}
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError:
                pass


def nougat_search(query: str, max_results: int = 5) -> List[Dict]:
    """Search arXiv for academic PDFs suitable for Nougat math OCR."""
    from .mineru import mineru_search
    results = mineru_search(query, max_results=max_results)
    for r in results:
        r["source"] = "arxiv-nougat"
        r["score"] = 0.78
    return results


def nougat_extract(urls: List[str]) -> List[Dict]:
    """Extract academic PDF content using Nougat adapter."""
    results = []
    for url in urls[:5]:
        if "arxiv.org" in url.lower() or url.lower().endswith(".pdf"):
            res = nougat_parse_pdf(url)
            if res and res.get("content"):
                results.append(res)
    return results
