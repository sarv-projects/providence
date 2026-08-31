"""
MinerU Document Parser Adapter — extracts structured Markdown from PDF and Office documents.

Supports:
  - Local PDF files or HTTP PDF links
  - MinerU API / CLI execution (if installed)
  - PyPDF / pdfplumber fallback for zero-dependency PDF parsing
"""

from __future__ import annotations

import logging

import os
import shutil
import subprocess
import tempfile
import urllib.request
from typing import List, Dict


def mineru_parse_pdf(file_or_url: str) -> Dict[str, str]:
    """Parse a PDF document (local path or URL) into clean Markdown using MinerU (or fallback)."""
    if not file_or_url:
        return {}

    temp_file = None
    pdf_path = file_or_url

    # Download if URL
    if file_or_url.startswith("http://") or file_or_url.startswith("https://"):
        try:
            req = urllib.request.Request(file_or_url, headers={"User-Agent": "AutonomousResearchAgent/1.0"})
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                data = resp.read()
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp.write(data)
            temp.close()
            temp_file = temp.name
            pdf_path = temp_file
        except Exception as e:
            return {"url": file_or_url, "content": "", "error": f"Download failed: {e}"}

    try:
        # Check if MinerU CLI ('magic-pdf' or 'mineru') is installed
        mineru_bin = shutil.which("magic-pdf") or shutil.which("mineru")
        if mineru_bin:
            out_dir = tempfile.mkdtemp()
            try:
                cmd = [mineru_bin, "-p", pdf_path, "-o", out_dir]
                subprocess.run(cmd, capture_output=True, timeout=60, check=True)
                
                # Look for output markdown file
                for root, _, files in os.walk(out_dir):
                    for f in files:
                        if f.endswith(".md"):
                            with open(os.path.join(root, f), "r", encoding="utf-8") as md_f:
                                md_content = md_f.read()
                            title = os.path.basename(file_or_url)
                            return {"url": file_or_url, "content": md_content, "title": f"PDF: {title}", "source": "mineru"}
            except Exception as e:
                print(f"  [mineru] CLI execution failed ({e}) — falling back to python PDF parser")
            finally:
                shutil.rmtree(out_dir, ignore_errors=True)

        # Fallback: PyPDF / pdfplumber / pypdfium2 extraction
        extracted_text = ""
        try:
            import pypdf
            reader = pypdf.PdfReader(pdf_path)
            pages_text = []
            for i, page in enumerate(reader.pages):
                txt = page.extract_text()
                if txt:
                    pages_text.append(f"## Page {i+1}\n\n{txt}")
            extracted_text = "\n\n".join(pages_text)
        except Exception:
            logging.getLogger(__name__).debug("ignored error", exc_info=True)

        if not extracted_text:
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(pdf_path)
                pages_text = [f"## Page {i+1}\n\n{page.get_text()}" for i, page in enumerate(doc)]
                extracted_text = "\n\n".join(pages_text)
            except Exception:
                logging.getLogger(__name__).debug("ignored error", exc_info=True)

        title = os.path.basename(file_or_url)
        return {
            "url": file_or_url,
            "content": extracted_text or f"PDF document at {file_or_url}",
            "title": f"PDF: {title}",
            "source": "mineru_fallback"
        }
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError:
                pass


def mineru_search(query: str, max_results: int = 5) -> List[Dict]:
    """Search academic PDFs via arXiv API (zero key) for MinerU extraction pipeline."""
    if not query.strip():
        return []
    try:
        import urllib.parse
        import xml.etree.ElementTree as ET

        q = urllib.parse.quote(query)
        url = (
            f"http://export.arxiv.org/api/query?search_query=all:{q}"
            f"&start=0&max_results={min(max_results, 10)}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "AutonomousResearchAgent/1.0"})
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            xml = resp.read().decode("utf-8", errors="ignore")
        root = ET.fromstring(xml)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        results: List[Dict] = []
        for entry in root.findall("a:entry", ns):
            title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
            summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
            pdf_url = ""
            for link in entry.findall("a:link", ns):
                if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                    pdf_url = link.attrib.get("href", "")
                    break
            if not pdf_url:
                id_url = entry.findtext("a:id", default="", namespaces=ns) or ""
                if "arxiv.org/abs/" in id_url:
                    pdf_url = id_url.replace("/abs/", "/pdf/") + ".pdf"
            if pdf_url:
                results.append({
                    "title": title.replace("\n", " "),
                    "url": pdf_url,
                    "content": summary[:800],
                    "raw_content": summary,
                    "score": 0.75,
                    "source": "arxiv-mineru",
                })
        return results[:max_results]
    except Exception as e:
        print(f"  [mineru] arxiv search failed ({e})")
        return []


def mineru_extract(urls: List[str]) -> List[Dict]:
    """Extract content from multiple PDF URLs using MinerU adapter."""
    results = []
    for url in urls[:5]:
        if url.lower().endswith(".pdf") or "/pdf/" in url.lower() or "arxiv.org" in url.lower():
            parsed = mineru_parse_pdf(url)
            if parsed and parsed.get("content"):
                results.append(parsed)
    return results
