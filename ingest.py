"""
Ingest sources from sources.csv: fetch content (PDF or HTML), chunk, embed with
sentence-transformers, and save a FAISS index + metadata. No LangChain. No API keys.
"""
import csv
import json
import os
import re
import sys
import tempfile
from typing import List, Tuple

import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# --- Config ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_DIR = "faiss_index"
METADATA_FILENAME = "metadata.json"


def get_sources_path() -> str:
    """Use sources.csv if present, otherwise Source.csv (for backwards compatibility)."""
    if os.path.exists("sources.csv"):
        return "sources.csv"
    if os.path.exists("Source.csv"):
        return "Source.csv"
    raise FileNotFoundError(
        "Missing sources file. Create sources.csv with columns: title,url (or rename Source.csv to sources.csv)"
    )


def read_sources(path: str) -> List[Tuple[str, str]]:
    """Read CSV with columns title, url. Returns list of (title, url)."""
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header. Expected: title, url")
        fieldnames_lower = {fn.strip().lower().replace(" ", ""): fn for fn in reader.fieldnames}
        title_key = fieldnames_lower.get("title") or next((v for k, v in fieldnames_lower.items() if "title" in k), None)
        url_key = fieldnames_lower.get("url") or next((v for k, v in fieldnames_lower.items() if "url" in k), None)
        if not title_key or not url_key:
            raise ValueError("CSV must have 'title' and 'url' columns")
        for i, row in enumerate(reader, start=2):
            title = (row.get(title_key) or "").strip()
            url = (row.get(url_key) or "").strip()
            if not title or not url:
                print(f"Skipping row {i}: missing title or url", file=sys.stderr)
                continue
            rows.append((title, url))
    return rows


def is_pdf_url(url: str, content_type: str | None) -> bool:
    if content_type and "pdf" in content_type.lower():
        return True
    return url.rstrip("/").lower().split("?")[0].endswith(".pdf")


def clean_text(text: str) -> str:
    """Normalize whitespace and remove excessive newlines."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_html_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return clean_text("\n".join(lines))


def fetch_url(title: str, url: str) -> Tuple[str | None, str | None]:
    """
    Fetch URL and return (text, error). If error, text is None and error is a string.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0",
        "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
    }
    try:
        # HEAD to check content type (some servers block HEAD, so we may use GET)
        content_type = None
        try:
            head = requests.head(url, headers=headers, timeout=15, allow_redirects=True)
            if head.ok:
                content_type = head.headers.get("Content-Type") or ""
        except Exception:
            pass

        if is_pdf_url(url, content_type):
            with requests.get(url, headers=headers, timeout=60, stream=True) as r:
                r.raise_for_status()
                fd, tmp = tempfile.mkstemp(suffix=".pdf")
                try:
                    with os.fdopen(fd, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                    reader = PdfReader(tmp)
                    parts = []
                    for page in reader.pages:
                        t = (page.extract_text() or "").strip()
                        if t:
                            parts.append(t)
                    text = "\n\n".join(parts)
                    return clean_text(text), None
                finally:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
        else:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            text = extract_html_text(resp.text)
            return text if text else None, "empty page content"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into chunks by character count with overlap. Pure Python."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if not chunk.strip():
            start = end - overlap
            continue
        chunks.append(chunk.strip())
        start = end - overlap
        if start >= len(text):
            break
    return chunks


def build_chunks_with_metadata(
    sources_path: str,
) -> Tuple[List[dict], List[Tuple[str, str]]]:
    """
    Load all sources, extract text, chunk, and return (metadata_list, failed_list).
    Each metadata item: {"text": chunk_text, "title": title, "source": url}.
    """
    rows = read_sources(sources_path)
    all_metadata = []
    failed = []

    for title, url in rows:
        text, err = fetch_url(title, url)
        if err or not (text and text.strip()):
            failed.append((title, err or "empty text"))
            print(f"FAILED: {title} - {err or 'empty text'}", file=sys.stderr)
            continue
        for chunk in chunk_text(text):
            if len(chunk) < 20:
                continue
            all_metadata.append({"text": chunk, "title": title, "source": url})

    return all_metadata, failed


def main() -> None:
    sources_path = get_sources_path()
    print(f"Using sources file: {sources_path}")
    total_urls = len(read_sources(sources_path))

    metadata_list, failed = build_chunks_with_metadata(sources_path)
    successful_docs = total_urls - len(failed)
    total_chunks = len(metadata_list)

    if not metadata_list:
        print("No chunks created. Fix failed URLs or add more sources. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [m["text"] for m in metadata_list]
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    index_path = os.path.join(FAISS_INDEX_DIR, "index.faiss")
    faiss.write_index(index, index_path)

    metadata_path = os.path.join(FAISS_INDEX_DIR, METADATA_FILENAME)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=0)

    print("---")
    print(f"Total URLs in CSV: {total_urls}")
    print(f"Successful documents: {successful_docs}")
    print(f"Failed documents: {len(failed)}")
    print(f"Total chunks: {total_chunks}")
    print(f"FAISS index and metadata saved to: {FAISS_INDEX_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
