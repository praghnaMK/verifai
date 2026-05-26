# ============================================================
#  ingestor.py  —  Document ingestion and chunking
#
#  Supports: PDF, TXT, MD
#  Output: List of Chunk objects with full provenance metadata
# ============================================================

import re
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# PDF extraction
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from .config import config


@dataclass
class Chunk:
    """A single text chunk with full provenance."""
    chunk_id:    str        # SHA256 hash of content
    doc_id:      str        # Source document identifier
    doc_name:    str        # Human-readable filename
    page_num:    int        # Page number (0-indexed, -1 if unknown)
    chunk_index: int        # Position in document
    text:        str        # Raw chunk text
    char_start:  int        # Character offset in document
    char_end:    int        # Character offset end
    embedding:   List[float] = field(default_factory=list)  # Set later

    def __repr__(self):
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk({self.doc_name}:p{self.page_num}:{self.chunk_index} '{preview}...')"


class DocumentIngestor:
    """
    Ingests documents into chunks with provenance tracking.

    Design decisions:
    - Sentence-aware chunking: never splits mid-sentence
    - Overlapping windows: context preserved across chunk boundaries
    - Page-level metadata: enables precise citation (page X, paragraph Y)
    """

    def __init__(self):
        self.chunk_size    = config.CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP

    # ── Public API ───────────────────────────────────────────

    def ingest(self, file_path: str | Path) -> List[Chunk]:
        """Ingest a document and return a list of Chunk objects."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        ext = path.suffix.lower()
        doc_id   = self._doc_id(path)
        doc_name = path.name

        if ext == ".pdf":
            pages = self._extract_pdf(path)
        elif ext in (".txt", ".md"):
            pages = self._extract_text(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Use PDF, TXT, or MD.")

        chunks = self._chunk_pages(pages, doc_id, doc_name)
        return chunks

    # ── Extraction ───────────────────────────────────────────

    def _extract_pdf(self, path: Path) -> List[dict]:
        """Extract text per page from PDF using PyMuPDF."""
        if not PYMUPDF_AVAILABLE:
            raise ImportError(
                "PyMuPDF not installed. Run: pip install pymupdf"
            )
        doc = fitz.open(str(path))
        pages = []
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            text = self._clean_text(text)
            if text.strip():
                pages.append({"page": page_num, "text": text})
        doc.close()
        return pages

    def _extract_text(self, path: Path) -> List[dict]:
        """Extract text from TXT/MD — treat entire file as page 0."""
        text = path.read_text(encoding="utf-8", errors="replace")
        text = self._clean_text(text)
        return [{"page": 0, "text": text}]

    # ── Chunking ─────────────────────────────────────────────

    def _chunk_pages(
        self, pages: List[dict], doc_id: str, doc_name: str
    ) -> List[Chunk]:
        """
        Sentence-aware chunking with sliding window overlap.

        Algorithm:
        1. Split page text into sentences
        2. Pack sentences into chunks up to CHUNK_SIZE characters
        3. Overlap: carry last N characters into next chunk
        """
        chunks   = []
        char_pos = 0
        idx      = 0

        for page_info in pages:
            page_num  = page_info["page"]
            page_text = page_info["text"]
            sentences = self._split_sentences(page_text)

            current_text  = ""
            current_start = char_pos

            for sentence in sentences:
                # If adding this sentence exceeds limit → emit chunk
                if len(current_text) + len(sentence) > self.chunk_size and current_text:
                    chunk = self._make_chunk(
                        current_text, doc_id, doc_name, page_num,
                        idx, current_start, char_pos
                    )
                    chunks.append(chunk)
                    idx += 1

                    # Overlap: keep last CHUNK_OVERLAP characters
                    overlap_text  = current_text[-self.chunk_overlap:]
                    current_start = char_pos - len(overlap_text)
                    current_text  = overlap_text + " " + sentence
                else:
                    current_text += (" " if current_text else "") + sentence

                char_pos += len(sentence) + 1  # +1 for space

            # Emit remaining text as final chunk for this page
            if current_text.strip():
                chunk = self._make_chunk(
                    current_text, doc_id, doc_name, page_num,
                    idx, current_start, char_pos
                )
                chunks.append(chunk)
                idx += 1

        return chunks

    # ── Helpers ──────────────────────────────────────────────

    def _make_chunk(
        self, text: str, doc_id: str, doc_name: str,
        page_num: int, idx: int, start: int, end: int
    ) -> Chunk:
        text = text.strip()
        return Chunk(
            chunk_id    = hashlib.sha256(text.encode()).hexdigest()[:16],
            doc_id      = doc_id,
            doc_name    = doc_name,
            page_num    = page_num,
            chunk_index = idx,
            text        = text,
            char_start  = start,
            char_end    = end,
        )

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, protecting common abbreviations."""
        # Temporarily mask periods in abbreviations so they don't trigger splits
        abbrev_pattern = (
            r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|Fig|No|Vol|pp?|ed|eds|'
            r'et al|e\.g|i\.e|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.'
        )
        text = re.sub(abbrev_pattern, r'\1<DOT>', text, flags=re.IGNORECASE)

        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

        # Restore masked periods
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        return [s.strip() for s in sentences if s.strip()]

    def _clean_text(self, text: str) -> str:
        """Normalise whitespace and remove PDF artifacts, preserving valid Unicode."""
        text = re.sub(r'-\n', '', text)                        # Rejoin hyphenated line breaks
        text = re.sub(r'[ \t]+', ' ', text)                    # Collapse horizontal whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)                 # Collapse excessive blank lines
        text = re.sub(r'[\x00-\x08\x0b\x0e-\x1f\x7f]', '', text)  # Remove control chars only
        return text.strip()

    def _doc_id(self, path: Path) -> str:
        """Stable document ID from file path."""
        return hashlib.md5(str(path).encode()).hexdigest()[:12]
