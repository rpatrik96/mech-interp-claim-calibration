"""Download arXiv papers and extract plain text for the annotation pipeline.

Reads paper IDs from candidate_papers.csv, downloads PDFs via the arxiv package,
extracts text with PyMuPDF (fitz), and saves to paper_texts/{arxiv_id}.txt.
Maintains SHA-256 checksums in paper_texts/checksums.json for reproducibility.
"""

import argparse
import csv
import hashlib
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import fitz
import requests

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "candidate_papers.csv"
TEXTS_DIR = SCRIPT_DIR / "paper_texts"
CHECKSUMS_PATH = TEXTS_DIR / "checksums.json"

SKIP_METHODS = frozenset(
    {
        "Review/Survey",
        "Position Paper",
        "Benchmark",
        "Survey",
        "Survey/Benchmark",
    }
)


def sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checksums() -> dict[str, str]:
    """Load existing checksums from disk."""
    if CHECKSUMS_PATH.exists():
        return json.loads(CHECKSUMS_PATH.read_text())
    return {}


def save_checksums(checksums: dict[str, str]) -> None:
    """Write checksums to disk."""
    CHECKSUMS_PATH.write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n")


def verify_checksums() -> bool:
    """Assert that all stored checksums match the files on disk.

    Returns True if all checksums match, False otherwise.
    """
    checksums = load_checksums()
    if not checksums:
        logger.warning("No checksums found — nothing to verify.")
        return True

    ok = True
    for arxiv_id, expected in checksums.items():
        txt_path = TEXTS_DIR / f"{arxiv_id}.txt"
        if not txt_path.exists():
            logger.error("Missing file for %s: %s", arxiv_id, txt_path)
            ok = False
            continue
        actual = sha256(txt_path)
        if actual != expected:
            logger.error(
                "Checksum mismatch for %s: expected %s, got %s",
                arxiv_id,
                expected,
                actual,
            )
            ok = False
        else:
            logger.debug("Checksum OK: %s", arxiv_id)
    return ok


def is_arxiv_id(paper_id: str) -> bool:
    """Heuristic: arXiv IDs look like YYMM.NNNNN or contain a dot with digits."""
    parts = paper_id.strip().split(".")
    if len(parts) != 2:
        return False
    return parts[0].isdigit() and parts[1].isdigit()


def read_candidate_papers(
    filter_ids: set[str] | None = None,
) -> list[dict[str, str]]:
    """Read candidate_papers.csv and filter to downloadable arXiv papers."""
    papers = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paper_id = row["paper_id"].strip()
            primary_method = row.get("primary_method", "").strip()

            if not is_arxiv_id(paper_id):
                logger.debug("Skipping non-arXiv ID: %s", paper_id)
                continue

            if primary_method in SKIP_METHODS:
                logger.info(
                    "Skipping %s (%s): %s", paper_id, row.get("title", ""), primary_method
                )
                continue

            if filter_ids is not None and paper_id not in filter_ids:
                continue

            papers.append(row)
    return papers


def download_and_extract(paper_id: str) -> str:
    """Download a single arXiv paper PDF directly and return extracted text."""
    pdf_url = f"https://arxiv.org/pdf/{paper_id}"

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / f"{paper_id.replace('/', '_')}.pdf"
        resp = requests.get(
            pdf_url, headers={"User-Agent": "fetch_papers/1.0"}, timeout=60
        )
        resp.raise_for_status()
        pdf_path.write_bytes(resp.content)
        logger.debug("Downloaded PDF: %s (%d bytes)", pdf_path.name, pdf_path.stat().st_size)

        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            pages.append(page.get_text(sort=True))
        doc.close()

    return "\n\n".join(pages)


def fetch_papers(filter_ids: set[str] | None = None) -> None:
    """Main fetch loop: download, extract, checksum."""
    TEXTS_DIR.mkdir(exist_ok=True)
    checksums = load_checksums()

    papers = read_candidate_papers(filter_ids=filter_ids)
    logger.info("Processing %d papers", len(papers))

    for i, row in enumerate(papers):
        paper_id = row["paper_id"].strip()
        title = row.get("title", "")
        txt_path = TEXTS_DIR / f"{paper_id}.txt"

        # Idempotent: skip if file exists and checksum matches
        if txt_path.exists() and paper_id in checksums:
            actual = sha256(txt_path)
            if actual == checksums[paper_id]:
                logger.info("Already fetched (checksum OK): %s — %s", paper_id, title)
                continue
            else:
                logger.warning(
                    "Checksum mismatch for existing %s — re-downloading", paper_id
                )

        # Rate-limit: wait between actual downloads to avoid 429s
        if i > 0:
            logger.debug("Sleeping 5s between downloads to respect rate limits")
            time.sleep(5)

        try:
            logger.info("Downloading: %s — %s", paper_id, title)
            text = download_and_extract(paper_id)
        except Exception:
            logger.exception("Failed to download %s — skipping", paper_id)
            continue

        txt_path.write_text(text, encoding="utf-8")
        checksums[paper_id] = sha256(txt_path)
        save_checksums(checksums)
        logger.info("Saved: %s (%d chars)", txt_path.name, len(text))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download arXiv papers and extract text for annotation."
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing checksums, do not download.",
    )
    parser.add_argument(
        "--paper-ids",
        type=str,
        default=None,
        help="Comma-separated arXiv IDs to process (default: all eligible).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.verify_only:
        ok = verify_checksums()
        if ok:
            logger.info("All checksums verified.")
        else:
            logger.error("Checksum verification FAILED.")
            sys.exit(1)
        return

    filter_ids = None
    if args.paper_ids:
        filter_ids = {pid.strip() for pid in args.paper_ids.split(",")}

    fetch_papers(filter_ids=filter_ids)


if __name__ == "__main__":
    main()
