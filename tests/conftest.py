"""Shared pytest fixtures for h2_pilot_study pipeline tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------

H2_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Minimal synthetic DataFrames
# ---------------------------------------------------------------------------

ORIGINAL_COLUMNS = [
    "paper_id",
    "claim_id",
    "claim_text",
    "claim_location",
    "claim_prominence",
    "method_used",
    "method_rung",
    "claim_rung",
    "gap_score",
    "confidence",
    "notes",
    "replication_status",
    "replication_evidence",
]


def _make_valid_row(**overrides) -> dict:
    """Return a single valid annotation row, with any field overridable."""
    base = {
        "paper_id": "2211.00593",
        "claim_id": "2211.00593-01",
        "claim_text": "The model uses attention heads to perform IOI.",
        "claim_location": "abstract",
        "claim_prominence": 3,
        "method_used": "Activation Patching",
        "method_rung": 2,
        "claim_rung": 3,
        "gap_score": 1,
        "confidence": 4,
        "notes": "",
        "replication_status": 0.5,
        "replication_evidence": "",
    }
    base.update(overrides)
    return base


@pytest.fixture()
def valid_annotations_df() -> pd.DataFrame:
    """Minimal valid annotations DataFrame with two rows."""
    rows = [
        _make_valid_row(claim_id="2211.00593-01", method_rung=2, claim_rung=3, gap_score=1),
        _make_valid_row(claim_id="2211.00593-02", method_rung=2, claim_rung=2, gap_score=0),
    ]
    return pd.DataFrame(rows, columns=ORIGINAL_COLUMNS)


@pytest.fixture()
def valid_candidate_papers_df() -> pd.DataFrame:
    """Minimal candidate_papers DataFrame."""
    return pd.DataFrame(
        [
            {
                "paper_id": "2211.00593",
                "title": "Interpretability in the Wild",
                "authors": "Wang et al.",
                "year": 2022,
                "venue": "NeurIPS",
                "primary_method": "Activation Patching",
                "expected_method_rung": 2,
                "arxiv_url": "https://arxiv.org/abs/2211.00593",
                "notes": "",
            }
        ]
    )


@pytest.fixture()
def annotations_csv(tmp_path, valid_annotations_df, valid_candidate_papers_df) -> tuple[Path, Path]:
    """Write synthetic annotations + candidate_papers CSVs and return (ann_path, cp_path)."""
    ann_path = tmp_path / "annotations.csv"
    cp_path = tmp_path / "candidate_papers.csv"
    valid_annotations_df.to_csv(ann_path, index=False)
    valid_candidate_papers_df.to_csv(cp_path, index=False)
    return ann_path, cp_path


# ---------------------------------------------------------------------------
# IAA fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def iaa_merged_df() -> pd.DataFrame:
    """Merged DataFrame with _claude and _gpt suffixes for three claims."""
    data = {
        "claim_id": ["p1-01", "p1-02", "p2-01"],
        "paper_id_claude": ["2211.00593", "2211.00593", "2202.05262"],
        "method_rung_claude": [2, 2, 1],
        "claim_rung_claude": [3, 2, 2],
        "gap_score_claude": [1, 0, 1],
        "confidence_claude": [4, 5, 3],
        "claim_location_claude": ["abstract", "body", "introduction"],
        "method_used_claude": ["Activation Patching", "Activation Patching", "Causal Tracing"],
        "method_rung_gpt": [2, 2, 1],
        "claim_rung_gpt": [3, 2, 2],
        "gap_score_gpt": [1, 0, 1],
        "confidence_gpt": [4, 5, 3],
        "claim_location_gpt": ["abstract", "body", "introduction"],
        "method_used_gpt": ["Activation Patching", "Activation Patching", "Causal Tracing"],
    }
    return pd.DataFrame(data)


@pytest.fixture()
def checksums_dir(tmp_path) -> Path:
    """A tmp directory set up as a paper_texts directory with one file and checksums.json."""
    texts_dir = tmp_path / "paper_texts"
    texts_dir.mkdir()
    return texts_dir
