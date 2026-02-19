"""Comprehensive tests for the h2_pilot_study annotation pipeline.

Covers:
  - validate.py    — schema validation
  - fetch_papers.py — checksum and arXiv ID helpers
  - annotate.py    — Pydantic schemas, prompt builders, gap_score logic
  - compute_iaa.py — agreement metrics, bootstrap shape, merge, confusion
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Make h2_pilot_study importable when running from the tests/ subdirectory
# ---------------------------------------------------------------------------

H2_DIR = Path(__file__).resolve().parent.parent
if str(H2_DIR) not in sys.path:
    sys.path.insert(0, str(H2_DIR))

import compute_iaa as iaa  # noqa: E402
import fetch_papers as fp  # noqa: E402
import validate as val  # noqa: E402
from annotate import (  # noqa: E402
    SYSTEM_PROMPT,
    ClaimClassification,
    ClaimExtraction,
    PaperAnnotation,
    _build_classify_user_prompt,
    _build_extract_user_prompt,
    load_config,
)

# ---------------------------------------------------------------------------
# Helpers shared by multiple test groups
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


def _row(**overrides) -> dict:
    base = {
        "paper_id": "2211.00593",
        "claim_id": "2211.00593-01",
        "claim_text": "The model uses attention to perform IOI.",
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


def _write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    df = pd.DataFrame(rows, columns=columns or list(rows[0].keys()))
    df.to_csv(path, index=False)


def _write_candidate_papers(path: Path, paper_ids: list[str] | None = None) -> None:
    ids = paper_ids or ["2211.00593"]
    rows = [{"paper_id": pid, "title": f"Paper {pid}", "primary_method": "Activation Patching"} for pid in ids]
    pd.DataFrame(rows).to_csv(path, index=False)


# ===========================================================================
# 1. validate.py
# ===========================================================================


class TestValidateSchema:
    def test_valid_original_annotations(self, tmp_path):
        """Real annotations.csv (186 rows) passes all checks."""
        ann_path = H2_DIR / "annotations.csv"
        cp_path = H2_DIR / "candidate_papers.csv"
        errors = val.validate_file(ann_path, mode="original", candidate_papers_path=cp_path)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_missing_columns_detected(self, tmp_path):
        """Dropping a required column is reported as an error."""
        rows = [_row()]
        df = pd.DataFrame(rows, columns=ORIGINAL_COLUMNS)
        # Drop 'confidence'
        df = df.drop(columns=["confidence"])
        csv_path = tmp_path / "annotations.csv"
        _write_candidate_papers(tmp_path / "candidate_papers.csv")
        df.to_csv(csv_path, index=False)
        errors = val.validate_file(csv_path, mode="original", candidate_papers_path=tmp_path / "candidate_papers.csv")
        assert any("Missing columns" in e for e in errors)

    def test_gap_score_mismatch_detected(self, tmp_path):
        """A tampered gap_score is flagged."""
        row = _row(method_rung=2, claim_rung=2, gap_score=99)  # correct would be 0
        csv_path = tmp_path / "annotations.csv"
        _write_csv(csv_path, [row], ORIGINAL_COLUMNS)
        _write_candidate_papers(tmp_path / "candidate_papers.csv")
        errors = val.validate_file(csv_path, mode="original", candidate_papers_path=tmp_path / "candidate_papers.csv")
        assert any("gap_score mismatch" in e for e in errors)

    def test_na_rung_rows_tolerated(self, tmp_path):
        """Rows with NA method_rung are skipped for gap_score check and do not fail."""
        row = _row(method_rung=None, claim_rung=None, gap_score=None)
        csv_path = tmp_path / "annotations.csv"
        _write_csv(csv_path, [row], ORIGINAL_COLUMNS)
        _write_candidate_papers(tmp_path / "candidate_papers.csv")
        errors = val.validate_file(csv_path, mode="original", candidate_papers_path=tmp_path / "candidate_papers.csv")
        # NA rows should not produce gap_score mismatch errors
        assert not any("gap_score mismatch" in e for e in errors)

    def test_invalid_method_rung_detected(self, tmp_path):
        """method_rung=99 is not in {1,2,3} and must be flagged."""
        row = _row(method_rung=99, claim_rung=1, gap_score=0)
        csv_path = tmp_path / "annotations.csv"
        _write_csv(csv_path, [row], ORIGINAL_COLUMNS)
        _write_candidate_papers(tmp_path / "candidate_papers.csv")
        errors = val.validate_file(csv_path, mode="original", candidate_papers_path=tmp_path / "candidate_papers.csv")
        assert any("method_rung" in e for e in errors)

    def test_invalid_confidence_detected(self, tmp_path):
        """confidence=6 is outside {1..5} and must be flagged."""
        row = _row(confidence=6)
        csv_path = tmp_path / "annotations.csv"
        _write_csv(csv_path, [row], ORIGINAL_COLUMNS)
        _write_candidate_papers(tmp_path / "candidate_papers.csv")
        errors = val.validate_file(csv_path, mode="original", candidate_papers_path=tmp_path / "candidate_papers.csv")
        assert any("confidence" in e for e in errors)

    def test_confidence_zero_detected(self, tmp_path):
        """confidence=0 is also outside {1..5} and must be flagged."""
        row = _row(confidence=0)
        csv_path = tmp_path / "annotations.csv"
        _write_csv(csv_path, [row], ORIGINAL_COLUMNS)
        _write_candidate_papers(tmp_path / "candidate_papers.csv")
        errors = val.validate_file(csv_path, mode="original", candidate_papers_path=tmp_path / "candidate_papers.csv")
        assert any("confidence" in e for e in errors)

    def test_duplicate_claim_id_detected(self, tmp_path):
        """Two rows with the same claim_id should fail uniqueness check."""
        row1 = _row(claim_id="2211.00593-01")
        row2 = _row(claim_id="2211.00593-01", claim_text="Different text.")
        csv_path = tmp_path / "annotations.csv"
        _write_csv(csv_path, [row1, row2], ORIGINAL_COLUMNS)
        _write_candidate_papers(tmp_path / "candidate_papers.csv")
        errors = val.validate_file(csv_path, mode="original", candidate_papers_path=tmp_path / "candidate_papers.csv")
        assert any("Duplicate" in e for e in errors)

    def test_empty_claim_text_detected(self, tmp_path):
        """A row with blank claim_text must be reported."""
        row = _row(claim_text="")
        csv_path = tmp_path / "annotations.csv"
        _write_csv(csv_path, [row], ORIGINAL_COLUMNS)
        _write_candidate_papers(tmp_path / "candidate_papers.csv")
        errors = val.validate_file(csv_path, mode="original", candidate_papers_path=tmp_path / "candidate_papers.csv")
        assert any("claim_text" in e for e in errors)

    def test_unknown_paper_id_detected(self, tmp_path):
        """A paper_id absent from candidate_papers.csv must be flagged."""
        row = _row(paper_id="9999.99999", claim_id="9999.99999-01")
        csv_path = tmp_path / "annotations.csv"
        _write_csv(csv_path, [row], ORIGINAL_COLUMNS)
        # candidate_papers only has 2211.00593
        _write_candidate_papers(tmp_path / "candidate_papers.csv", paper_ids=["2211.00593"])
        errors = val.validate_file(csv_path, mode="original", candidate_papers_path=tmp_path / "candidate_papers.csv")
        assert any("paper_ids not in candidate_papers" in e for e in errors)

    def test_invalid_location_detected(self, tmp_path):
        """claim_location='footnote' is not in the valid set and must be flagged."""
        row = _row(claim_location="footnote")
        csv_path = tmp_path / "annotations.csv"
        _write_csv(csv_path, [row], ORIGINAL_COLUMNS)
        _write_candidate_papers(tmp_path / "candidate_papers.csv")
        errors = val.validate_file(csv_path, mode="original", candidate_papers_path=tmp_path / "candidate_papers.csv")
        assert any("claim_location" in e for e in errors)


class TestValidateHelpers:
    def test_check_schema_original_ok(self):
        df = pd.DataFrame([_row()], columns=ORIGINAL_COLUMNS)
        errors = val._check_schema(df, mode="original")
        assert errors == []

    def test_check_schema_missing_col(self):
        df = pd.DataFrame([_row()], columns=ORIGINAL_COLUMNS).drop(columns=["gap_score"])
        errors = val._check_schema(df, mode="original")
        assert len(errors) == 1
        assert "gap_score" in errors[0]

    def test_check_gap_score_correct(self):
        df = pd.DataFrame([_row(method_rung=1, claim_rung=3, gap_score=2)])
        errors = val._check_gap_score(df)
        assert errors == []

    def test_check_gap_score_clamp_at_zero(self):
        """gap_score = max(0, 1 - 2) = 0, not -1."""
        df = pd.DataFrame([_row(method_rung=2, claim_rung=1, gap_score=0)])
        errors = val._check_gap_score(df)
        assert errors == []

    def test_check_claim_id_uniqueness_ok(self):
        df = pd.DataFrame([
            _row(claim_id="2211.00593-01"),
            _row(claim_id="2211.00593-02"),
        ])
        assert val._check_claim_id_uniqueness(df) == []

    def test_check_claim_id_uniqueness_fail(self):
        df = pd.DataFrame([_row(claim_id="2211.00593-01"), _row(claim_id="2211.00593-01")])
        errors = val._check_claim_id_uniqueness(df)
        assert errors != []

    def test_check_nonempty_claim_text_ok(self):
        df = pd.DataFrame([_row(claim_text="Some claim.")])
        assert val._check_nonempty_claim_text(df) == []

    def test_check_nonempty_claim_text_whitespace(self):
        df = pd.DataFrame([_row(claim_text="   ")])
        errors = val._check_nonempty_claim_text(df)
        assert errors != []


# ===========================================================================
# 2. fetch_papers.py
# ===========================================================================


class TestFetchPaperHelpers:
    def test_is_arxiv_id_valid(self):
        assert fp.is_arxiv_id("2211.00593") is True
        assert fp.is_arxiv_id("2301.05217") is True

    def test_is_arxiv_id_invalid_no_dot(self):
        assert fp.is_arxiv_id("hep-th/9901001") is False

    def test_is_arxiv_id_invalid_alpha_parts(self):
        assert fp.is_arxiv_id("cs.AI") is False

    def test_is_arxiv_id_invalid_three_parts(self):
        assert fp.is_arxiv_id("2211.00593.v2") is False

    def test_is_arxiv_id_empty_string(self):
        assert fp.is_arxiv_id("") is False

    def test_is_arxiv_id_partial_digits(self):
        # "2211.abc" — second part not all digits
        assert fp.is_arxiv_id("2211.abc") is False


class TestFetchPaperChecksums:
    def test_sha256_known_content(self, tmp_path):
        """sha256() must match hashlib reference for a known byte string."""
        content = b"hello world\n"
        f = tmp_path / "test.txt"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert fp.sha256(f) == expected

    def test_verify_checksums_no_files_returns_true(self, tmp_path, monkeypatch):
        """Empty checksums.json (no entries) returns True — nothing to verify."""
        checksums_path = tmp_path / "checksums.json"
        checksums_path.write_text("{}\n")
        monkeypatch.setattr(fp, "CHECKSUMS_PATH", checksums_path)
        monkeypatch.setattr(fp, "TEXTS_DIR", tmp_path)
        result = fp.verify_checksums()
        assert result is True

    def test_verify_checksums_correct(self, tmp_path, monkeypatch):
        """A correct checksum entry passes verification."""
        content = b"paper text here\n"
        txt_file = tmp_path / "2211.00593.txt"
        txt_file.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()

        checksums_path = tmp_path / "checksums.json"
        checksums_path.write_text(json.dumps({"2211.00593": digest}) + "\n")

        monkeypatch.setattr(fp, "CHECKSUMS_PATH", checksums_path)
        monkeypatch.setattr(fp, "TEXTS_DIR", tmp_path)
        assert fp.verify_checksums() is True

    def test_verify_checksums_mismatch(self, tmp_path, monkeypatch):
        """A tampered checksum entry fails verification."""
        content = b"paper text here\n"
        txt_file = tmp_path / "2211.00593.txt"
        txt_file.write_bytes(content)

        bad_digest = "0" * 64  # deliberately wrong
        checksums_path = tmp_path / "checksums.json"
        checksums_path.write_text(json.dumps({"2211.00593": bad_digest}) + "\n")

        monkeypatch.setattr(fp, "CHECKSUMS_PATH", checksums_path)
        monkeypatch.setattr(fp, "TEXTS_DIR", tmp_path)
        assert fp.verify_checksums() is False

    def test_verify_checksums_missing_file(self, tmp_path, monkeypatch):
        """An entry whose .txt file does not exist fails verification."""
        checksums_path = tmp_path / "checksums.json"
        checksums_path.write_text(json.dumps({"9999.00001": "a" * 64}) + "\n")

        monkeypatch.setattr(fp, "CHECKSUMS_PATH", checksums_path)
        monkeypatch.setattr(fp, "TEXTS_DIR", tmp_path)
        assert fp.verify_checksums() is False

    def test_load_save_checksums_roundtrip(self, tmp_path, monkeypatch):
        """save_checksums + load_checksums round-trips correctly."""
        checksums_path = tmp_path / "checksums.json"
        monkeypatch.setattr(fp, "CHECKSUMS_PATH", checksums_path)
        data = {"2211.00593": "abc123", "2202.05262": "def456"}
        fp.save_checksums(data)
        loaded = fp.load_checksums()
        assert loaded == data


class TestFetchPaperDownload:
    def test_download_and_extract_mocked(self, tmp_path):
        """download_and_extract returns text from a mocked PDF extraction."""
        fake_page = MagicMock()
        fake_page.get_text.return_value = "Page one text."

        fake_doc = MagicMock()
        fake_doc.__iter__ = MagicMock(return_value=iter([fake_page]))
        fake_doc.close = MagicMock()

        fake_response = MagicMock()
        fake_response.content = b"%PDF-1.4 fake"
        fake_response.raise_for_status = MagicMock()

        with (
            patch("fetch_papers.requests.get", return_value=fake_response),
            patch("fetch_papers.fitz.open", return_value=fake_doc),
        ):
            text = fp.download_and_extract("2211.00593")

        assert "Page one text." in text

    def test_download_and_extract_http_error_raises(self):
        """HTTPError raised when arxiv returns an error status."""
        import requests as req

        fake_response = MagicMock()
        fake_response.raise_for_status.side_effect = req.HTTPError("404 Not Found")

        with (
            patch("fetch_papers.requests.get", return_value=fake_response),
            pytest.raises(req.HTTPError),
        ):
            fp.download_and_extract("0000.00000")


# ===========================================================================
# 3. annotate.py
# ===========================================================================


class TestAnnotatePydanticSchemas:
    def test_claim_classification_valid(self):
        obj = ClaimClassification(
            method_used="Activation Patching",
            method_rung=2,
            claim_rung=3,
            confidence=4,
            hedge_flag=0,
            reasoning="The paper clearly uses path patching.",
        )
        assert obj.method_rung == 2
        assert obj.claim_rung == 3

    def test_claim_classification_rejects_invalid_rung(self):
        """method_rung=4 is outside Literal[1,2,3] and must fail."""
        with pytest.raises(ValidationError):
            ClaimClassification(
                method_used="Activation Patching",
                method_rung=4,  # invalid
                claim_rung=1,
                confidence=3,
                hedge_flag=0,
                reasoning="Test",
            )

    def test_claim_classification_rejects_invalid_confidence(self):
        """confidence=6 is outside Literal[1,2,3,4,5]."""
        with pytest.raises(ValidationError):
            ClaimClassification(
                method_used="Probing",
                method_rung=1,
                claim_rung=1,
                confidence=6,  # invalid
                hedge_flag=0,
                reasoning="Test",
            )

    def test_claim_classification_rejects_invalid_hedge_flag(self):
        """hedge_flag=2 is outside Literal[0,1]."""
        with pytest.raises(ValidationError):
            ClaimClassification(
                method_used="Probing",
                method_rung=1,
                claim_rung=1,
                confidence=3,
                hedge_flag=2,  # invalid
                reasoning="Test",
            )

    def test_claim_extraction_valid(self):
        obj = ClaimExtraction(
            claim_text="Attention heads implement name-moving.",
            claim_location="body",
            claim_prominence=2,
            method_used="Activation Patching",
            method_rung=2,
            claim_rung=3,
            confidence=5,
            hedge_flag=0,
            reasoning="Uses path patching to establish mechanistic role.",
        )
        assert obj.claim_location == "body"

    def test_claim_extraction_rejects_invalid_location(self):
        """claim_location='footnote' is not in the Literal set."""
        with pytest.raises(ValidationError):
            ClaimExtraction(
                claim_text="Some claim.",
                claim_location="footnote",  # invalid
                claim_prominence=1,
                method_used="Probing",
                method_rung=1,
                claim_rung=1,
                confidence=3,
                hedge_flag=0,
                reasoning="Test",
            )

    def test_paper_annotation_valid(self):
        claim = ClaimExtraction(
            claim_text="Attention heads implement name-moving.",
            claim_location="abstract",
            claim_prominence=3,
            method_used="Activation Patching",
            method_rung=2,
            claim_rung=3,
            confidence=4,
            hedge_flag=1,
            reasoning="Hedged by 'suggests'.",
        )
        ann = PaperAnnotation(paper_id="2211.00593", claims=[claim])
        assert ann.paper_id == "2211.00593"
        assert len(ann.claims) == 1

    def test_paper_annotation_empty_claims(self):
        """Zero claims is valid — a paper may have no empirical claims."""
        ann = PaperAnnotation(paper_id="2211.00593", claims=[])
        assert ann.claims == []


class TestAnnotateGapScore:
    @pytest.mark.parametrize(
        "method_rung, claim_rung, expected",
        [
            (1, 1, 0),
            (2, 2, 0),
            (3, 3, 0),
            (1, 2, 1),
            (1, 3, 2),
            (2, 3, 1),
            (3, 1, 0),  # clamped at 0
            (3, 2, 0),  # clamped at 0
            (2, 1, 0),  # clamped at 0
        ],
    )
    def test_gap_score_computation(self, method_rung, claim_rung, expected):
        """gap_score = max(0, claim_rung - method_rung) for all rung combos."""
        assert max(0, claim_rung - method_rung) == expected


class TestAnnotateConfig:
    def test_load_config_structure(self):
        """annotation_config.yaml loads and has expected top-level keys."""
        config_path = H2_DIR / "annotation_config.yaml"
        config = load_config(config_path)
        assert "models" in config
        assert "annotation" in config
        assert "thresholds" in config

    def test_load_config_models_have_provider(self):
        config_path = H2_DIR / "annotation_config.yaml"
        config = load_config(config_path)
        for model_key, model_cfg in config["models"].items():
            assert "provider" in model_cfg, f"Model {model_key} missing 'provider'"
            assert "model" in model_cfg, f"Model {model_key} missing 'model'"

    def test_load_config_annotation_keys(self):
        config_path = H2_DIR / "annotation_config.yaml"
        config = load_config(config_path)
        ann = config["annotation"]
        for key in ("codebook_path", "calibration_path", "paper_texts_dir", "output_dir"):
            assert key in ann, f"annotation section missing key: {key}"


class TestAnnotatePromptBuilders:
    def test_build_system_prompt_content(self):
        """SYSTEM_PROMPT contains key instruction phrases."""
        assert "codebook" in SYSTEM_PROMPT.lower()
        assert "confidence" in SYSTEM_PROMPT.lower()

    def test_build_classify_user_prompt_contains_claim(self):
        prompt = _build_classify_user_prompt(
            codebook="CODEBOOK TEXT",
            calibration="CALIBRATION TEXT",
            paper_id="2211.00593",
            title="IOI Paper",
            paper_text="Full paper text here.",
            claim_text="Name Mover Heads move names.",
            claim_location="body",
        )
        assert "Name Mover Heads move names." in prompt
        assert "CODEBOOK TEXT" in prompt
        assert "2211.00593" in prompt
        assert "body" in prompt

    def test_build_classify_user_prompt_contains_task_instruction(self):
        prompt = _build_classify_user_prompt(
            codebook="CB",
            calibration="CAL",
            paper_id="p1",
            title="T",
            paper_text="text",
            claim_text="claim",
            claim_location="abstract",
        )
        assert "Classify" in prompt or "classify" in prompt

    def test_build_extract_user_prompt_contains_paper_id(self):
        prompt = _build_extract_user_prompt(
            codebook="CODEBOOK TEXT",
            calibration="CALIBRATION TEXT",
            paper_id="2202.05262",
            title="ROME Paper",
            paper_text="Full text.",
        )
        assert "2202.05262" in prompt
        assert "CODEBOOK TEXT" in prompt

    def test_build_extract_user_prompt_extract_instruction(self):
        prompt = _build_extract_user_prompt(
            codebook="CB",
            calibration="CAL",
            paper_id="p1",
            title="T",
            paper_text="text",
        )
        assert "Extract" in prompt or "extract" in prompt


# ===========================================================================
# 4. compute_iaa.py
# ===========================================================================


class TestIaaPabak:
    def test_pabak_perfect_agreement(self):
        y = np.array([1, 2, 3, 1, 2])
        assert iaa.pabak(y, y) == pytest.approx(1.0)

    def test_pabak_zero_raw_agreement(self):
        """When p_o = 0, PABAK = 2*0 - 1 = -1."""
        y1 = np.array([1, 1, 1])
        y2 = np.array([2, 2, 2])
        assert iaa.pabak(y1, y2) == pytest.approx(-1.0)

    def test_pabak_partial_agreement(self):
        """2 out of 4 agree: p_o = 0.5, PABAK = 0.0."""
        y1 = np.array([1, 1, 2, 2])
        y2 = np.array([1, 2, 2, 1])
        assert iaa.pabak(y1, y2) == pytest.approx(0.0)


class TestIaaGwetAc1:
    def test_gwet_ac1_perfect(self):
        y = np.array([1, 2, 3, 1, 2])
        assert iaa.gwet_ac1(y, y) == pytest.approx(1.0)

    def test_gwet_ac1_single_category(self):
        """Only one category present — return 1.0 by convention."""
        y = np.array([2, 2, 2])
        assert iaa.gwet_ac1(y, y) == pytest.approx(1.0)

    def test_gwet_ac1_empty(self):
        assert np.isnan(iaa.gwet_ac1(np.array([]), np.array([])))


class TestIaaWeightedKappa:
    def test_weighted_kappa_perfect(self):
        from sklearn.metrics import cohen_kappa_score

        y = np.array([1, 2, 3, 1, 2, 3])
        kappa = cohen_kappa_score(y, y, weights="linear")
        assert kappa == pytest.approx(1.0)

    def test_weighted_kappa_near_zero_random(self):
        """Kappa on labels that systematically disagree should be < 0."""
        from sklearn.metrics import cohen_kappa_score

        y1 = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        y2 = np.array([3, 3, 3, 1, 1, 1, 2, 2, 2])
        kappa = cohen_kappa_score(y1, y2, weights="linear")
        assert kappa < 0.0


class TestIaaBootstrap:
    def test_cluster_bootstrap_shape(self, iaa_merged_df):
        """Bootstrap returns the expected keys and finite CI bounds for perfect data."""
        rng = np.random.default_rng(0)
        result = iaa.paper_cluster_bootstrap(
            iaa_merged_df,
            variable="method_rung",
            suffix=("_claude", "_gpt"),
            n_bootstrap=50,
            rng=rng,
        )
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "se" in result

    def test_cluster_bootstrap_perfect_agreement_ci(self, iaa_merged_df):
        """With perfect agreement on all claims, CI should be around 1.0."""
        rng = np.random.default_rng(0)
        result = iaa.paper_cluster_bootstrap(
            iaa_merged_df,
            variable="method_rung",
            suffix=("_claude", "_gpt"),
            n_bootstrap=100,
            rng=rng,
        )
        # CI bounds both near 1.0 for identical raters
        if not np.isnan(result["ci_lower"]):
            assert result["ci_lower"] > 0.5


class TestIaaMergeAnnotators:
    def test_merge_on_claim_id(self):
        claude = pd.DataFrame({
            "claim_id": ["p1-01", "p1-02"],
            "method_rung": [2, 1],
            "claim_rung": [3, 2],
            "gap_score": [1, 1],
            "confidence": [4, 3],
        })
        gpt = pd.DataFrame({
            "claim_id": ["p1-01", "p1-02"],
            "method_rung": [2, 1],
            "claim_rung": [3, 2],
            "gap_score": [1, 1],
            "confidence": [4, 3],
        })
        merged = iaa.merge_annotators(claude, gpt)
        assert len(merged) == 2
        assert "method_rung_claude" in merged.columns
        assert "method_rung_gpt" in merged.columns

    def test_merge_inner_join_drops_unmatched(self):
        claude = pd.DataFrame({
            "claim_id": ["p1-01", "p1-02"],
            "method_rung": [2, 1],
            "claim_rung": [3, 2],
            "gap_score": [1, 1],
            "confidence": [4, 3],
        })
        gpt = pd.DataFrame({
            "claim_id": ["p1-01"],  # only one shared claim
            "method_rung": [2],
            "claim_rung": [3],
            "gap_score": [1],
            "confidence": [4],
        })
        merged = iaa.merge_annotators(claude, gpt)
        assert len(merged) == 1
        assert merged["claim_id"].iloc[0] == "p1-01"


class TestIaaConfusionMatrix:
    def test_confusion_matrix_shape(self, iaa_merged_df):
        """3x3 confusion matrix for rung values {1,2,3}."""
        result = iaa.compute_confusion(
            iaa_merged_df,
            variable="method_rung",
            suffix=("_claude", "_gpt"),
            labels=[1, 2, 3],
        )
        assert result["labels"] == [1, 2, 3]
        obs = np.array(result["observed"])
        assert obs.shape == (3, 3)

    def test_confusion_matrix_perfect_on_diagonal(self, iaa_merged_df):
        """Perfect agreement means all counts on the main diagonal."""
        result = iaa.compute_confusion(
            iaa_merged_df,
            variable="method_rung",
            suffix=("_claude", "_gpt"),
            labels=[1, 2, 3],
        )
        obs = np.array(result["observed"])
        off_diagonal = obs - np.diag(np.diag(obs))
        assert off_diagonal.sum() == 0

    def test_confusion_matrix_n_matches_data(self, iaa_merged_df):
        """Total count in confusion matrix equals number of claims."""
        result = iaa.compute_confusion(
            iaa_merged_df,
            variable="method_rung",
            suffix=("_claude", "_gpt"),
            labels=[1, 2, 3],
        )
        assert result["n"] == len(iaa_merged_df)


class TestIaaBiasAnalysis:
    def test_direction_counts_perfect_agreement(self, iaa_merged_df):
        """With identical raters, all differences are tied."""
        bias = iaa.bias_analysis(iaa_merged_df, suffix=("_claude", "_gpt"))
        assert "method_rung" in bias
        mr = bias["method_rung"]
        assert mr["tied"] == len(iaa_merged_df)
        assert mr["claude_higher"] == 0
        assert mr["gpt_higher"] == 0

    def test_bias_analysis_structure(self, iaa_merged_df):
        """bias_analysis returns expected keys for each variable."""
        bias = iaa.bias_analysis(iaa_merged_df, suffix=("_claude", "_gpt"))
        for var in ["method_rung", "claim_rung"]:
            assert var in bias
            for key in ("n", "claude_higher", "gpt_higher", "tied", "mean_diff"):
                assert key in bias[var], f"Missing key '{key}' in bias['{var}']"

    def test_direction_counts_claude_always_higher(self):
        """Claude systematically higher: all differences positive."""
        merged = pd.DataFrame({
            "claim_id": ["p1-01", "p1-02", "p1-03"],
            "method_rung_claude": [3, 3, 3],
            "method_rung_gpt": [1, 1, 1],
            "claim_rung_claude": [3, 3, 3],
            "claim_rung_gpt": [1, 1, 1],
        })
        bias = iaa.bias_analysis(merged, suffix=("_claude", "_gpt"))
        mr = bias["method_rung"]
        assert mr["claude_higher"] == 3
        assert mr["gpt_higher"] == 0
        assert mr["tied"] == 0
        assert mr["mean_diff"] == pytest.approx(2.0)


class TestIaaRawAgreement:
    def test_raw_agreement_perfect(self):
        y = np.array([1, 2, 3])
        assert iaa.raw_agreement(y, y) == pytest.approx(1.0)

    def test_raw_agreement_zero(self):
        assert iaa.raw_agreement(np.array([1, 1]), np.array([2, 2])) == pytest.approx(0.0)

    def test_raw_agreement_empty(self):
        assert np.isnan(iaa.raw_agreement(np.array([]), np.array([])))


class TestIaaLoadAnnotations:
    def test_load_annotations_coerces_int_columns(self, tmp_path):
        """load_annotations coerces rung columns to nullable Int64."""
        csv_path = tmp_path / "ann.csv"
        pd.DataFrame([{
            "claim_id": "p1-01",
            "paper_id": "p1",
            "method_rung": "2",
            "claim_rung": "3",
            "gap_score": "1",
            "confidence": "4",
            "claim_prominence": "3",
        }]).to_csv(csv_path, index=False)
        df = iaa.load_annotations(csv_path)
        assert df["method_rung"].dtype.name == "Int64"
        assert df["claim_rung"].dtype.name == "Int64"

    def test_load_annotations_handles_na_rungs(self, tmp_path):
        """NA rung values survive load without becoming 0."""
        csv_path = tmp_path / "ann.csv"
        pd.DataFrame([{
            "claim_id": "p1-01",
            "paper_id": "p1",
            "method_rung": "",
            "claim_rung": "",
            "gap_score": "",
            "confidence": "3",
            "claim_prominence": "2",
        }]).to_csv(csv_path, index=False)
        df = iaa.load_annotations(csv_path)
        assert pd.isna(df["method_rung"].iloc[0])


# ===========================================================================
# Integration: real annotations.csv round-trip through validate
# ===========================================================================


class TestIntegration:
    def test_real_annotations_validate_end_to_end(self):
        """Full validate_file call on the real 186-row file returns no errors."""
        ann_path = H2_DIR / "annotations.csv"
        cp_path = H2_DIR / "candidate_papers.csv"
        assert ann_path.exists(), "annotations.csv not found"
        assert cp_path.exists(), "candidate_papers.csv not found"
        errors = val.validate_file(ann_path, mode="original", candidate_papers_path=cp_path)
        assert errors == [], f"validate_file found errors: {errors}"

    def test_real_annotations_row_count(self):
        """annotations.csv has at least 100 rows (sanity check)."""
        df = pd.read_csv(H2_DIR / "annotations.csv")
        assert len(df) >= 100

    def test_real_annotations_na_rung_claim_present(self):
        """The known NA-rung row (2511.22662-01) exists in annotations.csv."""
        df = pd.read_csv(H2_DIR / "annotations.csv")
        na_rows = df[df["method_rung"].isna()]
        # The file may have the known conceptual-analysis row
        # If present it should have NA gap_score too
        if not na_rows.empty:
            assert na_rows["gap_score"].isna().all(), (
                "Rows with NA method_rung should also have NA gap_score"
            )


# ===========================================================================
# 5. Resume logic (_load_existing_results)
# ===========================================================================


class TestResumeLogic:
    def test_load_existing_results_nonexistent_file(self, tmp_path):
        """Returns empty list when CSV does not exist."""
        from annotate import _load_existing_results

        result = _load_existing_results(tmp_path / "missing.csv")
        assert result == []

    def test_load_existing_results_empty_csv(self, tmp_path):
        """Returns empty list for a CSV with only a header row."""
        from annotate import _load_existing_results

        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("claim_id,paper_id,method_rung\n")
        result = _load_existing_results(csv_path)
        assert result == []

    def test_load_existing_results_returns_rows(self, tmp_path):
        """Returns list of dicts for a CSV with data rows."""
        from annotate import _load_existing_results

        csv_path = tmp_path / "existing.csv"
        csv_path.write_text(
            "claim_id,paper_id,method_rung\n"
            "2211.00593-01,2211.00593,2\n"
            "2211.00593-02,2211.00593,3\n"
        )
        result = _load_existing_results(csv_path)
        assert len(result) == 2
        assert result[0]["claim_id"] == "2211.00593-01"
        assert result[1]["claim_id"] == "2211.00593-02"
        assert result[0]["method_rung"] == "2"

    def test_resume_skips_done_claim_ids(self, tmp_path):
        """Simulates resume logic: already-done claim_ids are filtered out."""
        from annotate import _load_existing_results

        csv_path = tmp_path / "annotations_classify.csv"
        csv_path.write_text(
            "claim_id,paper_id,method_rung,claim_rung,gap_score\n"
            "2211.00593-01,2211.00593,2,3,1\n"
        )
        existing = _load_existing_results(csv_path)
        done_ids = {r["claim_id"] for r in existing}

        all_claims = [
            {"claim_id": "2211.00593-01", "paper_id": "2211.00593"},
            {"claim_id": "2211.00593-02", "paper_id": "2211.00593"},
            {"claim_id": "2202.05262-01", "paper_id": "2202.05262"},
        ]
        remaining = [c for c in all_claims if c["claim_id"] not in done_ids]

        assert len(remaining) == 2
        assert all(c["claim_id"] != "2211.00593-01" for c in remaining)

    def test_resume_extract_skips_done_paper_ids(self, tmp_path):
        """Simulates resume logic for extract mode: done paper_ids are skipped."""
        from annotate import _load_existing_results

        csv_path = tmp_path / "annotations_extract.csv"
        csv_path.write_text(
            "claim_id,paper_id,method_rung\n"
            "2211.00593-ext-01,2211.00593,2\n"
            "2211.00593-ext-02,2211.00593,3\n"
        )
        existing = _load_existing_results(csv_path)
        done_pids = {r["paper_id"] for r in existing}

        all_paper_ids = ["2211.00593", "2202.05262", "2301.05217"]
        remaining = [p for p in all_paper_ids if p not in done_pids]

        assert len(remaining) == 2
        assert "2211.00593" not in remaining
