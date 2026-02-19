"""Validate annotation CSV files for the pilot study pipeline.

Checks schema compliance, value ranges, gap_score consistency,
paper_id existence, claim_id uniqueness, and non-empty claim_text.

Usage:
    python validate.py annotations.csv
    python validate.py annotations_multi/claude/annotations_classify.csv --mode classify
    python validate.py --all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

COLUMNS_ORIGINAL = [
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

COLUMNS_CLASSIFY = [
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
    "hedge_flag",
    "reasoning",
]

VALID_LOCATIONS = {
    "abstract",
    "introduction",
    "body",
    "methods",
    "results",
    "discussion",
    "conclusion",
}

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_schema(df: pd.DataFrame, mode: str) -> list[str]:
    """Check that all required columns are present."""
    expected = COLUMNS_CLASSIFY if mode == "classify" else COLUMNS_ORIGINAL
    missing = [c for c in expected if c not in df.columns]
    if missing:
        return [f"Missing columns: {missing}"]
    return []


def _check_gap_score(df: pd.DataFrame) -> list[str]:
    """Verify gap_score == max(0, claim_rung - method_rung) for every row.

    Rows where method_rung or claim_rung is NA are skipped (logged as warnings).
    This handles known cases like conceptual-analysis papers.
    """
    errors: list[str] = []
    na_mask = df["method_rung"].isna() | df["claim_rung"].isna()
    n_na = na_mask.sum()
    if n_na > 0:
        na_ids = df.loc[na_mask, "claim_id"].tolist()
        logger.warning(
            "Skipping gap_score check for %d row(s) with NA rung values: %s",
            n_na, na_ids,
        )
    valid = df[~na_mask].copy()
    if valid.empty:
        return errors
    expected = (valid["claim_rung"] - valid["method_rung"]).clip(lower=0)
    mask = valid["gap_score"] != expected
    if mask.any():
        bad = valid.loc[mask, ["claim_id", "method_rung", "claim_rung", "gap_score"]]
        for _, row in bad.iterrows():
            errors.append(
                f"gap_score mismatch for {row['claim_id']}: "
                f"got {row['gap_score']}, expected max(0, {row['claim_rung']} - {row['method_rung']}) "
                f"= {max(0, row['claim_rung'] - row['method_rung'])}"
            )
    return errors


def _check_value_ranges(df: pd.DataFrame, mode: str) -> list[str]:
    """Validate that numeric and categorical columns fall within allowed ranges."""
    errors: list[str] = []

    range_checks: list[tuple[str, set, bool]] = [
        ("method_rung", {1, 2, 3}, True),      # allow NA
        ("claim_rung", {1, 2, 3}, True),        # allow NA
        ("gap_score", {0, 1, 2}, True),         # allow NA
        ("confidence", {1, 2, 3, 4, 5}, False),
        ("claim_prominence", {1, 2, 3}, False),
    ]
    if mode == "classify":
        range_checks.append(("hedge_flag", {0, 1}, False))

    for col, valid_vals, allow_na in range_checks:
        if col not in df.columns:
            continue
        check_df = df.dropna(subset=[col]) if allow_na else df
        invalid = check_df[~check_df[col].isin(valid_vals)]
        if not invalid.empty:
            bad_vals = sorted(invalid[col].unique())
            errors.append(
                f"Invalid {col} values {bad_vals} in rows: "
                f"{invalid.index.tolist()}"
            )

    # claim_location is categorical text
    if "claim_location" in df.columns:
        locs = df["claim_location"].str.strip().str.lower()
        invalid_locs = locs[~locs.isin(VALID_LOCATIONS)]
        if not invalid_locs.empty:
            errors.append(
                f"Invalid claim_location values: "
                f"{sorted(invalid_locs.unique().tolist())}"
            )

    return errors


def _check_paper_ids(df: pd.DataFrame, candidate_papers_path: Path) -> list[str]:
    """Every paper_id must exist in candidate_papers.csv."""
    errors: list[str] = []
    if not candidate_papers_path.exists():
        return [f"candidate_papers file not found: {candidate_papers_path}"]

    cp = pd.read_csv(candidate_papers_path)
    if "paper_id" not in cp.columns:
        return ["candidate_papers.csv missing 'paper_id' column"]

    # Coerce both to string for comparison
    valid_ids = set(cp["paper_id"].astype(str))
    annotation_ids = set(df["paper_id"].astype(str))
    unknown = annotation_ids - valid_ids
    if unknown:
        errors.append(f"paper_ids not in candidate_papers.csv: {sorted(unknown)}")
    return errors


def _check_claim_id_uniqueness(df: pd.DataFrame) -> list[str]:
    """No duplicate claim_ids."""
    dupes = df["claim_id"][df["claim_id"].duplicated(keep=False)]
    if not dupes.empty:
        return [f"Duplicate claim_ids: {sorted(dupes.unique().tolist())}"]
    return []


def _check_nonempty_claim_text(df: pd.DataFrame) -> list[str]:
    """Every row must have non-empty claim_text."""
    empty = df[df["claim_text"].astype(str).str.strip().eq("") | df["claim_text"].isna()]
    if not empty.empty:
        return [
            f"Empty claim_text in {len(empty)} row(s): index {empty.index.tolist()}"
        ]
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

CHECK_NAMES = [
    "schema",
    "gap_score",
    "value_ranges",
    "paper_ids",
    "claim_id_uniqueness",
    "nonempty_claim_text",
]


def validate_file(
    path: Path,
    mode: str = "original",
    candidate_papers_path: Path | None = None,
) -> list[str]:
    """Validate a single annotation CSV. Returns a list of error strings (empty = all good)."""
    if not path.exists():
        return [f"File not found: {path}"]

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return [f"Failed to read CSV: {exc}"]

    if df.empty:
        return ["CSV file is empty (no data rows)"]

    if candidate_papers_path is None:
        candidate_papers_path = path.parent / "candidate_papers.csv"
        # For files nested in subdirs, walk up to find candidate_papers.csv
        if not candidate_papers_path.exists():
            for parent in path.parents:
                candidate = parent / "candidate_papers.csv"
                if candidate.exists():
                    candidate_papers_path = candidate
                    break

    errors: list[str] = []
    errors.extend(_check_schema(df, mode))
    # Only run remaining checks if schema is OK (columns exist)
    if not errors:
        errors.extend(_check_gap_score(df))
        errors.extend(_check_value_ranges(df, mode))
        errors.extend(_check_paper_ids(df, candidate_papers_path))
        errors.extend(_check_claim_id_uniqueness(df))
        errors.extend(_check_nonempty_claim_text(df))

    return errors


def _find_all_csvs(root: Path) -> list[tuple[Path, str]]:
    """Find all annotation CSVs under *root* and infer their mode."""
    results: list[tuple[Path, str]] = []
    for csv_path in sorted(root.rglob("*.csv")):
        name = csv_path.name.lower()
        # Skip non-annotation files
        if name in {"candidate_papers.csv", "annotation_template.csv", "calibration_annotations.csv"}:
            continue
        if "annotation" not in name:
            continue
        mode = "classify" if "classify" in name else "original"
        results.append((csv_path, mode))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate annotation CSV files for the pilot study pipeline."
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        type=Path,
        help="Path to annotation CSV file",
    )
    parser.add_argument(
        "--mode",
        choices=["original", "classify"],
        default="original",
        help="Annotation mode: 'original' or 'classify' (default: original)",
    )
    parser.add_argument(
        "--candidate-papers",
        type=Path,
        default=None,
        help="Path to candidate_papers.csv (default: auto-detect)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="validate_all",
        help="Validate all annotation CSVs found in the directory tree",
    )

    args = parser.parse_args(argv)

    if not args.csv_file and not args.validate_all:
        parser.error("Provide a CSV file path or use --all")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Collect files to validate
    files: list[tuple[Path, str]] = []
    if args.validate_all:
        root = Path.cwd()
        files = _find_all_csvs(root)
        if not files:
            logger.info("[WARN] No annotation CSVs found under %s", root)
            return 0
    else:
        files = [(args.csv_file, args.mode)]

    total_pass = 0
    total_fail = 0
    all_ok = True

    for csv_path, mode in files:
        logger.info("=" * 60)
        logger.info("Validating: %s  (mode=%s)", csv_path, mode)
        logger.info("-" * 60)

        errors = validate_file(csv_path, mode=mode, candidate_papers_path=args.candidate_papers)

        if errors:
            for err in errors:
                logger.info("[FAIL] %s", err)
            total_fail += len(errors)
            all_ok = False
        else:
            logger.info("[PASS] All checks passed")

        # Report per-check pass/fail for the structured checks
        n_checks = len(CHECK_NAMES)
        n_failed = len(errors)
        n_passed = max(0, n_checks - n_failed)
        total_pass += n_passed

        logger.info("")
        logger.info(
            "Summary for %s: %d/%d checks passed",
            csv_path.name,
            n_passed,
            n_checks,
        )
        logger.info("")

    if len(files) > 1:
        logger.info("=" * 60)
        logger.info(
            "Overall: %d file(s) validated, %s",
            len(files),
            "ALL PASSED" if all_ok else "SOME FAILURES",
        )

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
