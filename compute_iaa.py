#!/usr/bin/env python3
"""
Multi-Model Inter-Annotator Agreement (IAA) for LLM Pilot Study

Computes agreement metrics across all LLM annotators following
Option D: 7 primary models (186 claims) + DeepSeek R1 supplementary (166 claims).

Metrics:
    - Krippendorff's alpha (ordinal/nominal, handles missing data)
    - Fleiss' kappa (on complete-case intersection)
    - Light's kappa (mean pairwise Cohen's kappa)
    - Pairwise Cohen's kappa matrices
    - Paper-level cluster bootstrap CIs

Outputs:
    - LaTeX tables (booktabs) for paper appendix
    - JSON with all metrics
    - Pairwise agreement heatmaps (PDF)

Usage:
    python compute_iaa.py
    python compute_iaa.py --n-bootstrap 1000          # fast run
    python compute_iaa.py --primary-only               # skip supplementary R1
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

try:
    import krippendorff

    HAS_KRIPPENDORFF = True
except ImportError:
    HAS_KRIPPENDORFF = False
    warnings.warn("krippendorff not installed; alpha metrics will be skipped.", stacklevel=2)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CALIBRATION_IDS = {"2211.00593", "2202.05262", "2301.05217", "2409.04478", "2601.11516"}

DISPLAY_NAMES: dict[str, str] = {
    "claude": "Claude",
    "gpt5": "GPT-5",
    "sonnet": "Sonnet",
    "openrouter-deepseek-r1": "DS-R1",
    "openrouter-deepseek-v3": "DS-V3",
    "openrouter-gemini": "Gemini",
    "openrouter-mistral": "Mistral",
    "openrouter-qwen": "Qwen",
}

SUPPLEMENTARY_ANNOTATORS = {"openrouter-deepseek-r1"}

METHOD_TO_TYPE: dict[str, str] = {
    "Activation Patching": "Circuit discovery",
    "Circuit Analysis": "Circuit discovery",
    "Circuit Analysis + Ablation": "Circuit discovery",
    "Causal Tracing": "Knowledge localization",
    "Causal Tracing + ROME": "Knowledge localization",
    "ROME Editing": "Knowledge localization",
    "SAE Attribution": "Evaluation/benchmark",
    "Interchange Intervention": "Evaluation/benchmark",
    "Intervention Benchmark": "Evaluation/benchmark",
    "Linear Probing": "Applied/production",
    "Steering Vectors": "Applied/production",
    "Steering + Probing": "Applied/production",
    "Probing": "Applied/production",
    "Weight Analysis": "Other",
    "DAS/Interchange Intervention": "Evaluation/benchmark",
    "Causal Mediation": "Circuit discovery",
    "Causal Analysis": "Circuit discovery",
}

# Variables and their measurement properties
VAR_CONFIG: dict[str, dict] = {
    "method_rung": {"categories": [1, 2, 3], "level": "ordinal", "weights": "linear"},
    "claim_rung": {"categories": [1, 2, 3], "level": "ordinal", "weights": "linear"},
    "gap_score": {"categories": [0, 1, 2], "level": "ordinal", "weights": "linear"},
    "gap_binary": {"categories": [0, 1], "level": "nominal", "weights": None},
    "confidence": {"categories": [1, 2, 3, 4, 5], "level": "ordinal", "weights": "linear"},
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_annotations(path: Path) -> pd.DataFrame:
    """Load annotations CSV, coercing numeric columns."""
    df = pd.read_csv(path, dtype=str)
    int_cols = ["method_rung", "claim_rung", "gap_score", "confidence", "claim_prominence"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["paper_id", "claim_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    # Derive gap_binary
    if "gap_score" in df.columns:
        df["gap_binary"] = (df["gap_score"] > 0).astype("Int64")
    return df


def discover_annotators(base_dir: Path, min_claims: int = 10) -> dict[str, pd.DataFrame]:
    """Auto-discover annotator directories with valid CSVs."""
    annotators: dict[str, pd.DataFrame] = {}
    for d in sorted(base_dir.iterdir()):
        if not d.is_dir():
            continue
        csv_path = d / "annotations_classify.csv"
        if not csv_path.exists():
            continue
        df = load_annotations(csv_path)
        if len(df) < min_claims:
            log.info("Skipping %s: only %d claims", d.name, len(df))
            continue
        annotators[d.name] = df
        log.info(
            "Loaded %-25s %d claims, %d papers",
            d.name,
            len(df),
            df["paper_id"].nunique(),
        )
    return annotators


def split_annotators(
    annotators: dict[str, pd.DataFrame],
) -> tuple[list[str], list[str]]:
    """Classify annotators into primary and supplementary (Option D)."""
    primary, supplementary = [], []
    for name in annotators:
        if name in SUPPLEMENTARY_ANNOTATORS:
            supplementary.append(name)
        else:
            primary.append(name)
    return sorted(primary), sorted(supplementary)


# ---------------------------------------------------------------------------
# Wide-format construction
# ---------------------------------------------------------------------------


def build_wide(
    annotators: dict[str, pd.DataFrame],
    names: list[str],
    variable: str,
) -> pd.DataFrame:
    """Pivot to wide format: (claim_id index, annotator columns).

    All claim_ids across selected annotators are included; missing values are NaN.
    """
    frames = []
    for name in names:
        df = annotators[name]
        if variable not in df.columns:
            continue
        s = df.set_index("claim_id")[variable].rename(name)
        s = pd.to_numeric(s, errors="coerce")
        frames.append(s)
    if not frames:
        return pd.DataFrame()
    wide = pd.concat(frames, axis=1)
    return wide.reindex(columns=names)


def claim_to_paper_map(annotators: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Build claim_id -> paper_id mapping from all annotators."""
    mapping: dict[str, str] = {}
    for df in annotators.values():
        for _, row in df.iterrows():
            mapping[row["claim_id"]] = row["paper_id"]
    return mapping


# ---------------------------------------------------------------------------
# Multi-rater agreement metrics
# ---------------------------------------------------------------------------


def fleiss_kappa(matrix: np.ndarray) -> float:
    """Fleiss' kappa for multi-rater nominal agreement.

    Args:
        matrix: (n_items, n_categories), each cell = count of raters for that category.
    """
    n_items, n_cats = matrix.shape
    raters_per_item = matrix.sum(axis=1)
    n_raters = int(raters_per_item.max())

    # Only keep items with the expected number of raters
    valid = raters_per_item == n_raters
    matrix = matrix[valid]
    n_items = len(matrix)
    if n_items < 2 or n_raters < 2:
        return np.nan

    p_i = ((matrix**2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = float(p_i.mean())

    p_j = matrix.sum(axis=0) / (n_items * n_raters)
    P_e = float((p_j**2).sum())

    if np.isclose(P_e, 1.0):
        return 1.0 if np.isclose(P_bar, 1.0) else np.nan
    return (P_bar - P_e) / (1.0 - P_e)


def compute_kripp_alpha(wide: pd.DataFrame, level: str = "ordinal") -> float:
    """Krippendorff's alpha from wide-format DataFrame."""
    if not HAS_KRIPPENDORFF:
        return np.nan
    matrix = wide.to_numpy(dtype=float, na_value=np.nan).T  # (n_raters, n_items)
    try:
        return float(krippendorff.alpha(reliability_data=matrix, level_of_measurement=level))
    except Exception:
        return np.nan


def compute_fleiss_from_wide(wide: pd.DataFrame, categories: list) -> tuple[float, int]:
    """Fleiss' kappa from wide-format on complete cases only."""
    complete = wide.dropna()
    n_items = len(complete)
    if n_items < 4:
        return np.nan, n_items

    cat_to_idx = {c: i for i, c in enumerate(categories)}
    matrix = np.zeros((n_items, len(categories)), dtype=int)

    for item_idx, (_, row) in enumerate(complete.iterrows()):
        for val in row.values:
            v = int(val)
            if v in cat_to_idx:
                matrix[item_idx, cat_to_idx[v]] += 1

    return fleiss_kappa(matrix), n_items


def compute_lights_kappa(wide: pd.DataFrame, weights: str | None = "linear") -> dict:
    """Light's kappa: mean of all pairwise Cohen's kappas."""
    names = wide.columns.tolist()
    kappas = []
    for a, b in combinations(names, 2):
        mask = wide[a].notna() & wide[b].notna()
        y1 = wide.loc[mask, a].values
        y2 = wide.loc[mask, b].values
        if len(y1) < 4:
            continue
        try:
            k = cohen_kappa_score(y1, y2, weights=weights)
            kappas.append(k)
        except Exception:
            continue

    if not kappas:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "n_pairs": 0}

    return {
        "mean": float(np.mean(kappas)),
        "std": float(np.std(kappas)),
        "min": float(np.min(kappas)),
        "max": float(np.max(kappas)),
        "n_pairs": len(kappas),
    }


def compute_pairwise_matrix(wide: pd.DataFrame, weights: str | None = "linear") -> np.ndarray:
    """NxN pairwise Cohen's kappa matrix."""
    names = wide.columns.tolist()
    n = len(names)
    matrix = np.full((n, n), np.nan)
    np.fill_diagonal(matrix, 1.0)

    for i in range(n):
        for j in range(i + 1, n):
            mask = wide.iloc[:, i].notna() & wide.iloc[:, j].notna()
            y1 = wide.iloc[:, i][mask].values
            y2 = wide.iloc[:, j][mask].values
            if len(y1) < 4:
                continue
            try:
                k = cohen_kappa_score(y1, y2, weights=weights)
                matrix[i, j] = matrix[j, i] = k
            except Exception:
                continue

    return matrix


def unanimous_agreement(wide: pd.DataFrame) -> tuple[float, int, int]:
    """Fraction of complete-case items where ALL raters agree exactly.

    Returns (rate, n_agree, n_complete).
    """
    complete = wide.dropna()
    if len(complete) == 0:
        return np.nan, 0, 0
    n_agree = int((complete.nunique(axis=1) == 1).sum())
    return float(n_agree / len(complete)), n_agree, len(complete)


# ---------------------------------------------------------------------------
# Full multi-rater analysis
# ---------------------------------------------------------------------------


def compute_multi_rater_analysis(
    annotators: dict[str, pd.DataFrame],
    names: list[str],
) -> dict:
    """Compute all multi-rater agreement metrics for each variable."""
    results: dict = {
        "_meta": {
            "n_raters": len(names),
            "raters": names,
            "rater_claims": {n: len(annotators[n]) for n in names},
        }
    }

    for var, cfg in VAR_CONFIG.items():
        log.info("  analyzing %s ...", var)
        wide = build_wide(annotators, names, var)
        if wide.empty:
            continue

        vr: dict = {
            "n_claims_total": len(wide),
            "n_claims_complete": int(wide.dropna().shape[0]),
        }

        # Krippendorff's alpha (handles missing data natively)
        vr["krippendorff_alpha"] = compute_kripp_alpha(wide, cfg["level"])

        # Fleiss' kappa (complete cases only)
        fk, n_fleiss = compute_fleiss_from_wide(wide, cfg["categories"])
        vr["fleiss_kappa"] = fk
        vr["fleiss_n"] = n_fleiss

        # Light's kappa (mean pairwise)
        vr["lights_kappa"] = compute_lights_kappa(wide, weights=cfg["weights"])

        # Pairwise matrix
        pw = compute_pairwise_matrix(wide, weights=cfg["weights"])
        vr["pairwise_matrix"] = pw.tolist()
        vr["pairwise_names"] = wide.columns.tolist()

        # Unanimous agreement
        unan_rate, unan_n, unan_total = unanimous_agreement(wide)
        vr["unanimous_pct"] = unan_rate
        vr["unanimous_n"] = unan_n
        vr["unanimous_total"] = unan_total

        results[var] = vr

    return results


# ---------------------------------------------------------------------------
# Paper-level cluster bootstrap
# ---------------------------------------------------------------------------


def paper_cluster_bootstrap(
    annotators: dict[str, pd.DataFrame],
    names: list[str],
    variable: str,
    level: str = "ordinal",
    n_bootstrap: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Paper-level cluster bootstrap CI for Krippendorff's alpha."""
    if not HAS_KRIPPENDORFF:
        return {"ci_lower": np.nan, "ci_upper": np.nan, "se": np.nan}

    wide = build_wide(annotators, names, variable)
    c2p = claim_to_paper_map(annotators)
    wide["paper_id"] = wide.index.map(c2p)

    papers = wide["paper_id"].dropna().unique()
    n_papers = len(papers)
    if n_papers < 3:
        return {"ci_lower": np.nan, "ci_upper": np.nan, "se": np.nan}

    paper_index = {pid: wide[wide["paper_id"] == pid].drop(columns="paper_id") for pid in papers}

    rng = np.random.default_rng(seed)
    boot_vals = np.full(n_bootstrap, np.nan)
    report_every = max(1, n_bootstrap // 20)

    for b in range(n_bootstrap):
        if (b + 1) % report_every == 0:
            pct = (b + 1) / n_bootstrap * 100
            print(f"  bootstrap {variable}: {b + 1}/{n_bootstrap} ({pct:.0f}%)", end="\r", flush=True)

        sampled = rng.choice(papers, size=n_papers, replace=True)
        boot_wide = pd.concat([paper_index[p] for p in sampled], ignore_index=True)

        matrix = boot_wide.to_numpy(dtype=float, na_value=np.nan).T  # (n_raters, n_items)
        try:
            boot_vals[b] = krippendorff.alpha(
                reliability_data=matrix, level_of_measurement=level
            )
        except Exception:
            continue

    print()
    valid = boot_vals[~np.isnan(boot_vals)]
    if len(valid) < 100:
        log.warning("Only %d valid bootstrap samples for %s", len(valid), variable)

    lo = float(np.percentile(valid, 100 * alpha / 2)) if len(valid) > 0 else np.nan
    hi = float(np.percentile(valid, 100 * (1 - alpha / 2))) if len(valid) > 0 else np.nan
    se = float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan

    return {"ci_lower": lo, "ci_upper": hi, "se": se, "n_valid": int(len(valid))}


# ---------------------------------------------------------------------------
# Per-model bias
# ---------------------------------------------------------------------------


def per_model_bias(annotators: dict[str, pd.DataFrame], names: list[str]) -> dict:
    """Per-model annotation tendencies."""
    results: dict = {}
    for name in names:
        df = annotators[name]
        results[name] = {
            "n_claims": len(df),
            "mean_method_rung": float(df["method_rung"].mean()) if "method_rung" in df.columns else np.nan,
            "mean_claim_rung": float(df["claim_rung"].mean()) if "claim_rung" in df.columns else np.nan,
            "mean_gap_score": float(df["gap_score"].mean()) if "gap_score" in df.columns else np.nan,
            "overclaim_rate": float((df["gap_score"] > 0).mean()) if "gap_score" in df.columns else np.nan,
            "strong_overclaim_rate": float((df["gap_score"] > 1).mean()) if "gap_score" in df.columns else np.nan,
            "mean_confidence": float(df["confidence"].mean()) if "confidence" in df.columns else np.nan,
        }
    return results


# ---------------------------------------------------------------------------
# Stratified analysis
# ---------------------------------------------------------------------------


def stratified_alpha(
    annotators: dict[str, pd.DataFrame],
    names: list[str],
    variable: str,
    group_fn,
    level: str = "ordinal",
) -> dict:
    """Krippendorff's alpha stratified by a grouping function on paper_id."""
    # Build long then group
    frames = []
    for name in names:
        df = annotators[name][["claim_id", "paper_id", variable]].copy()
        df["annotator"] = name
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    long["group"] = long["paper_id"].map(group_fn).fillna("Other")

    results: dict = {}
    for group, sub in long.groupby("group"):
        wide = sub.pivot(index="claim_id", columns="annotator", values=variable)
        wide = wide.reindex(columns=names)
        if len(wide) < 4:
            continue
        a = compute_kripp_alpha(wide, level)
        results[str(group)] = {"n": len(wide), "krippendorff_alpha": a}

    return results


# ---------------------------------------------------------------------------
# LaTeX output helpers
# ---------------------------------------------------------------------------


def _fmt(val, decimals: int = 2) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "--"
    return f"{val:.{decimals}f}"


def _fmt_ci(lo, hi) -> str:
    if isinstance(lo, float) and np.isnan(lo):
        return "--"
    if isinstance(hi, float) and np.isnan(hi):
        return "--"
    return f"[{lo:.2f}, {hi:.2f}]"


def _dname(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def _esc(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&")


# ---------------------------------------------------------------------------
# LaTeX table generators
# ---------------------------------------------------------------------------


def generate_summary_tex(analysis: dict, cis: dict, n_raters: int) -> str:
    """Multi-rater IAA summary table (replaces old iaa_summary.tex)."""
    rows = []
    specs = [
        ("method_rung", r"\texttt{method\_rung}", r"$\alpha_{ord}$", r"Light's $\kappa_w$"),
        ("claim_rung", r"\texttt{claim\_rung}", r"$\alpha_{ord}$", r"Light's $\kappa_w$"),
        ("gap_score", r"\texttt{gap\_score}", r"$\alpha_{ord}$", r"Light's $\kappa_w$"),
        ("gap_binary", r"\texttt{gap\_binary}", r"Fleiss' $\kappa$", r"$\alpha_{nom}$"),
        ("confidence", r"\texttt{confidence}", r"$\alpha_{ord}$", r"Light's $\kappa_w$"),
    ]

    for var, label, prim_label, sec_label in specs:
        m = analysis.get(var, {})
        ci = cis.get(var, {})

        if var == "gap_binary":
            prim = _fmt(m.get("fleiss_kappa"))
            sec = _fmt(m.get("krippendorff_alpha"))
        else:
            prim = _fmt(m.get("krippendorff_alpha"))
            lk = m.get("lights_kappa", {})
            sec = _fmt(lk.get("mean") if isinstance(lk, dict) else lk)

        ci_str = _fmt_ci(ci.get("ci_lower", np.nan), ci.get("ci_upper", np.nan))
        unan = m.get("unanimous_pct")
        unan_str = f"{unan * 100:.1f}\\%" if unan is not None and not np.isnan(unan) else "--"
        n = m.get("n_claims_complete", "?")

        rows.append(
            f"    {label} & {n} & {prim_label} = {prim} & {ci_str} & {sec_label} = {sec} & {unan_str} \\\\"
        )

    return rf"""\begin{{table}}[t]
\centering
\caption{{Multi-rater agreement across {n_raters} LLM annotators.
Primary metric: Krippendorff's $\alpha$ (ordinal) for rung variables,
Fleiss' $\kappa$ for binary overclaim.
Bootstrap CIs use paper-level cluster resampling ($n = 10\,000$).}}
\label{{tab:iaa-summary}}
\begin{{tabular}}{{llcccc}}
\toprule
Variable & $N$ & Primary & 95\% CI & Secondary & Unanimous \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def generate_pairwise_tex(matrix: np.ndarray, names: list[str], variable: str, caption: str) -> str:
    """Lower-triangular pairwise kappa table."""
    n = len(names)
    display = [_dname(name) for name in names]

    header = " & ".join(rf"\textbf{{{d}}}" for d in display[:-1])

    rows = []
    for i in range(1, n):
        cells = [_fmt(matrix[i, j]) for j in range(i)]
        while len(cells) < n - 1:
            cells.append("")
        rows.append(rf"    \textbf{{{display[i]}}} & {' & '.join(cells)} \\")

    col_spec = "l" + "c" * (n - 1)

    return rf"""\begin{{table}}[t]
\centering
\caption{{{caption}}}
\label{{tab:iaa-pairwise-{variable.replace('_', '-')}}}
\small
\begin{{tabular}}{{{col_spec}}}
\toprule
 & {header} \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def generate_bias_tex(bias: dict) -> str:
    """Per-model annotation tendencies table."""
    rows = []
    for name in sorted(bias.keys()):
        b = bias[name]
        dname = _dname(name)
        oc = b.get("overclaim_rate", np.nan)
        oc_str = f"{oc * 100:.1f}" if not np.isnan(oc) else "--"
        rows.append(
            rf"    \textbf{{{dname}}} & {b['n_claims']} & "
            rf"{_fmt(b['mean_method_rung'])} & {_fmt(b['mean_claim_rung'])} & "
            rf"{_fmt(b['mean_gap_score'])} & {oc_str}\% & "
            rf"{_fmt(b['mean_confidence'])} \\"
        )

    return rf"""\begin{{table}}[t]
\centering
\caption{{Per-model annotation tendencies across all rated claims.
$\bar{{R}}_{{m}}$/$\bar{{R}}_{{c}}$: mean method/claim rung;
$\bar{{g}}$: mean gap score;
OC: overclaim rate (gap $> 0$).}}
\label{{tab:iaa-bias}}
\begin{{tabular}}{{lcccccc}}
\toprule
Model & $N$ & $\bar{{R}}_{{m}}$ & $\bar{{R}}_{{c}}$ & $\bar{{g}}$ & OC\,\% & $\bar{{conf}}$ \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def generate_stratified_tex(strat: dict, group_label: str) -> str:
    """Stratified Krippendorff's alpha table."""
    rows = []
    for group in sorted(strat.keys()):
        vals = strat[group]
        rows.append(
            rf"    {_esc(str(group))} & {vals.get('n', '?')} & {_fmt(vals.get('krippendorff_alpha'))} \\"
        )

    return rf"""\begin{{table}}[t]
\centering
\caption{{Krippendorff's $\alpha$ (ordinal) for \texttt{{gap\_score}} stratified by {_esc(group_label)}.}}
\label{{tab:iaa-by-{group_label.lower().replace(' ', '-')}}}
\begin{{tabular}}{{lcc}}
\toprule
{_esc(group_label)} & $N$ & $\alpha_{{ord}}$ \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def generate_supplementary_tex(
    primary_analysis: dict,
    supp_analysis: dict,
    n_primary: int,
    n_all: int,
) -> str:
    """Comparison table: primary vs supplementary (with R1)."""
    rows = []
    for var in ["method_rung", "claim_rung", "gap_score", "gap_binary", "confidence"]:
        pm = primary_analysis.get(var, {})
        sm = supp_analysis.get(var, {})
        label = rf"\texttt{{{_esc(var)}}}"
        pa = _fmt(pm.get("krippendorff_alpha"))
        sa = _fmt(sm.get("krippendorff_alpha"))
        pn = pm.get("n_claims_complete", "?")
        sn = sm.get("n_claims_complete", "?")
        rows.append(rf"    {label} & {pn} & {pa} & {sn} & {sa} \\")

    return rf"""\begin{{table}}[t]
\centering
\caption{{Krippendorff's $\alpha$ (ordinal): {n_primary} primary annotators vs.\
{n_all} annotators (including DeepSeek R1 supplementary on 166 claims).}}
\label{{tab:iaa-supplementary}}
\begin{{tabular}}{{lcccc}}
\toprule
 & \multicolumn{{2}}{{c}}{{\textbf{{Primary ({n_primary})}}}} & \multicolumn{{2}}{{c}}{{\textbf{{+R1 ({n_all})}}}} \\
\cmidrule(lr){{2-3}} \cmidrule(lr){{4-5}}
Variable & $N$ & $\alpha$ & $N$ & $\alpha$ \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}"""


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_pairwise_heatmaps(analysis: dict, output_dir: Path) -> None:
    """Plot pairwise kappa heatmaps for key variables."""
    if not HAS_PLOT:
        log.warning("matplotlib/seaborn not available; skipping plots.")
        return

    plot_vars = ["method_rung", "claim_rung", "gap_score"]
    fig, axes = plt.subplots(1, len(plot_vars), figsize=(6 * len(plot_vars), 5))
    if len(plot_vars) == 1:
        axes = [axes]

    for ax, var in zip(axes, plot_vars):
        m = analysis.get(var, {})
        pw = m.get("pairwise_matrix")
        pw_names = m.get("pairwise_names", [])
        if pw is None:
            continue

        matrix = np.array(pw)
        display = [_dname(n) for n in pw_names]

        mask = np.zeros_like(matrix, dtype=bool)
        np.fill_diagonal(mask, True)

        sns.heatmap(
            matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            xticklabels=display,
            yticklabels=display,
            vmin=0,
            vmax=1,
            ax=ax,
            square=True,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(var.replace("_", " "), fontsize=12)

    plt.tight_layout()
    out = output_dir / "iaa_pairwise_heatmap.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out)


def plot_model_bias(bias: dict, output_path: Path) -> None:
    """Bar chart of per-model annotation tendencies."""
    if not HAS_PLOT:
        return

    names = sorted(bias.keys())
    display = [_dname(n) for n in names]
    colors = sns.color_palette("Set2", len(names))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, metric, label in [
        (axes[0], "mean_method_rung", "Mean method rung"),
        (axes[1], "mean_claim_rung", "Mean claim rung"),
        (axes[2], "overclaim_rate", "Overclaim rate"),
    ]:
        vals = [bias[n].get(metric, np.nan) for n in names]
        ax.barh(display, vals, color=colors)
        ax.set_xlabel(label)
        grand_mean = np.nanmean(vals)
        ax.axvline(grand_mean, color="red", linestyle="--", alpha=0.7, label=f"mean={grand_mean:.2f}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", output_path)


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def sanitize(obj):
    """Recursively convert numpy types to Python-native for JSON."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-model IAA for LLM pilot study.")
    script_dir = Path(__file__).parent

    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=script_dir / "annotations_multi",
    )
    parser.add_argument(
        "--candidate-papers",
        type=Path,
        default=script_dir / "candidate_papers.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=script_dir / "output")
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--primary-only", action="store_true", help="Skip supplementary R1 analysis.")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Discover annotators ----
    log.info("Discovering annotators in %s ...", args.annotations_dir)
    annotators = discover_annotators(args.annotations_dir)
    primary_names, supp_names = split_annotators(annotators)

    log.info("Primary (%d): %s", len(primary_names), ", ".join(primary_names))
    log.info("Supplementary (%d): %s", len(supp_names), ", ".join(supp_names))

    # ---- Paper type mapping for stratification ----
    pid_to_type: dict[str, str] = {}
    if args.candidate_papers.exists():
        cp = pd.read_csv(args.candidate_papers, dtype=str)
        if "primary_method" in cp.columns:
            for _, row in cp.iterrows():
                pid = str(row.get("paper_id", "")).strip()
                method = str(row.get("primary_method", "")).strip()
                pid_to_type[pid] = METHOD_TO_TYPE.get(method, "Other")

    # ==================================================================
    # PRIMARY ANALYSIS
    # ==================================================================
    log.info("=" * 60)
    log.info("PRIMARY ANALYSIS: %d annotators", len(primary_names))
    log.info("=" * 60)

    primary_analysis = compute_multi_rater_analysis(annotators, primary_names)

    # ---- Bootstrap CIs ----
    log.info("Paper-level cluster bootstrap (n=%d) ...", args.n_bootstrap)
    primary_cis: dict[str, dict] = {}
    for var in ["method_rung", "claim_rung", "gap_score", "confidence"]:
        log.info("  bootstrapping %s ...", var)
        primary_cis[var] = paper_cluster_bootstrap(
            annotators,
            primary_names,
            var,
            level="ordinal",
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )

    # gap_binary bootstrap (nominal)
    log.info("  bootstrapping gap_binary ...")
    primary_cis["gap_binary"] = paper_cluster_bootstrap(
        annotators,
        primary_names,
        "gap_binary",
        level="nominal",
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 1,
    )

    # ---- Per-model bias ----
    log.info("Computing per-model bias ...")
    all_names = primary_names + supp_names
    bias = per_model_bias(annotators, all_names)

    # ---- Stratified analysis ----
    log.info("Computing stratified analysis ...")
    strat_by_type = stratified_alpha(
        annotators,
        primary_names,
        "gap_score",
        group_fn=lambda pid: pid_to_type.get(pid, "Other"),
    )
    strat_by_calib = stratified_alpha(
        annotators,
        primary_names,
        "gap_score",
        group_fn=lambda pid: "Calibration" if pid in CALIBRATION_IDS else "Non-calibration",
    )

    # ==================================================================
    # SUPPLEMENTARY ANALYSIS (with R1)
    # ==================================================================
    supp_analysis: dict = {}
    supp_cis: dict[str, dict] = {}
    if supp_names and not args.primary_only:
        log.info("=" * 60)
        log.info("SUPPLEMENTARY ANALYSIS: %d annotators (incl. R1)", len(all_names))
        log.info("=" * 60)

        supp_analysis = compute_multi_rater_analysis(annotators, all_names)

        for var in ["method_rung", "claim_rung", "gap_score"]:
            log.info("  bootstrapping %s (supplementary) ...", var)
            supp_cis[var] = paper_cluster_bootstrap(
                annotators,
                all_names,
                var,
                level="ordinal",
                n_bootstrap=args.n_bootstrap,
                seed=args.seed + 100,
            )

    # ==================================================================
    # Outputs
    # ==================================================================
    all_results = {
        "primary": {
            "analysis": primary_analysis,
            "bootstrap_cis": primary_cis,
            "annotators": primary_names,
        },
        "bias": bias,
        "stratified_by_paper_type": strat_by_type,
        "stratified_by_calibration": strat_by_calib,
    }
    if supp_analysis:
        all_results["supplementary"] = {
            "analysis": supp_analysis,
            "bootstrap_cis": supp_cis,
            "annotators": all_names,
        }

    # ---- JSON ----
    json_path = args.output_dir / "iaa_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sanitize(all_results), f, indent=2)
    log.info("Wrote %s", json_path)

    # ---- LaTeX tables ----
    tables: dict[str, str] = {}

    tables["iaa_summary.tex"] = generate_summary_tex(
        primary_analysis, primary_cis, len(primary_names)
    )

    for var in ["method_rung", "claim_rung", "gap_score"]:
        m = primary_analysis.get(var, {})
        pw = m.get("pairwise_matrix")
        pw_names = m.get("pairwise_names", [])
        if pw is not None:
            var_esc = var.replace("_", r"\_")
            tables[f"iaa_pairwise_{var}.tex"] = generate_pairwise_tex(
                np.array(pw),
                pw_names,
                var,
                rf"Pairwise weighted $\kappa$ for \texttt{{{var_esc}}} across {len(pw_names)} annotators.",
            )

    tables["iaa_bias.tex"] = generate_bias_tex(bias)
    tables["iaa_by_paper_type.tex"] = generate_stratified_tex(strat_by_type, "Paper type")

    if supp_analysis:
        tables["iaa_supplementary.tex"] = generate_supplementary_tex(
            primary_analysis, supp_analysis, len(primary_names), len(all_names)
        )

    for filename, content in tables.items():
        out = args.output_dir / filename
        with open(out, "w", encoding="utf-8") as f:
            f.write(content)
        log.info("Wrote %s", out)

    # ---- Plots ----
    plot_pairwise_heatmaps(primary_analysis, args.output_dir)
    plot_model_bias(bias, args.output_dir / "iaa_model_bias.pdf")

    # ---- Summary to stdout ----
    print()
    print("=" * 60)
    print("MULTI-MODEL IAA ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Primary annotators: {len(primary_names)} ({', '.join(_dname(n) for n in primary_names)})")
    if supp_names:
        print(f"Supplementary: {', '.join(_dname(n) for n in supp_names)}")
    print()

    print("PRIMARY ANALYSIS:")
    for var in ["method_rung", "claim_rung", "gap_score", "gap_binary", "confidence"]:
        m = primary_analysis.get(var, {})
        ci = primary_cis.get(var, {})
        alpha_val = m.get("krippendorff_alpha", np.nan)
        lk = m.get("lights_kappa", {})
        lk_mean = lk.get("mean", np.nan) if isinstance(lk, dict) else np.nan
        ci_lo = ci.get("ci_lower", np.nan)
        ci_hi = ci.get("ci_upper", np.nan)
        print(f"  {var:20s}  alpha={_fmt(alpha_val)}  Light's kappa={_fmt(lk_mean)}  95% CI {_fmt_ci(ci_lo, ci_hi)}")

    if supp_analysis:
        print()
        print("SUPPLEMENTARY ANALYSIS (with R1):")
        for var in ["method_rung", "claim_rung", "gap_score"]:
            m = supp_analysis.get(var, {})
            ci = supp_cis.get(var, {})
            alpha_val = m.get("krippendorff_alpha", np.nan)
            ci_lo = ci.get("ci_lower", np.nan)
            ci_hi = ci.get("ci_upper", np.nan)
            print(f"  {var:20s}  alpha={_fmt(alpha_val)}  95% CI {_fmt_ci(ci_lo, ci_hi)}")

    print()
    print(f"Output: {args.output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
