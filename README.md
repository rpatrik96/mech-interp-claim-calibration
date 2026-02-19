# Multi-LLM Claim Calibration for Mechanistic Interpretability

[![arXiv](https://img.shields.io/badge/arXiv-2602.16698-b31b1b.svg)](https://arxiv.org/abs/2602.16698)

Automated pipeline for detecting overclaiming in mechanistic interpretability papers. Multiple LLMs independently classify claims using Pearl's causal ladder, enabling inter-annotator agreement analysis and systematic overclaim detection.

This repository contains the code and data for the pilot study in:

> **Position: Causality is Key for Interpretability Claims to Generalise**
> *Patrik Reizinger et al.* ([arXiv:2602.16698](https://arxiv.org/abs/2602.16698))

## Overview

Mechanistic interpretability papers often make claims whose causal strength exceeds what their methods can support. This pipeline quantifies such overclaiming by:

1. **Classifying methods** on Pearl's causal ladder: observational (R1), interventional (R2), counterfactual (R3)
2. **Classifying claims** on the same ladder
3. **Computing gap scores**: `max(0, claim_rung - method_rung)` — 0 = calibrated, 1 = mild overclaim, 2 = strong overclaim
4. **Measuring agreement** across 7 LLM annotators via Krippendorff's alpha with bootstrap CIs

## Pipeline

```
candidate_papers.csv
        |
        v
  fetch_papers.py -----> paper_texts/{id}.txt (+ checksums.json)
        |
        v
  annotate.py ----------> annotations_multi/{model}/annotations_classify.csv
   (classify mode)         annotations_multi/{model}/run_metadata.json
   (extract mode)          annotations_multi/{model}/raw/{paper}_{claim}.json
        |
        v
  validate.py ----------> schema + content validation
        |
        v
  compute_iaa.py --------> IAA metrics (Krippendorff alpha, kappa, ICC, bootstrap CIs)
        |
        v
  analyze_pilot.py ------> LaTeX tables, overclaim distributions, statistical tests
```

## Quick Start

```bash
# Install dependencies
uv sync

# Fetch paper texts from arXiv
uv run python fetch_papers.py

# Run annotation with a model
uv run python annotate.py --mode classify --model claude --papers all

# Validate outputs
uv run python validate.py --all

# Compute inter-annotator agreement
uv run python compute_iaa.py

# Generate analysis tables and figures
uv run python analyze_pilot.py
```

## Models

Seven primary annotators classify all 186 claims; one supplementary annotator covers 166/186.

| Provider | Model | Key | Claims |
|----------|-------|-----|--------|
| Anthropic | Claude Opus 4.5 | `claude` | 186 |
| OpenAI | GPT-5.2 | `gpt5` | 186 |
| Anthropic | Claude Sonnet 4 | `sonnet` | 186 |
| OpenRouter | Gemini 3 Flash | `openrouter-gemini` | 186 |
| OpenRouter | Mistral Large 2512 | `openrouter-mistral` | 186 |
| OpenRouter | DeepSeek V3 | `openrouter-deepseek-v3` | 186 |
| OpenRouter | Qwen 3 235B | `openrouter-qwen` | 186 |
| OpenRouter | DeepSeek R1 | `openrouter-deepseek-r1` | 166 (supplementary) |

## Key Concepts

**Causal Rungs (Pearl's Ladder):**
- **R1 — Observational:** Probing, activation logging, attention visualization
- **R2 — Interventional:** Patching, causal tracing, ablation, steering
- **R3 — Counterfactual:** Counterfactual patching, causal scrubbing, mechanistic explanation

**Gap Score:** `max(0, claim_rung - method_rung)` — 0 = no overclaim, 1 = mild, 2 = strong.

**Pre-registered threshold:** Krippendorff's alpha >= 0.6 for "substantial agreement."

## Data

| File | Description |
|------|-------------|
| `candidate_papers.csv` | 54 papers with metadata (paper_id, title, authors, year, venue, primary_method) |
| `annotations.csv` | 186 human annotations (13 columns including rungs, gap_score, replication status) |
| `calibration_annotations.csv` | Calibration subset (5 papers) |
| `annotation_config.yaml` | Model definitions, paths, thresholds, calibration paper IDs |
| `CODEBOOK.md` | Annotation guidelines: rung definitions, linguistic markers, decision trees |
| `calibration_rationales.md` | Ground-truth rationales for calibration papers |

## Directory Structure

```
.
├── annotate.py                 # Multi-LLM annotation pipeline
├── fetch_papers.py             # arXiv download & text extraction
├── validate.py                 # CSV schema/content validation
├── compute_iaa.py              # Inter-annotator agreement metrics
├── analyze_pilot.py            # Statistical analysis & LaTeX output
├── annotation_config.yaml      # Model & pipeline configuration
├── CODEBOOK.md                 # Annotation guidelines
├── calibration_rationales.md   # Ground-truth calibration examples
├── candidate_papers.csv        # Paper metadata (54 papers)
├── annotations.csv             # Human annotations (186 claims)
├── calibration_annotations.csv # Calibration subset
├── fact_checking_log.md        # Replication evidence
├── pyproject.toml              # Dependencies & project config
├── paper_texts/                # Extracted paper texts + checksums
├── annotations_multi/          # Multi-model annotation outputs
│   ├── claude/                 #   Claude Opus 4.5 (186 claims)
│   ├── gpt5/                   #   GPT-5.2 (186 claims)
│   ├── sonnet/                 #   Claude Sonnet 4 (186 claims)
│   ├── openrouter-gemini/      #   Gemini 3 Flash (186 claims)
│   ├── openrouter-mistral/     #   Mistral Large (186 claims)
│   ├── openrouter-deepseek-v3/ #   DeepSeek V3 (186 claims)
│   ├── openrouter-qwen/        #   Qwen 3 (186 claims)
│   └── openrouter-deepseek-r1/ #   DeepSeek R1 (166 claims, supplementary)
├── output/                     # Analysis outputs (LaTeX tables, figures, JSON)
└── tests/                      # Pytest suite
```

## Environment Variables

| Variable | Provider | Required For |
|----------|----------|--------------|
| `ANTHROPIC_API_KEY` | Anthropic | `claude`, `sonnet` |
| `OPENAI_API_KEY` | OpenAI | `gpt5` |
| `OPENROUTER_API_KEY` | OpenRouter | `openrouter-gemini`, `openrouter-mistral`, `openrouter-deepseek-v3`, `openrouter-qwen`, `openrouter-deepseek-r1` |

## Testing

```bash
uv run pytest tests/ -v --tb=short
```

## Reproducibility

- **PyMuPDF pinned to 1.25.3** for deterministic text extraction
- **SHA-256 checksums** in `paper_texts/checksums.json` verify paper downloads
- **Run metadata** (timestamps, config hashes, model info) in each model's `run_metadata.json`
- **Pre-specified thresholds** set before data analysis
- **Full test suite** covering validation, IAA computation, and end-to-end pipeline

## Citation

```bibtex
@article{reizinger2026causal,
  title={Position: Causality is Key for Interpretability Claims to Generalise},
  author={Reizinger, Patrik and others},
  journal={arXiv preprint arXiv:2602.16698},
  year={2026}
}
```

## License

This project is released under the MIT License.
