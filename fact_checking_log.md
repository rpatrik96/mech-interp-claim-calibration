# Fact-Checking Log for H2 Pilot Study

> **Terminology note:** This document was written during the pilot study when the metric was called "overclaim score." The paper refers to the same quantity as "gap score" ($\max(0, \text{claim\_rung} - \text{method\_rung})$) to emphasise that gaps may reflect linguistic convention rather than epistemic overclaiming. The CSV column has been renamed to `gap_score`.

## Overview
This document records verification of 43 claims (across 12 papers) in annotations.csv against original arXiv papers.

**Date started:** 2026-01-20
**Date completed:** 2026-01-20
**Methodology:** Paper-first approach using arXiv MCP tools

---

## Phase 1: Calibration Papers

### Paper: 2211.00593 - IOI Circuit (Wang et al.)
*Status: VERIFIED*

| Claim ID | Text | Location | Method | Rungs | Status |
|----------|------|----------|--------|-------|--------|
| 01 | Verified verbatim | Correct | Correct | Correct | OK |
| 02 | Verified verbatim | Correct | Correct | Correct | OK |
| 03 | Verified verbatim | Correct | Correct | Correct | OK |
| 04 | Accurate paraphrase | Correct | Correct | Correct | OK (minor) |
| 05 | Accurate paraphrase | Correct | Correct | Correct | OK (minor) |
| 06 | Verified verbatim | **CORRECTED: abstract** | Correct | Correct | Note: Paper hedges claim |

**Corrections applied:** Claim 06 location changed from "conclusion" to "abstract"

### Paper: 2202.05262 - ROME (Meng et al.)
*Status: VERIFIED*

| Claim ID | Text | Location | Method | Rungs | Status |
|----------|------|----------|--------|-------|--------|
| 01 | Verified (minor comma) | Correct | Correct | Correct | OK |
| 02 | Verified verbatim | Correct | Correct | Correct | OK |
| 03 | Verified verbatim | Correct | Correct | Correct | OK |
| 04 | Verified verbatim | **CORRECTED: abstract** | Correct | Correct | OK |

**Corrections applied:** Claim 04 location changed from "results" to "abstract", prominence 1->3

### Paper: 2301.05217 - Grokking (Nanda et al.)
*Status: VERIFIED*

| Claim ID | Text | Location | Method | Rungs | Status |
|----------|------|----------|--------|-------|--------|
| 01 | Verified verbatim | Correct | Correct | Correct | OK |
| 02 | Verified verbatim | Correct | Correct | Correct | OK |
| 03 | Verified verbatim | Correct | **CORRECTED** | **CORRECTED** | See below |

**Corrections applied for Claim 03:**
- Method: "Weight Analysis" -> "Weight Analysis + Ablation" (paper uses ablation-based progress measures)
- method_rung: 1 -> 2 (ablation is interventional)
- gap_score: 2 -> 1 (with method_rung=2, gap is only 1)

### Paper: 2409.04478 - SAE Evaluation
*Status: VERIFIED*

| Claim ID | Text | Location | Method | Rungs | Status |
|----------|------|----------|--------|-------|--------|
| 01 | Verified verbatim | Correct | Correct | Correct | OK |
| 02 | Verified verbatim | Correct | Correct | Correct | OK |

**No corrections needed.** Low-overclaim evaluation paper correctly annotated.

### Paper: 2601.11516 - Gemini Probes
*Status: VERIFIED*

| Claim ID | Text | Location | Method | Rungs | Status |
|----------|------|----------|--------|-------|--------|
| 01 | Verified verbatim | Correct | OK* | Correct | OK |
| 02 | Verified verbatim | Correct | OK* | Correct | OK |

**Note:** Method could be "Activation Probing" (paper uses multiple probe architectures), but "Linear Probing" is acceptable as primary baseline.

---

## Phase 2: Additional Papers

### Paper: 2304.14997 - ACDC (Conmy et al.)
*Status: VERIFIED*

| Claim ID | Text | Location | Method | Rungs | Status |
|----------|------|----------|--------|-------|--------|
| 01 | Verified verbatim | Correct | Correct | Correct | OK |
| 02 | Verified verbatim | Correct | Correct | Correct | OK |
| 03 | Verified verbatim | Correct | Correct | Correct | OK |
| 04 | Verified verbatim | **CORRECTED: abstract** | Correct | Correct | OK |

**Corrections applied:** Claim 04 location changed from "body" to "abstract", prominence 1->3

### Paper: 2407.14008 - Mamba IOI
*Status: VERIFIED*

| Claim ID | Text | Location | Method | Rungs | Status |
|----------|------|----------|--------|-------|--------|
| 01 | Verified verbatim | Correct | Correct | Correct | OK |
| 02 | Verified verbatim | Correct | Correct | Correct | OK |
| 03 | Verified verbatim | Correct | Correct | Correct | OK |
| 04 | Verified verbatim | Correct | **CORRECTED** | **CORRECTED** | See below |

**Corrections applied for Claim 04:**
- Method: "Linear Probing" -> "Activation Steering (CAA)" (paper uses subtract-and-add steering intervention with >95% success)
- method_rung: 1 -> 2 (steering is interventional)
- gap_score: 2 -> 1 (with method_rung=2, gap is only 1)

### Paper: 2501.17148 - AxBench
*Status: VERIFIED*

| Claim ID | Text | Location | Method | Rungs | Status |
|----------|------|----------|--------|-------|--------|
| 01 | Verified verbatim | Correct | OK* | Correct | OK |
| 02 | Verified verbatim | Correct | Correct | Correct | OK |
| 03 | Verified verbatim | Correct | Correct | Correct | OK |

**Note:** Claim 01 method could be just "Steering" rather than "Steering + Probing".

### Paper: 2404.03646 - Mamba Factual Knowledge
*Status: VERIFIED*

| Claim ID | Text | Location | Method | Rungs | Status |
|----------|------|----------|--------|-------|--------|
| 01 | Verified verbatim | Correct | Correct | Correct | OK |
| 02 | Verified verbatim | Correct | Correct | Correct | OK |
| 03 | Verified verbatim | Correct | Debatable | Debatable | OK (see note) |

**Note for Claim 03:** LRE method is more sophisticated than simple probing but still observational. Current annotation (method_rung=1, gap_score=2) is defensible given paper's hedged language.

### Paper: 2505.14685 - Theory of Mind
*Status: VERIFIED*

| Claim ID | Text | Location | Method | Rungs | Status |
|----------|------|----------|--------|-------|--------|
| 01 | Verified verbatim | Correct | OK* | Correct | OK |
| 02 | Verified verbatim | Correct | OK* | Correct | OK |
| 03 | Verified verbatim | **CORRECTED: abstract** | Correct | Correct | OK |
| 04 | Verified verbatim | Correct | OK* | Correct | OK |

**Corrections applied:** Claim 03 location changed from "body" to "abstract", prominence 1->3

**Note:** Method annotations incomplete - paper uses BOTH causal mediation AND causal abstraction.

### Paper: 2510.06182 - Entity Binding
*Status: VERIFIED*

| Claim ID | Text | Location | Method | Rungs | Status |
|----------|------|----------|--------|-------|--------|
| 01 | Verified verbatim | Correct | Correct | Correct | OK |
| 02 | Verified verbatim | Correct | Correct | Correct | OK |
| 03 | Verified verbatim | OK* | Correct | Correct | OK |
| 04 | Verified verbatim | Correct | Correct | Correct | OK |

**Note:** Claim 03 appears in both abstract and body; "body" annotation is acceptable.

### Paper: 2411.16105 - Circuit Generalization
*Status: VERIFIED*

| Claim ID | Text | Location | Method | Rungs | Status |
|----------|------|----------|--------|-------|--------|
| 01 | Verified verbatim | Correct | OK* | Correct | OK |
| 02 | Verified verbatim | Correct | OK* | Correct | OK |
| 03 | Verified verbatim | **CORRECTED: abstract** | OK* | Correct | OK |
| 04 | Verified verbatim | Correct | OK* | Correct | See note |

**Corrections applied:** Claim 03 location changed from "body" to "abstract", prominence 1->3

**Notes:**
- Method is "path patching/mean ablation" not explicitly "activation patching" (related methods)
- Claim 04 is field-level definitional framing from prior work, not a novel claim by authors

---

## Summary Statistics

| Metric | Original | Verified | Change |
|--------|----------|----------|--------|
| Total claims | 43 | 43 | 0 |
| Overclaim 0 | 17 (38.6%) | 17 (38.6%) | 0 |
| Overclaim +1 | 23 (52.3%) | 25 (56.8%) | +2 |
| Overclaim +2 | 4 (9.1%) | 2 (4.5%) | -2 |
| Corrections needed | - | 7 | - |

### Key Statistical Changes
Two claims that were originally scored as +2 overclaim have been corrected to +1:
1. **2301.05217-03** (Grokking): method_rung 1->2 (ablation used), overclaim 2->1
2. **2407.14008-04** (Mamba IOI): method_rung 1->2 (steering used), overclaim 2->1

### Corrections Summary
| Claim ID | Field | Original | Corrected |
|----------|-------|----------|-----------|
| 2211.00593-06 | claim_location | conclusion | abstract |
| 2202.05262-04 | claim_location | results | abstract |
| 2301.05217-03 | method_rung | 1 | 2 |
| 2301.05217-03 | gap_score | 2 | 1 |
| 2304.14997-04 | claim_location | body | abstract |
| 2407.14008-04 | method_used | Linear Probing | Activation Steering |
| 2407.14008-04 | method_rung | 1 | 2 |
| 2407.14008-04 | gap_score | 2 | 1 |
| 2505.14685-03 | claim_location | body | abstract |
| 2411.16105-03 | claim_location | body | abstract |

---

## Verification Methodology

1. **Paper reading:** Used `mcp__arxiv__read_paper_content` to access full text
2. **Claim search:** Used `mcp__arxiv__search_in_paper` to locate specific phrases
3. **Cross-reference:** Verified against CODEBOOK.md classification criteria
4. **Documentation:** All findings recorded in corrections_needed.csv

## Conclusion

The H2 pilot study annotations are largely accurate. Of 43 claims:
- **36 claims** (84%): Fully verified, no corrections needed
- **7 claims** (16%): Required corrections

The corrections primarily involved:
- **Location errors** (5): Claims marked as "body" or other locations that actually appear in abstracts
- **Method misclassification** (2): Methods that include intervention were marked as observational-only

The overall overclaim distribution shifted slightly:
- Original: 38.6% no overclaim, 52.3% mild (+1), 9.1% strong (+2)
- Verified: 38.6% no overclaim, 56.8% mild (+1), 4.5% strong (+2)

This reflects more accurate identification of interventional methods used to support mechanistic claims.
