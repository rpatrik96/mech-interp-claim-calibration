# Calibration Set Rationales

## Overview

This document provides detailed rationales for the 5 calibration papers, serving as anchor examples for consistent annotation of the remaining papers.

---

## Paper 1: IOI Circuit (2211.00593) - PRIMARY CALIBRATION ANCHOR

**Wang et al., "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small"**

### Method Classification: Rung 2 (Interventional)
- **Primary method:** Path patching (activation patching variant)
- **Supporting methods:** Attention pattern analysis (R1), ablation (R2)
- **Rationale:** The paper's core evidence comes from causal interventions that measure effects of patching activations. This establishes causal sufficiency but not counterfactual necessity/uniqueness.

### Key Overclaim Patterns Identified

| Claim | Linguistic Marker | Method-Claim Gap |
|-------|------------------|------------------|
| "performs IOI task" | "performs" = functional | +1 (R2→R3) |
| "Name Movers move names" | "move" = mechanistic | +1 (R2→R3) |
| "S-Inhibition heads inhibit" | "inhibit" = functional | +1 (R2→R3) |
| "the circuit" | definite article = uniqueness | +1 (R2→R3) |
| "reverse-engineering" | implies complete mechanism | +1 (R2→R3) |

### Replication Status: PARTIAL (0.5)
- **Known issues:** Different ablation strategies (mean ablation vs. zero ablation vs. resample ablation) yield different circuits
- **Evidence:** Zhang et al. (2024), Conmy et al. (2023) ACDC paper notes
- **Implication:** The "circuit" found depends on methodological choices, undermining uniqueness claims

### Calibration Lesson
The IOI paper is the canonical example of **Rung 2 → Rung 3 overclaiming** via:
1. Using definite articles ("THE circuit")
2. Functional verbs ("moves," "inhibits," "performs")
3. Mechanistic narratives ("reverse-engineering the algorithm")

**Use this pattern to identify similar overclaims in other circuit-discovery papers.**

---

## Paper 2: ROME (2202.05262)

**Meng et al., "Locating and Editing Factual Associations in GPT"**

### Method Classification: Rung 2 (Interventional)
- **Primary method:** Causal tracing (activation patching on corrupted inputs)
- **Secondary method:** ROME editing (weight modification)
- **Rationale:** Both methods involve interventions but establish causal effects, not mechanisms.

### Key Overclaim Patterns Identified

| Claim | Linguistic Marker | Method-Claim Gap |
|-------|------------------|------------------|
| "storing factual associations" | "storing" = memory mechanism | +1 (R2→R3) |
| "correspond to localized computations" | "correspond" = identity claim | +1 (R2→R3) |
| "stored in a localized manner" | "stored" + "localized" | +1 (R2→R3) |

### Appropriate Claims (No Overclaim)
- "mediate factual predictions" - "mediate" is proper R2 language
- "ROME is effective" - empirical claim matched to method

### Replication Status: PARTIAL (0.5)
- **Known issues:**
  - Hase et al. (2023) "Does Localization Imply Representation?" questions causal tracing interpretation
  - ROME edits have side effects on related knowledge
  - Localization claims sensitive to prompt variations
- **Implication:** Causal effects real, but "storage" interpretation overclaims

### Calibration Lesson
Storage/memory language ("stores," "encodes," "contains") typically implies Rung 3 mechanistic claims. Causal tracing only establishes causal mediation (R2), not storage mechanisms.

---

## Paper 3: Grokking (2301.05217)

**Nanda et al., "Progress measures for grokking via mechanistic interpretability"**

### Method Classification: Rung 2 (Interventional)
- **Primary method:** Ablation in Fourier space
- **Supporting methods:** Weight analysis (R1), activation analysis (R1)
- **Rationale:** Ablation establishes causal necessity of Fourier components

### Key Overclaim Patterns Identified

| Claim | Linguistic Marker | Method-Claim Gap |
|-------|------------------|------------------|
| "fully reverse engineer" | completeness claim | +1 (R2→R3) |
| "the algorithm" | definite article = uniqueness | +1 (R2→R3) |
| "uses DFT... to convert" | functional mechanism | +1 (R2→R3) |
| "encoded in the weights" | from weight analysis alone | +2 (R1→R3) |

### Replication Status: REPLICATED (0)
- **Strong replication:** Multiple groups have confirmed the Fourier structure
- **Why different from IOI?**
  - Simpler, controlled setting (synthetic task)
  - Algorithm structure mathematically constrained
  - Predictions verified through multiple methods

### Calibration Lesson
Even well-replicated papers can have overclaims at the linguistic level. The grokking claims are less problematic because:
1. Multiple methods converge
2. Mathematical structure constrains possibilities
3. Authors make specific testable predictions

**Pattern:** Small overclaim gap + strong replication = less concern

---

## Paper 4: SAE Evaluation (2409.04478)

**Chaudhary & Geiger, "Evaluating Open-Source Sparse Autoencoders on Disentangling Factual Knowledge"**

### Method Classification: Mixed Rung 1-2
- **Primary method:** SAE feature attribution (R1)
- **Evaluation method:** Interchange intervention (R2)
- **Rationale:** Paper evaluates R1 method using R2 evaluation

### Claim Analysis
This paper is methodologically careful and largely avoids overclaiming:

| Claim | Rung | Notes |
|-------|------|-------|
| "SAEs struggle to reach baseline" | R2 | Appropriate for intervention evidence |
| "features that mediate knowledge" | R2 | "mediate" matches intervention method |
| "useful for causal analysis" | R2 | Claims about causal utility, not mechanism |

### Replication Status: REPLICATED (0)
- Paper is itself an evaluation/replication study
- Findings consistent with other SAE evaluations (Marks et al., Engels et al.)

### Calibration Lesson
**Evaluation papers** tend to have lower overclaim rates because:
1. Explicit comparison to baselines/skylines
2. Focus on method utility, not mechanism claims
3. Negative results naturally cautious

**Pattern:** Papers that evaluate methods rather than discover mechanisms tend to have better claim-method alignment.

---

## Paper 5: Gemini Probes (2601.11516)

**Kramár et al., "Building Production-Ready Probes For Gemini"**

### Method Classification: Rung 1 (Observational)
- **Primary method:** Linear probing
- **Rationale:** Probing is purely observational/correlational

### Claim Analysis
This paper is well-calibrated to its method:

| Claim | Rung | Notes |
|-------|------|-------|
| "probes may be promising" | R1 | Hedged, correlational |
| "probes fail to generalize" | R1 | Empirical observation |
| "successful deployment" | R1 | Outcome claim, not mechanism |

### Overclaim Analysis
No significant overclaims detected. The paper:
- Uses appropriate hedging ("may be")
- Focuses on empirical performance, not mechanisms
- Does not claim probes "detect" or "identify" internal states (which would be R3)

### Replication Status: NA
- Production paper, not standard academic replication context

### Calibration Lesson
**Production/applied papers** focused on probe performance tend to have appropriate claim levels because:
1. Focus on external validity (does it work?)
2. Less incentive for mechanistic narratives
3. Engineering framing vs. science framing

---

## Summary: Overclaim Patterns by Paper Type

| Paper Type | Typical Overclaim | Example |
|------------|------------------|---------|
| Circuit discovery | "THE circuit" + functional verbs | IOI |
| Knowledge localization | "stores," "encodes" | ROME |
| Algorithm analysis | "reverse-engineer," "the algorithm" | Grokking |
| Method evaluation | Low overclaim (comparative) | SAE Eval |
| Production/applied | Low overclaim (empirical focus) | Gemini Probes |

## Key Linguistic Markers Summary

### Rung 3 (Mechanistic) - Watch for:
- "encodes," "represents," "stores," "contains"
- "performs," "computes," "executes," "implements"
- "THE circuit/mechanism/algorithm" (uniqueness)
- "uses X to do Y" (mechanistic narrative)
- "is responsible for," "controls," "underlies"

### Rung 2 (Causal) - Appropriate for interventions:
- "causally affects," "has causal effect"
- "mediates," "influences"
- "is sufficient for," "can produce"
- "intervening on X changes Y"

### Rung 1 (Correlational) - Appropriate for probing/attribution:
- "correlates with," "is associated with"
- "predicts," "is decodable from"
- "activates on," "fires when"
- "information is present"
---

## Inter-Annotator Calibration Notes

For the pilot study (single annotator), use these decision rules:

1. **When in doubt about claim_rung:**
   - Check for functional verbs (performs, computes) → R3
   - Check for uniqueness language (the, only) → R3
   - Check for storage/encoding language → R3

2. **When in doubt about method_rung:**
   - If no intervention on model → R1
   - If intervention but not per-instance counterfactual → R2
   - If establishes unique/necessary mechanism → R3

3. **Edge cases:**
   - Hedged R3 claims ("may encode") → still R3, note hedge in confidence
   - Multi-method papers → use highest-rung method that directly supports claim
   - Implicit claims from narrative → code but weight lower in confidence
