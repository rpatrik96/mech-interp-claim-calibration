# H2 Pilot Study Codebook
## Annotation Guidelines for Testing "Rung-Appropriate Claims Predict Reproducibility"

---

## Overview

For each paper in the sample, extract **all empirical claims** about model internals and annotate:
1. What method was used (determines method_rung)
2. What the paper claims (determines claim_rung)
3. Whether claim_rung > method_rung (overclaiming)

---

## Field Definitions

### paper_id
- arXiv ID or venue-year-title abbreviation
- Example: "2202.05262" or "NeurIPS2022-ROME"

### claim_id
- Unique identifier within paper: paper_id + sequential number
- Example: "2202.05262-01", "2202.05262-02"

### claim_text
- **Verbatim quote** from the paper
- Include enough context to understand the claim
- Use ellipsis [...] for long quotes

### claim_location
- Where in the paper: abstract, introduction, methods, results, discussion, conclusion

### claim_prominence
- **3** = Abstract or title claim (highest visibility)
- **2** = Introduction contribution list or conclusion claim
- **1** = Body text claim (methods, results, discussion)

---

## Method Rung Classification

### Rung 1: Observational/Associational
Methods that establish **correlational evidence only**. No intervention on the model.

| Method | Description | Example Evidence |
|--------|-------------|------------------|
| Linear probing | Train classifier on frozen activations | "Probe accuracy of 85%" |
| Activation logging | Record activations without intervention | "Feature X activates on..." |
| SAE feature attribution | Identify which SAE features activate | "Feature 4123 fires on..." |
| Attention visualization | Inspect attention weights | "Attention concentrates on..." |
| PCA/SVD | Dimensionality reduction analysis | "First PC correlates with..." |
| Correlation analysis | Statistical associations | "r=0.7 between activation and..." |

### Rung 2: Interventional
Methods that establish **causal effects under specific interventions**.

| Method | Description | Example Evidence |
|--------|-------------|------------------|
| Activation patching | Replace activation, measure effect | "Patching head 9.1 restores 80%..." |
| Causal tracing | Systematic patching across positions | "Layer 15 shows highest causal effect" |
| Ablation | Zero/mean out components | "Ablating heads reduces accuracy by 40%" |
| Steering vectors | Add direction, observe output change | "Adding v shifts sentiment..." |
| DAS interchange | Swap aligned subspaces | "IIA of 0.92 on agreement task" |
| ROME/MEMIT edits | Modify weights, observe change | "After edit, model outputs..." |

### Rung 3: Counterfactual
Methods that establish **what would have happened** or **unique mechanisms**.

| Method | Description | Example Evidence |
|--------|-------------|------------------|
| Counterfactual patching | Per-instance counterfactual | "For THIS prompt, had activation been X..." |
| Causal scrubbing | Test if mechanism fully explains | "Scrubbing preserves behavior" |
| Necessity tests | Show component is necessary | "No alternative achieves same behavior" |
| Uniqueness proofs | Demonstrate unique structure | "This is THE circuit" |

---

## Claim Rung Classification

### Rung 1 Linguistic Markers (Associational Claims)
- "correlates with," "is associated with"
- "predicts," "co-occurs with"
- "information is present," "is decodable from"
- "can be extracted," "activates on," "fires when"

**Examples:**
- "Sentiment information is linearly decodable from layer 6"
- "The feature correlates with Python code inputs"
- "Probe accuracy predicts model behavior"

### Rung 2 Linguistic Markers (Causal Claims)
- "causally affects," "has causal effect on"
- "mediates," "influences"
- "is sufficient for," "can produce," "enables"
- "intervening on X changes Y"
- "ablating X degrades Y"

**Examples:**
- "Head 9.1 causally affects the output"
- "This component is sufficient for the behavior"
- "Ablating these heads degrades performance"

### Rung 3 Linguistic Markers (Mechanistic/Counterfactual Claims)
- "encodes," "represents," "computes," "performs"
- "THE mechanism," "THE circuit," "THE feature" (uniqueness)
- "controls," "is responsible for," "underlies"
- "this head DOES X" (functional attribution)
- "the model uses X to do Y" (mechanistic narrative)

### Decision Trees for Polysemous Terms

#### "encodes" / "represents" / "stores"
1. Does the paper provide interventional evidence for this claim?
   - **NO** → Does context make clear the author means "is linearly decodable from"?
     - YES → Code as **R1**. Note: "encodes used in decodability sense"
     - NO → Code as **R3** (default mechanistic reading)
   - **YES** → Is the claim about the intervention's *result* (what changed) or the underlying *mechanism* (how it works)?
     - Result → Code as **R2**
     - Mechanism → Code as **R3**

#### "the circuit" / "the mechanism" / "the algorithm"
1. Does the paper test uniqueness (e.g., show no alternative circuit exists)?
   - **YES** → Code as **R3**
   - **NO** → Is "the" a naming convention (referring to the circuit they found) or a uniqueness claim?
     - If qualifications exist elsewhere in the paper → Code as **R3**, add note: "definite article likely naming convention; qualification at [location]"
     - If no qualifications → Code as **R3**

#### "controls" / "is responsible for"
1. Is the evidence from an intervention (ablation, patching, steering)?
   - **YES** → Does the paper claim the component is the *unique* controller?
     - YES → Code as **R3**
     - NO → Code as **R2** (causal sufficiency, not uniqueness)
   - **NO** → Code as **R3** (mechanistic claim without interventional support)

**Examples:**
- "The model **encodes** subject-verb agreement in this subspace"
- "These heads **perform** the IOI task"
- "**The circuit** moves names from subject to output"
- "This feature **represents** the concept of deception"
- "The model **uses** these components to track entities"

---

## Overclaim Patterns (Common)

| Pattern | Method Used | Typical Claim | Gap |
|---------|-------------|---------------|-----|
| Probing → "encodes" | Linear probe (R1) | "Model encodes X" (R3) | +2 |
| Patching → "THE circuit" | Activation patching (R2) | "This is the circuit" (R3) | +1 |
| Steering → "controls" | Steering vectors (R2) | "Controls concept X" (R3) | +1 |
| SAE → "represents" | SAE attribution (R1) | "Model represents X" (R3) | +2 |
| Attention → "performs" | Attention viz (R1) | "Head performs X" (R3) | +2 |
| Ablation → "necessary" | Ablation (R2) | "Necessary for behavior" (R3) | +1 |

---

## Hedge Flag

### hedge_flag
- **1** = Claim contains an explicit hedge (e.g., "may," "suggests," "potentially," "we hypothesize")
- **0** = No hedge present; claim is stated as established fact

Record hedging separately from confidence. A claim can be high-confidence R3 *with* a hedge (the annotator is confident the claim is R3, and the author hedged it).

---

## Confidence Scoring

Rate your confidence in the rung assignments (1-5):
- **5** = Very confident, clear case
- **4** = Confident, minor ambiguity
- **3** = Moderately confident, some ambiguity
- **2** = Low confidence, significant ambiguity
- **1** = Very uncertain, edge case

Document ambiguous cases in the notes field.

---

## Replication Status

### Coding
- **0** = Successfully replicated (all main claims hold)
- **0.5** = Partially replicated (some claims hold, others fail)
- **1** = Failed replication (main claims do not hold)
- **NA** = No replication attempt found

### Evidence Sources (in priority order)
1. Published replication studies
2. Replication sections in subsequent papers
3. GitHub issues documenting failures
4. Author corrections/errata
5. BlackboxNLP reproducibility track

---

## Annotation Process

1. **Read abstract and introduction** - identify main claims
2. **Identify methods used** - classify each method's rung
3. **For each claim:**
   - Quote verbatim
   - Identify linguistic markers
   - Assign claim_rung based on markers
   - Calculate gap_score
   - Assign confidence
4. **Search for replication evidence** - cite sources
5. **Document edge cases** in notes

---

## Edge Cases and Guidance

### Hedged Claims
- "may encode" → still Rung 3 if followed by mechanistic narrative
- "suggests that" → code based on the underlying claim, not the hedge
- Note hedging in confidence score

### Multiple Methods
- If paper uses multiple methods, code each claim-method pair separately
- Use the method that directly supports each specific claim

### Implicit Claims
- Code both explicit and implicit claims
- Implicit claims from narrative framing should be noted
- Weight implicit claims lower in confidence

### Review/Survey Papers
- Code as NA for replication (not empirical)
- Still useful for method classification reference

---

## Calibration Cases

### Ground Truth: IOI Circuit Paper (Wang et al., 2022)
- **Method:** Activation patching (Rung 2)
- **Claim:** "The circuit" (implies uniqueness, Rung 3)
- **Overclaim:** +1
- **Known issue:** Different ablation strategies yield different circuits

Use this as calibration anchor for Rung 2→3 overclaiming pattern.
