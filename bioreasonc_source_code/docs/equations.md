# BioREASONC-Bench: Formal Score Definitions

## GRASS: Genetic Risk Aggregate Score

The Weighted Gene Risk Score (WGRS) aggregates SNP-level risk contributions with gene length normalization and GWAS disease association weighting.

### SNP-Level Risk Score

For each SNP *i*, the risk contribution is computed as a weighted sum of five components:

$$
\text{SNP\_Risk}_i = w_{\text{clin}} \cdot S_{\text{ClinVar}} + w_{\text{impact}} \cdot S_{\text{Impact}} + w_{\text{freq}} \cdot S_{\text{MAF}} + w_{\text{causal}} \cdot S_{\text{Causal}} + w_{\text{gwas}} \cdot S_{\text{GWAS}}
$$

**Default weights** (sum to 1.0):

| Weight | Value | Component |
|--------|-------|-----------|
| $w_{\text{clin}}$ | 0.25 | Clinical significance (ClinVar) |
| $w_{\text{impact}}$ | 0.20 | Functional impact (VEP) |
| $w_{\text{freq}}$ | 0.15 | Minor allele frequency |
| $w_{\text{causal}}$ | 0.20 | Fine-mapping PIPs (CAUSALdb) |
| $w_{\text{gwas}}$ | 0.20 | GWAS effect size and significance |

#### Component Scores

**ClinVar Score** ($S_{\text{ClinVar}}$):

| Clinical Significance | Score |
|-----------------------|-------|
| Pathogenic | 1.0 |
| Likely Pathogenic | 0.8 |
| Uncertain Significance (VUS) | 0.3 |
| Likely Benign | 0.1 |
| Benign | 0.0 |

**Functional Impact Score** ($S_{\text{Impact}}$):

| Impact Level | Score |
|--------------|-------|
| HIGH | 1.0 |
| MODERATE | 0.6 |
| LOW | 0.3 |
| MODIFIER | 0.1 |

**MAF Score** ($S_{\text{MAF}}$):

$$
S_{\text{MAF}} = 1 - \min(\text{MAF}, 0.5)
$$

**Causal Score** ($S_{\text{Causal}}$):

$$
S_{\text{Causal}} = \max(\text{PIP}_{\text{ABF}}, \text{PIP}_{\text{FINEMAP}}, \text{PIP}_{\text{SuSiE}}, \text{PIP}_{\text{PAINTOR}}, \text{PIP}_{\text{CAVIARBF}}, \text{PIP}_{\text{PolyFun}})
$$

**GWAS Score** ($S_{\text{GWAS}}$):

$$
S_{\text{GWAS}} = \sqrt{S_{\text{effect}} \cdot S_{\text{sig}}}
$$

where:

$$
S_{\text{effect}} = \min\left(\frac{|\beta|}{\beta_{\max}}, 1.0\right)
$$

$$
S_{\text{sig}} = \min\left(\frac{-\log_{10}(p)}{10}, 1.0\right)
$$

### Gene-Level Score

For gene *g* with length $L_g$ (in base pairs):

$$
\text{Gene\_Score}_g = \frac{\sum_{i \in g} \text{SNP\_Risk}_i}{L_g / 1000}
$$

### Weighted Gene Risk Score (WGRS)

The final WGRS incorporates GWAS disease association:

$$
\boxed{\text{WGRS}_g = \text{Gene\_Score}_g \times (1 + \lambda \cdot \text{GWAS\_Score}_g)}
$$

where $\lambda = 5.0$ is the GWAS weight multiplier.

---

## CARES: Causal-Aware Reasoning Evaluation Score

CARES evaluates LLM biomedical reasoning across four taxonomy categories with hallucination penalty and calibration error adjustment.

### Reasoning Categories

| Category | Name | Description |
|----------|------|-------------|
| **S** | Structure-aware | Understanding molecular/clinical structure |
| **C** | Causal-aware | Recognizing causal relationships |
| **R** | Risk-aware | Evaluating risk assessment |
| **M** | Semantic-aware | Semantic knowledge understanding |

### Score Scale

For each question *i*, the score $s_i \in \{0, 1, 2, 3, 4, 5\}$:

| Score | Interpretation |
|-------|----------------|
| 5 | Fully correct, semantically equivalent |
| 4 | Mostly correct, minor imprecisions |
| 3 | Partially correct, missing >20% of key details |
| 2 | Safe abstention with expressed uncertainty |
| 1 | Partial hallucination, mixed correct/incorrect |
| 0 | Complete hallucination, confidently incorrect |

### Category Score

For category $k \in \{S, C, R, M\}$ with question set $Q_k$:

$$
\text{CARES}_k = \frac{1}{|Q_k|} \sum_{i \in Q_k} \frac{s_i}{5}
$$

### Hallucination Rate

A response is classified as a hallucination if $s_i \leq 1$ and confidence $c_i > 0.7$:

$$
\text{HR}_k = \frac{|\{q_i \in Q_k : s_i \leq 1 \land c_i > 0.7\}|}{|Q_k|}
$$

### Hallucination Penalty

The penalty function with domain-calibrated $\alpha$:

$$
\Phi(\text{HR}) = e^{-\alpha \cdot \text{HR}}
$$

**Domain-specific α values** (where $\alpha = -\ln(0.5) / \text{HR}_{\max}$):

| Domain | α | HR_max |
|--------|---|--------|
| Drug Interaction | 10.0 | 7% |
| Clinical Decision | 6.9 | 10% |
| Literature Summary | 4.6 | 15% |
| Research Exploration | 3.5 | 20% |
| Default | 3.5 | 20% |

### Expected Calibration Error

ECE measures the alignment between model confidence and actual accuracy:

$$
\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{N} \cdot |\text{acc}_b - \text{conf}_b|
$$

where:
- $B$ = number of bins (default 10)
- $B_b$ = set of predictions in bin *b*
- $\text{acc}_b$ = average accuracy in bin *b*
- $\text{conf}_b$ = average confidence in bin *b*

### Final CARES Score

The overall CARES score combines weighted category scores with hallucination and calibration adjustments:

$$
\boxed{\text{CARES} = \left[\sum_{k} \tilde{w}_k \cdot \text{CARES}_k\right] \times \sqrt{\Phi(\text{HR}) \times (1 - \text{ECE})}}
$$

where $\tilde{w}_k$ are the normalized category weights (default: $\tilde{w}_k = 0.25$ for all $k$).
