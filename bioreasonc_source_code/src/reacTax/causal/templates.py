"""
Question Templates for Causal Reasoning

Defines templates for generating benchmark questions about:
- Causal vs. associative relationships
- Mendelian Randomization inference
- PC algorithm discovery
- Refutation test results
- Sensitivity analysis
- Estimator comparison
"""

from typing import Dict, List

# Question templates organized by label type
CAUSAL_QUESTION_TEMPLATES: Dict[str, List[str]] = {
    # Original templates (preserved for backward compatibility)
    "C-CAUSAL-VS-ASSOC": [
        "Is the relationship between {gene} and {disease} severity causal or merely associative based on the genetic evidence?",
        "Based on GWAS data showing OR={odds_ratio:.2f} and p={p_value:.2e}, can we conclude that {gene} has a causal effect on {disease}?",
        "What type of evidence would be needed to establish causality between {gene} and {disease} beyond the observed association?",
    ],

    "C-MR-INFERENCE": [
        "Based on Mendelian Randomization using {method} method, what is the causal effect estimate of {exposure} on {outcome}?",
        "The IVW MR estimate for {exposure} on {outcome} is {estimate:.3f} (p={p_value:.2e}). What does this suggest about causality?",
        "How do we interpret the MR-Egger intercept of {intercept:.3f} for the {exposure}-{outcome} relationship?",
    ],

    "C-PC-DISCOVERY": [
        "According to PC algorithm causal discovery, what is the causal relationship between {source} and {target}?",
        "The PC algorithm discovered an edge {source} {edge_type} {target}. What does this imply about their causal relationship?",
        "Why did the PC algorithm orient the edge from {source} to {target} rather than the reverse?",
    ],

    # New templates for DoWhy integration
    "C-REFUTATION": [
        "After applying the {refutation_method} refutation test, is the causal effect of {treatment} on {outcome} robust?",
        "The {refutation_method} test changed the estimate from {original:.3f} to {new:.3f}. Does this support or undermine the causal claim?",
        "What does a {result} in the {refutation_method} refutation test tell us about the validity of the {treatment}-{outcome} causal effect?",
        "If adding a random common cause changes the estimate by {change:.1%}, should we trust the causal inference?",
    ],

    "C-SENSITIVITY": [
        "How sensitive is the causal effect estimate of {treatment} on {outcome} to unobserved confounding?",
        "What magnitude of unobserved confounding would be needed to nullify the {treatment}-{outcome} causal effect of {estimate:.3f}?",
        "The sensitivity analysis shows the estimate would change by {change:.1%} with a confounder of effect size {confounder_effect:.2f}. Is this concerning?",
        "Using sensitivity analysis, assess the robustness of the {gene}-disease relationship to potential unmeasured confounders.",
    ],

    "C-ESTIMATOR-COMPARE": [
        "Compare the causal effect estimates from IVW ({ivw:.3f}), Egger ({egger:.3f}), and weighted median ({wm:.3f}) methods for {exposure}-{outcome}.",
        "Do different causal estimators agree on the {treatment}-{outcome} effect? IVW gives {ivw:.3f} while Egger gives {egger:.3f}.",
        "Which estimation method provides the most reliable causal effect for {gene} on {disease}: backdoor adjustment or instrumental variables?",
        "The IVW and weighted median estimates differ by {diff:.1%}. What might explain this discrepancy?",
    ],

    "C-IDENTIFIABILITY": [
        "Is the causal effect of {treatment} on {outcome} identifiable given the causal graph?",
        "What adjustment set is needed to identify the {treatment}-{outcome} causal effect?",
        "Are there valid instruments for estimating the causal effect of {gene} on disease severity?",
        "Does the backdoor criterion hold for estimating the effect of {treatment} on {outcome}?",
    ],

    "C-DISCOVERY-COMPARE": [
        "Do PC, FCI, and GES algorithms agree on the causal structure between {genes}?",
        "What causal relationships does the {algorithm} algorithm discover among {genes}?",
        "Compare the causal graphs produced by PC and GES for {genes}. What are the key differences?",
        "The PC algorithm found {pc_edges} directed edges while GES found {ges_edges}. How do we reconcile these results?",
    ],

    "C-CONFOUNDER-DETECT": [
        "Based on the causal graph, is {variable} a confounder of the {treatment}-{outcome} relationship?",
        "What variables should be adjusted for when estimating the effect of {treatment} on {outcome}?",
        "The MR-Egger intercept is {intercept:.3f} (p={p_value:.2e}). Does this indicate horizontal pleiotropy?",
    ],
}

# Answer templates
CAUSAL_ANSWER_TEMPLATES: Dict[str, List[str]] = {
    "C-CAUSAL-VS-ASSOC": [
        "The evidence suggests {conclusion}. GWAS shows OR={odds_ratio:.2f}, p={p_value:.2e}. "
        "Causal inference requires additional evidence from MR or intervention studies.",
        "Based on the GWAS association alone, we can only conclude {conclusion}. "
        "To establish causality, we would need: (1) MR evidence, (2) temporal precedence, "
        "(3) dose-response relationship, and (4) biological plausibility.",
    ],

    "C-REFUTATION": [
        "The estimate is {robustness}. Original effect: {original:.3f}, after refutation: {new:.3f} "
        "({change:.1%} change). {interpretation}",
        "The {refutation_method} test {result}. This {implication} the causal interpretation of the "
        "{treatment}-{outcome} relationship.",
    ],

    "C-SENSITIVITY": [
        "Effect estimate: {estimate:.3f} (SE={std_error:.3f}). Sensitivity analysis shows the "
        "estimate would be nullified by unobserved confounding with effect size greater than {threshold:.3f}.",
        "The causal effect appears {robustness} to unobserved confounding. A confounder would need "
        "to have an effect of at least {threshold:.2f} on both {treatment} and {outcome} to explain away the observed effect.",
    ],
}

# Explanation templates
CAUSAL_EXPLANATION_TEMPLATES: Dict[str, str] = {
    "C-CAUSAL-VS-ASSOC": (
        "Distinguishing causation from association requires: (1) consistent association, "
        "(2) temporal precedence, (3) dose-response relationship, and (4) experimental or MR evidence. "
        "GWAS associations alone demonstrate correlation but not causation."
    ),

    "C-MR-INFERENCE": (
        "Mendelian Randomization uses genetic variants as instrumental variables, leveraging "
        "random allocation of alleles at conception to estimate causal effects. The {method} method "
        "{method_description}."
    ),

    "C-PC-DISCOVERY": (
        "The PC algorithm identifies causal structure by testing conditional independence. "
        "Directed edges (→) indicate causal direction determined by v-structure orientation. "
        "Undirected edges (--) indicate association without determined causal direction."
    ),

    "C-REFUTATION": (
        "Refutation tests validate causal estimates by checking robustness to assumption violations. "
        "The {refutation_method} test specifically checks {assumption_tested}. "
        "A passed test increases confidence in the causal estimate."
    ),

    "C-SENSITIVITY": (
        "Sensitivity analysis quantifies how strong unobserved confounding would need to be "
        "to explain away the observed effect. Robust estimates require large confounding "
        "(relative to observed effect) to nullify the causal relationship."
    ),

    "C-ESTIMATOR-COMPARE": (
        "Different estimators make different assumptions: IVW assumes no pleiotropy, "
        "MR-Egger allows for directional pleiotropy (detectable via intercept), "
        "and weighted median is robust to up to 50% invalid instruments. "
        "Agreement across methods strengthens causal inference."
    ),

    "C-IDENTIFIABILITY": (
        "A causal effect is identifiable if it can be computed from observational data "
        "given the causal graph. The backdoor criterion requires adjusting for all "
        "confounders, while the frontdoor criterion uses mediators."
    ),
}

# Method descriptions for explanations
MR_METHOD_DESCRIPTIONS: Dict[str, str] = {
    "ivw": "assumes all instruments are valid (no horizontal pleiotropy)",
    "egger": "allows for directional pleiotropy by including an intercept term",
    "weighted_median": "is robust to up to 50% of instruments being invalid",
}

# Refutation assumption descriptions
REFUTATION_ASSUMPTIONS: Dict[str, str] = {
    "random_common_cause": "sensitivity to unobserved confounding",
    "placebo_treatment_refuter": "the validity of the treatment-outcome relationship",
    "data_subset_refuter": "stability across different data subsets",
    "bootstrap_refuter": "statistical reliability of the estimate",
    "dummy_outcome_refuter": "correct specification of the outcome variable",
}


def get_question_template(label: str, index: int = 0) -> str:
    """Get a question template by label and index"""
    templates = CAUSAL_QUESTION_TEMPLATES.get(label, [])
    if not templates:
        return ""
    return templates[index % len(templates)]


def get_answer_template(label: str, index: int = 0) -> str:
    """Get an answer template by label and index"""
    templates = CAUSAL_ANSWER_TEMPLATES.get(label, [])
    if not templates:
        return ""
    return templates[index % len(templates)]


def get_explanation(label: str) -> str:
    """Get explanation text for a label"""
    return CAUSAL_EXPLANATION_TEMPLATES.get(label, "")
