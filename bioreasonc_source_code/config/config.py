"""
BioREASONC Configuration File

Add your API keys here to run the pipeline.
"""

# =============================================================================
# API KEYS - ADD YOUR KEYS HERE
# =============================================================================

# OpenAI API Key (for GPT-4o-mini)
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY = ""  # <-- ADD YOUR KEY HERE

# Anthropic API Key (for Claude)
# Get from: https://console.anthropic.com/
ANTHROPIC_API_KEY = ""  # <-- ADD YOUR KEY HERE

# Google Gemini API Key (optional)
# Get from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY = ""  # <-- ADD YOUR KEY HERE (optional)


# =============================================================================
# KNOWLEDGE GRAPH DATA PATH
# =============================================================================

# CAUSALdb2 v2.1 Knowledge Graph (updated source)
KG_DATA_PATH = "/ibex/user/alsaedsb/ROCKET/Data/CAUSALdb2/v2.1/kg/gene_disease_kg_corrected.csv"
SNP_DISEASE_PATH = "/ibex/user/alsaedsb/ROCKET/Data/CAUSALdb2/v2.1/kg/snp_disease_kg.csv"
SNP_GENE_PATH = "/ibex/user/alsaedsb/ROCKET/Data/CAUSALdb2/v2.1/kg/snp_gene_kg.csv"


# =============================================================================
# PIPELINE SETTINGS
# =============================================================================

PIPELINE_SETTINGS = {
    # ==========================================================================
    # GENERATOR SETTINGS
    # ==========================================================================
    "use_cot_answers": True,      # Use Chain-of-Thought answers for Causal taxonomy
    "max_per_template": 50,       # Max items per question template

    # ==========================================================================
    # PARAPHRASER SETTINGS
    # ==========================================================================
    "num_paraphrases": 2,         # Number of paraphrases per question (1-3)
    "use_llm_paraphrase": True,   # Use LLM for paraphrasing (vs rule-based)

    # ==========================================================================
    # EXPLAINER SETTINGS
    # ==========================================================================
    "target_explanation_words": 35,  # Target word count for explanations
    "use_llm_explanation": True,     # Use LLM for explanations (vs templates)

    # ==========================================================================
    # VALIDATOR SETTINGS
    # ==========================================================================
    "passing_threshold": 4.0,     # Minimum score to pass (1-5)
    "require_majority": True,     # Require majority of validators to agree
    "min_validators": 2,          # Minimum number of validators needed

    # ==========================================================================
    # FEEDBACK LOOP SETTINGS (prevents infinite loops)
    # ==========================================================================
    "max_feedback_iterations": 3,      # Maximum human review rounds
    "target_agreement_rate": 0.95,     # Stop when 95% agreement reached
    "skip_already_regenerated": True,  # Don't regenerate same item twice

    # ==========================================================================
    # OUTPUT SETTINGS
    # ==========================================================================
    "output_dir": "./bioreasonc_output",  # Output directory for results
}


# =============================================================================
# BENCHMARK GENERATION SETTINGS (User configurable)
# =============================================================================

BENCHMARK_SETTINGS = {
    # Which taxonomies to generate
    "taxonomies": ["S", "C", "R", "M"],  # S=Structure, C=Causal, R=Risk, M=Semantic

    # Data sources (paths to GWAS data files)
    "data_sources": [
        {"path": "./data/COVID-19_risk_gene.csv", "disease": "COVID-19"},
        {"path": "./data/RA_risk genes.csv", "disease": "Rheumatoid Arthritis"},
    ],

    # Generation limits
    "max_items_per_disease": 100,  # Max items per disease
    "max_total_items": 500,        # Max total items in benchmark

    # Human review settings
    "skip_human_review": False,    # Set True to skip human validation step
    "auto_approve_threshold": 4.5, # Auto-approve items with score >= this
}


# =============================================================================
# HELPER FUNCTION
# =============================================================================

def get_pipeline_config():
    """Get pipeline configuration with API keys."""
    import sys
    import os
    # Add src to path (relative to repo root, not config/)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo_root, 'src'))

    from bioreasonc_creator.pipeline import PipelineConfig

    return PipelineConfig(
        openai_api_key=OPENAI_API_KEY,
        anthropic_api_key=ANTHROPIC_API_KEY,
        gemini_api_key=GEMINI_API_KEY,
        **PIPELINE_SETTINGS
    )


def validate_api_keys():
    """Check if API keys are configured."""
    keys = {
        "OpenAI": OPENAI_API_KEY,
        "Anthropic": ANTHROPIC_API_KEY,
        "Gemini": GEMINI_API_KEY
    }

    print("API Key Status:")
    print("-" * 40)

    configured = 0
    for name, key in keys.items():
        if key and len(key) > 10:
            print(f"  {name}: ✓ Configured")
            configured += 1
        else:
            print(f"  {name}: ✗ Not configured")

    print("-" * 40)
    print(f"Total configured: {configured}/3")

    if configured < 2:
        print("\n⚠️  Warning: At least 2 API keys recommended for consensus validation")

    return configured >= 1


if __name__ == "__main__":
    validate_api_keys()
