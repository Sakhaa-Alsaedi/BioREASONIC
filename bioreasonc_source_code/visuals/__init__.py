"""
BioREASONC-Bench Visualization Module

Provides comprehensive visualization and summary generation for:
- GRASS (Gene Risk Aggregation) score distributions
- CARES (Causal-Aware Reasoning Evaluation) results
- ROCKET (Risk-Omics Causal Knowledge Enrichment Trust) metrics
- Comparative model performance analysis
- Biomedical knowledge graph visualizations
"""

from .plots import (
    EvaluationPlotter,
    plot_score_distribution,
    plot_model_comparison,
    plot_metric_radar,
    plot_confidence_calibration,
    plot_hallucination_analysis,
    plot_causal_chain_accuracy,
    plot_risk_score_correlation,
    plot_knowledge_coverage,
)
from .summary import (
    SummaryGenerator,
    generate_evaluation_report,
    generate_model_card,
    generate_benchmark_summary,
    export_results_table,
)
from .dashboard import (
    DashboardGenerator,
    create_interactive_dashboard,
    create_comparison_view,
)

__all__ = [
    # Plotter class
    'EvaluationPlotter',
    # Individual plot functions
    'plot_score_distribution',
    'plot_model_comparison',
    'plot_metric_radar',
    'plot_confidence_calibration',
    'plot_hallucination_analysis',
    'plot_causal_chain_accuracy',
    'plot_risk_score_correlation',
    'plot_knowledge_coverage',
    # Summary generation
    'SummaryGenerator',
    'generate_evaluation_report',
    'generate_model_card',
    'generate_benchmark_summary',
    'export_results_table',
    # Dashboard
    'DashboardGenerator',
    'create_interactive_dashboard',
    'create_comparison_view',
]
