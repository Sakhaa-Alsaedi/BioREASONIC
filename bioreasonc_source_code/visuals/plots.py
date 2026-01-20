"""
BioREASONC-Bench Evaluation Plots

Comprehensive visualization functions for benchmark evaluation results.
Supports GRASS, CARES, and ROCKET score visualizations.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

# Lazy imports for plotting libraries
def _get_plt():
    import matplotlib.pyplot as plt
    return plt

def _get_sns():
    import seaborn as sns
    return sns


@dataclass
class PlotConfig:
    """Configuration for plot styling."""
    figsize: Tuple[int, int] = (10, 6)
    dpi: int = 150
    style: str = "whitegrid"
    palette: str = "husl"
    title_fontsize: int = 14
    label_fontsize: int = 12
    tick_fontsize: int = 10
    legend_fontsize: int = 10
    save_format: str = "png"
    transparent: bool = False


class EvaluationPlotter:
    """
    Main class for generating evaluation visualizations.

    Supports plotting for:
    - Score distributions (GRASS, CARES, ROCKET)
    - Model comparisons
    - Metric radar charts
    - Confidence calibration
    - Hallucination analysis
    - Causal chain accuracy
    """

    def __init__(self, config: Optional[PlotConfig] = None, output_dir: str = "plots"):
        """
        Initialize the plotter.

        Args:
            config: Plot configuration settings
            output_dir: Directory to save plots
        """
        self.config = config or PlotConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set default style
        sns = _get_sns()
        sns.set_style(self.config.style)
        sns.set_palette(self.config.palette)

    def _setup_figure(self, figsize: Optional[Tuple[int, int]] = None) -> Tuple[Any, Any]:
        """Create a new figure with default settings."""
        plt = _get_plt()
        fig, ax = plt.subplots(figsize=figsize or self.config.figsize, dpi=self.config.dpi)
        return fig, ax

    def _save_figure(self, fig: Any, filename: str) -> str:
        """Save figure to output directory."""
        plt = _get_plt()
        filepath = self.output_dir / f"{filename}.{self.config.save_format}"
        fig.savefig(filepath, bbox_inches='tight', transparent=self.config.transparent)
        plt.close(fig)
        return str(filepath)

    def plot_score_distribution(
        self,
        scores: Dict[str, List[float]],
        title: str = "Score Distribution",
        xlabel: str = "Score",
        ylabel: str = "Density",
        filename: str = "score_distribution"
    ) -> str:
        """
        Plot distribution of scores across models.

        Args:
            scores: Dictionary mapping model names to score lists
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            filename: Output filename (without extension)

        Returns:
            Path to saved figure
        """
        plt = _get_plt()
        sns = _get_sns()

        fig, ax = self._setup_figure()

        for model_name, model_scores in scores.items():
            sns.kdeplot(model_scores, label=model_name, ax=ax, fill=True, alpha=0.3)

        ax.set_xlabel(xlabel, fontsize=self.config.label_fontsize)
        ax.set_ylabel(ylabel, fontsize=self.config.label_fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.legend(fontsize=self.config.legend_fontsize)
        ax.tick_params(labelsize=self.config.tick_fontsize)

        return self._save_figure(fig, filename)

    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: Optional[List[str]] = None,
        title: str = "Model Comparison",
        filename: str = "model_comparison"
    ) -> str:
        """
        Create grouped bar chart comparing models across metrics.

        Args:
            results: Dict mapping model names to metric dicts
            metrics: List of metrics to include (default: all)
            title: Plot title
            filename: Output filename

        Returns:
            Path to saved figure
        """
        plt = _get_plt()
        sns = _get_sns()

        models = list(results.keys())
        if metrics is None:
            metrics = list(results[models[0]].keys())

        # Prepare data
        x = np.arange(len(metrics))
        width = 0.8 / len(models)

        fig, ax = self._setup_figure(figsize=(12, 6))

        colors = sns.color_palette(self.config.palette, len(models))

        for i, model in enumerate(models):
            values = [results[model].get(m, 0) for m in metrics]
            offset = (i - len(models)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=model, color=colors[i])

            # Add value labels on bars
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.2f}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Metric', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Score', fontsize=self.config.label_fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend(fontsize=self.config.legend_fontsize, loc='upper right')
        ax.tick_params(labelsize=self.config.tick_fontsize)
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        return self._save_figure(fig, filename)

    def plot_metric_radar(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: Optional[List[str]] = None,
        title: str = "Metric Radar Chart",
        filename: str = "metric_radar"
    ) -> str:
        """
        Create radar/spider chart for multi-dimensional comparison.

        Args:
            results: Dict mapping model names to metric dicts
            metrics: List of metrics to include
            title: Plot title
            filename: Output filename

        Returns:
            Path to saved figure
        """
        plt = _get_plt()
        sns = _get_sns()

        models = list(results.keys())
        if metrics is None:
            metrics = list(results[models[0]].keys())

        # Number of metrics
        num_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True), dpi=self.config.dpi)

        colors = sns.color_palette(self.config.palette, len(models))

        for i, model in enumerate(models):
            values = [results[model].get(m, 0) for m in metrics]
            values += values[:1]  # Complete the loop

            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=self.config.tick_fontsize)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=self.config.title_fontsize, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=self.config.legend_fontsize)

        return self._save_figure(fig, filename)

    def plot_confidence_calibration(
        self,
        predictions: List[float],
        confidences: List[float],
        actuals: List[int],
        n_bins: int = 10,
        title: str = "Confidence Calibration",
        filename: str = "calibration"
    ) -> str:
        """
        Plot calibration curve for model confidence.

        Args:
            predictions: Model predictions (0-1)
            confidences: Model confidence scores (0-1)
            actuals: Ground truth labels (0 or 1)
            n_bins: Number of bins for calibration
            title: Plot title
            filename: Output filename

        Returns:
            Path to saved figure
        """
        plt = _get_plt()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=self.config.dpi)

        # Calibration curve
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for i in range(n_bins):
            mask = (np.array(confidences) >= bin_edges[i]) & (np.array(confidences) < bin_edges[i+1])
            if np.sum(mask) > 0:
                bin_accuracies.append(np.mean(np.array(actuals)[mask]))
                bin_confidences.append(np.mean(np.array(confidences)[mask]))
                bin_counts.append(np.sum(mask))
            else:
                bin_accuracies.append(np.nan)
                bin_confidences.append(np.nan)
                bin_counts.append(0)

        # Plot calibration curve
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax1.plot(bin_confidences, bin_accuracies, 'o-', label='Model', markersize=8)
        ax1.fill_between([0, 1], [0, 1], alpha=0.1)
        ax1.set_xlabel('Mean Predicted Confidence', fontsize=self.config.label_fontsize)
        ax1.set_ylabel('Fraction of Positives', fontsize=self.config.label_fontsize)
        ax1.set_title('Calibration Curve', fontsize=self.config.title_fontsize)
        ax1.legend(fontsize=self.config.legend_fontsize)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)

        # Histogram of confidences
        ax2.hist(confidences, bins=n_bins, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Confidence', fontsize=self.config.label_fontsize)
        ax2.set_ylabel('Count', fontsize=self.config.label_fontsize)
        ax2.set_title('Confidence Distribution', fontsize=self.config.title_fontsize)

        fig.suptitle(title, fontsize=self.config.title_fontsize + 2, y=1.02)
        plt.tight_layout()

        return self._save_figure(fig, filename)

    def plot_hallucination_analysis(
        self,
        results: Dict[str, Dict[str, float]],
        title: str = "Hallucination Analysis",
        filename: str = "hallucination_analysis"
    ) -> str:
        """
        Visualize hallucination rates and types across models.

        Args:
            results: Dict with model -> {hallucination_rate, fabrication_rate,
                                         contradiction_rate, unsupported_rate}
            title: Plot title
            filename: Output filename

        Returns:
            Path to saved figure
        """
        plt = _get_plt()
        sns = _get_sns()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=self.config.dpi)

        models = list(results.keys())
        hallucination_types = ['fabrication_rate', 'contradiction_rate', 'unsupported_rate']

        # Overall hallucination rate
        overall_rates = [results[m].get('hallucination_rate', 0) for m in models]
        colors = sns.color_palette('Reds_r', len(models))
        bars = ax1.bar(models, overall_rates, color=colors, edgecolor='black')
        ax1.set_ylabel('Hallucination Rate', fontsize=self.config.label_fontsize)
        ax1.set_title('Overall Hallucination Rate', fontsize=self.config.title_fontsize)
        ax1.set_ylim(0, max(overall_rates) * 1.2 if overall_rates else 1)

        # Add value labels
        for bar, val in zip(bars, overall_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.1%}', ha='center', va='bottom', fontsize=10)

        # Breakdown by type (stacked bar)
        x = np.arange(len(models))
        width = 0.6

        bottom = np.zeros(len(models))
        type_colors = sns.color_palette('Set2', len(hallucination_types))

        for i, h_type in enumerate(hallucination_types):
            values = [results[m].get(h_type, 0) for m in models]
            ax2.bar(x, values, width, label=h_type.replace('_rate', '').title(),
                   bottom=bottom, color=type_colors[i])
            bottom += np.array(values)

        ax2.set_xticks(x)
        ax2.set_xticklabels(models)
        ax2.set_ylabel('Rate', fontsize=self.config.label_fontsize)
        ax2.set_title('Hallucination Type Breakdown', fontsize=self.config.title_fontsize)
        ax2.legend(fontsize=self.config.legend_fontsize)

        fig.suptitle(title, fontsize=self.config.title_fontsize + 2, y=1.02)
        plt.tight_layout()

        return self._save_figure(fig, filename)

    def plot_causal_chain_accuracy(
        self,
        chain_results: Dict[str, List[Dict[str, Any]]],
        title: str = "Causal Chain Accuracy",
        filename: str = "causal_chain_accuracy"
    ) -> str:
        """
        Visualize accuracy at each step of causal reasoning chains.

        Args:
            chain_results: Dict with model -> list of chain evaluations
                          Each chain has 'step_accuracies' list
            title: Plot title
            filename: Output filename

        Returns:
            Path to saved figure
        """
        plt = _get_plt()
        sns = _get_sns()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=self.config.dpi)

        models = list(chain_results.keys())
        colors = sns.color_palette(self.config.palette, len(models))

        # Average accuracy by chain step
        max_steps = max(
            max(len(chain.get('step_accuracies', [])) for chain in chains)
            for chains in chain_results.values()
        )

        for i, model in enumerate(models):
            step_accs = []
            for step in range(max_steps):
                accs = [
                    chain['step_accuracies'][step]
                    for chain in chain_results[model]
                    if len(chain.get('step_accuracies', [])) > step
                ]
                if accs:
                    step_accs.append(np.mean(accs))
                else:
                    step_accs.append(np.nan)

            ax1.plot(range(1, max_steps + 1), step_accs, 'o-', label=model,
                    color=colors[i], linewidth=2, markersize=8)

        ax1.set_xlabel('Chain Step', fontsize=self.config.label_fontsize)
        ax1.set_ylabel('Accuracy', fontsize=self.config.label_fontsize)
        ax1.set_title('Accuracy by Chain Step', fontsize=self.config.title_fontsize)
        ax1.legend(fontsize=self.config.legend_fontsize)
        ax1.set_ylim(0, 1.05)

        # Chain length distribution
        for i, model in enumerate(models):
            chain_lengths = [len(chain.get('step_accuracies', [])) for chain in chain_results[model]]
            ax2.hist(chain_lengths, bins=range(1, max_steps + 2), alpha=0.5,
                    label=model, color=colors[i], edgecolor='black')

        ax2.set_xlabel('Chain Length', fontsize=self.config.label_fontsize)
        ax2.set_ylabel('Count', fontsize=self.config.label_fontsize)
        ax2.set_title('Chain Length Distribution', fontsize=self.config.title_fontsize)
        ax2.legend(fontsize=self.config.legend_fontsize)

        fig.suptitle(title, fontsize=self.config.title_fontsize + 2, y=1.02)
        plt.tight_layout()

        return self._save_figure(fig, filename)

    def plot_risk_score_correlation(
        self,
        predicted_scores: List[float],
        ground_truth_scores: List[float],
        model_name: str = "Model",
        title: str = "Risk Score Correlation",
        filename: str = "risk_correlation"
    ) -> str:
        """
        Plot correlation between predicted and ground truth risk scores.

        Args:
            predicted_scores: Model's predicted risk scores
            ground_truth_scores: Ground truth risk scores
            model_name: Name of the model
            title: Plot title
            filename: Output filename

        Returns:
            Path to saved figure
        """
        plt = _get_plt()
        sns = _get_sns()

        fig, ax = self._setup_figure(figsize=(8, 8))

        # Scatter plot with regression line
        ax.scatter(ground_truth_scores, predicted_scores, alpha=0.5, s=50)

        # Add regression line
        z = np.polyfit(ground_truth_scores, predicted_scores, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(ground_truth_scores), max(ground_truth_scores), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Regression line')

        # Perfect correlation line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect correlation')

        # Calculate correlation
        correlation = np.corrcoef(ground_truth_scores, predicted_scores)[0, 1]

        ax.set_xlabel('Ground Truth Risk Score', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Predicted Risk Score', fontsize=self.config.label_fontsize)
        ax.set_title(f'{title}\n{model_name} (r = {correlation:.3f})',
                    fontsize=self.config.title_fontsize)
        ax.legend(fontsize=self.config.legend_fontsize)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        return self._save_figure(fig, filename)

    def plot_knowledge_coverage(
        self,
        coverage_data: Dict[str, Dict[str, float]],
        title: str = "Knowledge Base Coverage",
        filename: str = "knowledge_coverage"
    ) -> str:
        """
        Visualize coverage across different knowledge bases.

        Args:
            coverage_data: Dict with model -> {kb_name: coverage_score}
            title: Plot title
            filename: Output filename

        Returns:
            Path to saved figure
        """
        plt = _get_plt()
        sns = _get_sns()

        fig, ax = self._setup_figure(figsize=(12, 6))

        models = list(coverage_data.keys())
        kbs = list(coverage_data[models[0]].keys())

        x = np.arange(len(kbs))
        width = 0.8 / len(models)

        colors = sns.color_palette(self.config.palette, len(models))

        for i, model in enumerate(models):
            values = [coverage_data[model].get(kb, 0) for kb in kbs]
            offset = (i - len(models)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=model, color=colors[i])

        ax.set_xlabel('Knowledge Base', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Coverage', fontsize=self.config.label_fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.set_xticks(x)
        ax.set_xticklabels(kbs, rotation=45, ha='right')
        ax.legend(fontsize=self.config.legend_fontsize)
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        return self._save_figure(fig, filename)

    def plot_grass_breakdown(
        self,
        grass_results: Dict[str, Dict[str, float]],
        title: str = "GRASS Score Breakdown",
        filename: str = "grass_breakdown"
    ) -> str:
        """
        Visualize GRASS score components.

        Args:
            grass_results: Dict with model -> {gene_accuracy, snp_accuracy,
                                               risk_calibration, aggregate_score}
            title: Plot title
            filename: Output filename

        Returns:
            Path to saved figure
        """
        plt = _get_plt()
        sns = _get_sns()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=self.config.dpi)

        models = list(grass_results.keys())
        components = ['gene_accuracy', 'snp_accuracy', 'risk_calibration']

        # Component breakdown
        x = np.arange(len(models))
        width = 0.25
        colors = sns.color_palette('Set2', len(components))

        for i, comp in enumerate(components):
            values = [grass_results[m].get(comp, 0) for m in models]
            ax1.bar(x + i*width, values, width, label=comp.replace('_', ' ').title(),
                   color=colors[i])

        ax1.set_xticks(x + width)
        ax1.set_xticklabels(models)
        ax1.set_ylabel('Score', fontsize=self.config.label_fontsize)
        ax1.set_title('GRASS Components', fontsize=self.config.title_fontsize)
        ax1.legend(fontsize=self.config.legend_fontsize)
        ax1.set_ylim(0, 1.1)

        # Overall GRASS score
        overall = [grass_results[m].get('aggregate_score', 0) for m in models]
        colors = sns.color_palette('Blues_d', len(models))
        bars = ax2.bar(models, overall, color=colors, edgecolor='black')

        for bar, val in zip(bars, overall):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax2.set_ylabel('GRASS Score', fontsize=self.config.label_fontsize)
        ax2.set_title('Overall GRASS Score', fontsize=self.config.title_fontsize)
        ax2.set_ylim(0, 1.1)

        fig.suptitle(title, fontsize=self.config.title_fontsize + 2, y=1.02)
        plt.tight_layout()

        return self._save_figure(fig, filename)

    def plot_cares_analysis(
        self,
        cares_results: Dict[str, Dict[str, float]],
        title: str = "CARES Score Analysis",
        filename: str = "cares_analysis"
    ) -> str:
        """
        Visualize CARES score with hallucination penalty breakdown.

        Args:
            cares_results: Dict with model -> {base_score, hallucination_penalty,
                                               final_score, confidence}
            title: Plot title
            filename: Output filename

        Returns:
            Path to saved figure
        """
        plt = _get_plt()
        sns = _get_sns()

        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=self.config.dpi)

        models = list(cares_results.keys())

        # 1. Base score vs Final score
        ax1 = axes[0, 0]
        x = np.arange(len(models))
        width = 0.35

        base = [cares_results[m].get('base_score', 0) for m in models]
        final = [cares_results[m].get('final_score', 0) for m in models]

        ax1.bar(x - width/2, base, width, label='Base Score', color='steelblue')
        ax1.bar(x + width/2, final, width, label='Final Score', color='darkgreen')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.set_ylabel('Score')
        ax1.set_title('Base vs Final Score')
        ax1.legend()
        ax1.set_ylim(0, 1.1)

        # 2. Hallucination penalty
        ax2 = axes[0, 1]
        penalty = [cares_results[m].get('hallucination_penalty', 0) for m in models]
        colors = sns.color_palette('Reds', len(models))
        ax2.bar(models, penalty, color=colors, edgecolor='black')
        ax2.set_ylabel('Penalty')
        ax2.set_title('Hallucination Penalty')
        ax2.set_xticklabels(models, rotation=45, ha='right')

        # 3. Confidence distribution
        ax3 = axes[1, 0]
        confidence = [cares_results[m].get('confidence', 0) for m in models]
        ax3.bar(models, confidence, color='purple', alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Confidence')
        ax3.set_title('Model Confidence')
        ax3.set_xticklabels(models, rotation=45, ha='right')
        ax3.set_ylim(0, 1.1)

        # 4. Score decomposition (waterfall-like)
        ax4 = axes[1, 1]
        for i, model in enumerate(models):
            base_val = cares_results[model].get('base_score', 0)
            penalty_val = cares_results[model].get('hallucination_penalty', 0)
            final_val = cares_results[model].get('final_score', 0)

            ax4.plot([0, 1, 2], [base_val, base_val - penalty_val, final_val],
                    'o-', label=model, markersize=8)

        ax4.set_xticks([0, 1, 2])
        ax4.set_xticklabels(['Base', 'After Penalty', 'Final'])
        ax4.set_ylabel('Score')
        ax4.set_title('Score Decomposition')
        ax4.legend(fontsize=8)
        ax4.set_ylim(0, 1.1)

        fig.suptitle(title, fontsize=self.config.title_fontsize + 2)
        plt.tight_layout()

        return self._save_figure(fig, filename)

    def plot_rocket_components(
        self,
        rocket_results: Dict[str, Dict[str, float]],
        title: str = "ROCKET Score Components",
        filename: str = "rocket_components"
    ) -> str:
        """
        Visualize ROCKET score components.

        Args:
            rocket_results: Dict with model -> {R_component, O_component,
                                                C_component, K_component,
                                                E_component, T_component,
                                                overall_score}
            title: Plot title
            filename: Output filename

        Returns:
            Path to saved figure
        """
        plt = _get_plt()
        sns = _get_sns()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=self.config.dpi)

        models = list(rocket_results.keys())
        components = ['R_component', 'O_component', 'C_component',
                     'K_component', 'E_component', 'T_component']
        labels = ['Risk\nAccuracy', 'Omics\nIntegration', 'Causal\nReasoning',
                 'Knowledge\nCoverage', 'Evidence\nQuality', 'Trust\nCalibration']

        # Radar chart for components
        num_vars = len(components)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        colors = sns.color_palette(self.config.palette, len(models))

        ax1 = plt.subplot(121, polar=True)

        for i, model in enumerate(models):
            values = [rocket_results[model].get(c, 0) for c in components]
            values += values[:1]
            ax1.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
            ax1.fill(angles, values, alpha=0.15, color=colors[i])

        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(labels, fontsize=9)
        ax1.set_ylim(0, 1)
        ax1.set_title('Component Breakdown', fontsize=self.config.title_fontsize, pad=20)
        ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)

        # Overall ROCKET score bar chart
        ax2 = plt.subplot(122)
        overall = [rocket_results[m].get('overall_score', 0) for m in models]
        bars = ax2.bar(models, overall, color=colors, edgecolor='black')

        for bar, val in zip(bars, overall):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax2.set_ylabel('ROCKET Score', fontsize=self.config.label_fontsize)
        ax2.set_title('Overall ROCKET Score', fontsize=self.config.title_fontsize)
        ax2.set_ylim(0, 1.1)

        fig.suptitle(title, fontsize=self.config.title_fontsize + 2, y=1.02)
        plt.tight_layout()

        return self._save_figure(fig, filename)

    def plot_standard_vs_novel_metrics(
        self,
        results: Dict[str, Dict[str, float]],
        title: str = "Standard vs Novel Metrics Comparison",
        filename: str = "standard_vs_novel"
    ) -> str:
        """
        Plot standard NLP metrics alongside novel BioREASONC metrics.

        Args:
            results: Dict with model -> {token_f1, bleu_1, rouge_l, meteor,
                                          grass_score, cares_score, rocket_score, ...}
            title: Plot title
            filename: Output filename

        Returns:
            Path to saved figure
        """
        plt = _get_plt()
        sns = _get_sns()

        fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=self.config.dpi)

        models = list(results.keys())
        colors = sns.color_palette(self.config.palette, len(models))

        # 1. Standard vs Novel bar comparison
        ax1 = axes[0, 0]
        standard_metrics = ['token_f1', 'bleu_1', 'rouge_l', 'meteor']
        novel_metrics = ['grass_score', 'cares_score', 'rocket_score']

        x = np.arange(len(models))
        width = 0.12

        for i, metric in enumerate(standard_metrics):
            values = [results[m].get(metric, 0) for m in models]
            ax1.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())

        for i, metric in enumerate(novel_metrics):
            values = [results[m].get(metric, 0) for m in models]
            ax1.bar(x + (len(standard_metrics) + i)*width, values, width,
                   label=metric.replace('_', ' ').upper().replace('SCORE', ''),
                   hatch='//', edgecolor='black')

        ax1.set_xticks(x + width * 3)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.set_ylabel('Score')
        ax1.set_title('All Metrics by Model')
        ax1.legend(loc='upper right', fontsize=8, ncol=2)
        ax1.set_ylim(0, 1.1)

        # 2. Correlation heatmap
        ax2 = axes[0, 1]
        all_metrics = standard_metrics + novel_metrics

        # Build correlation matrix
        metric_values = {metric: [results[m].get(metric, 0) for m in models]
                        for metric in all_metrics}

        corr_matrix = np.zeros((len(all_metrics), len(all_metrics)))
        for i, m1 in enumerate(all_metrics):
            for j, m2 in enumerate(all_metrics):
                if np.std(metric_values[m1]) > 0 and np.std(metric_values[m2]) > 0:
                    corr_matrix[i, j] = np.corrcoef(metric_values[m1], metric_values[m2])[0, 1]
                else:
                    corr_matrix[i, j] = 1.0 if i == j else 0.0

        labels = [m.replace('_', '\n').title() for m in all_metrics]
        im = ax2.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
        ax2.set_xticks(range(len(all_metrics)))
        ax2.set_yticks(range(len(all_metrics)))
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax2.set_yticklabels(labels, fontsize=8)
        ax2.set_title('Metric Correlations')

        # Add correlation values
        for i in range(len(all_metrics)):
            for j in range(len(all_metrics)):
                ax2.text(j, i, f'{corr_matrix[i, j]:.2f}',
                        ha='center', va='center', fontsize=7,
                        color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')

        plt.colorbar(im, ax=ax2, shrink=0.8)

        # 3. Standard metrics grouped
        ax3 = axes[1, 0]
        std_categories = {
            'QA': ['token_f1', 'exact_match'],
            'Generation': ['bleu_1', 'rouge_l', 'meteor'],
            'Semantic': ['semantic_similarity']
        }

        x = np.arange(len(models))
        width = 0.15
        offset = 0

        for cat_name, cat_metrics in std_categories.items():
            for metric in cat_metrics:
                if metric in list(results.values())[0]:
                    values = [results[m].get(metric, 0) for m in models]
                    ax3.bar(x + offset*width, values, width,
                           label=f'{cat_name}: {metric.replace("_", " ").title()}')
                    offset += 1

        ax3.set_xticks(x + width * (offset/2))
        ax3.set_xticklabels(models, rotation=45, ha='right')
        ax3.set_ylabel('Score')
        ax3.set_title('Standard Metrics by Category')
        ax3.legend(loc='upper right', fontsize=7, ncol=2)
        ax3.set_ylim(0, 1.1)

        # 4. Novel metrics with standard baseline
        ax4 = axes[1, 1]

        # Plot standard metric average as baseline
        std_avg = [np.mean([results[m].get(metric, 0) for metric in standard_metrics])
                   for m in models]

        x = np.arange(len(models))
        width = 0.2

        ax4.bar(x - 1.5*width, std_avg, width, label='Std Metrics Avg',
               color='gray', alpha=0.7)
        ax4.bar(x - 0.5*width, [results[m].get('grass_score', 0) for m in models],
               width, label='GRASS', color=colors[0])
        ax4.bar(x + 0.5*width, [results[m].get('cares_score', 0) for m in models],
               width, label='CARES', color=colors[1])
        ax4.bar(x + 1.5*width, [results[m].get('rocket_score', 0) for m in models],
               width, label='ROCKET', color=colors[2])

        ax4.set_xticks(x)
        ax4.set_xticklabels(models, rotation=45, ha='right')
        ax4.set_ylabel('Score')
        ax4.set_title('Novel Metrics vs Standard Baseline')
        ax4.legend(loc='upper right')
        ax4.set_ylim(0, 1.1)

        # Add horizontal line at 0.5 as threshold
        ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')

        fig.suptitle(title, fontsize=self.config.title_fontsize + 2, y=1.02)
        plt.tight_layout()

        return self._save_figure(fig, filename)

    def plot_comprehensive_evaluation(
        self,
        results: Dict[str, Dict[str, float]],
        title: str = "Comprehensive Evaluation Dashboard",
        filename: str = "comprehensive_evaluation"
    ) -> str:
        """
        Create a comprehensive evaluation plot with all metrics.

        Args:
            results: Full evaluation results with all metrics
            title: Plot title
            filename: Output filename

        Returns:
            Path to saved figure
        """
        plt = _get_plt()
        sns = _get_sns()

        fig = plt.figure(figsize=(16, 14), dpi=self.config.dpi)

        models = list(results.keys())
        colors = sns.color_palette(self.config.palette, len(models))

        # Layout: 3x3 grid
        # Row 1: Overview bar chart, Radar chart, Ranking
        # Row 2: Generation metrics, QA metrics, Novel metrics
        # Row 3: Summary table (spans full width)

        # 1. Overall comparison bar chart
        ax1 = fig.add_subplot(3, 3, 1)
        key_metrics = ['token_f1', 'rouge_l', 'grass_score', 'cares_score', 'rocket_score']
        x = np.arange(len(key_metrics))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            values = [results[model].get(m, 0) for m in key_metrics]
            ax1.bar(x + i*width, values, width, label=model, color=colors[i])

        ax1.set_xticks(x + width * len(models) / 2)
        ax1.set_xticklabels([m.replace('_', '\n').title() for m in key_metrics], fontsize=8)
        ax1.set_ylabel('Score')
        ax1.set_title('Key Metrics Overview')
        ax1.legend(fontsize=7, loc='upper right')
        ax1.set_ylim(0, 1.1)

        # 2. Radar chart
        ax2 = fig.add_subplot(3, 3, 2, polar=True)
        radar_metrics = ['token_f1', 'bleu_1', 'rouge_l', 'grass_score', 'cares_score']
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]

        for i, model in enumerate(models[:4]):  # Limit to 4 models
            values = [results[model].get(m, 0) for m in radar_metrics]
            values += values[:1]
            ax2.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
            ax2.fill(angles, values, alpha=0.15, color=colors[i])

        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels([m.replace('_', '\n').title() for m in radar_metrics], fontsize=8)
        ax2.set_ylim(0, 1)
        ax2.set_title('Multi-Metric Radar', pad=15)
        ax2.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.3, 1.0))

        # 3. Ranking chart
        ax3 = fig.add_subplot(3, 3, 3)
        rocket_scores = [(m, results[m].get('rocket_score', 0)) for m in models]
        rocket_scores.sort(key=lambda x: x[1], reverse=True)

        y_pos = np.arange(len(models))
        ax3.barh(y_pos, [s for _, s in rocket_scores],
                color=[colors[models.index(m)] for m, _ in rocket_scores])
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([m for m, _ in rocket_scores])
        ax3.set_xlabel('ROCKET Score')
        ax3.set_title('Model Ranking')
        ax3.set_xlim(0, 1.1)

        # Add rank labels
        for i, (model, score) in enumerate(rocket_scores):
            ax3.text(score + 0.02, i, f'#{i+1}', va='center', fontsize=10, fontweight='bold')

        # 4. Generation metrics
        ax4 = fig.add_subplot(3, 3, 4)
        gen_metrics = ['bleu_1', 'bleu_4', 'rouge_1', 'rouge_l', 'meteor']
        x = np.arange(len(gen_metrics))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            values = [results[model].get(m, 0) for m in gen_metrics]
            ax4.bar(x + i*width, values, width, label=model, color=colors[i])

        ax4.set_xticks(x + width * len(models) / 2)
        ax4.set_xticklabels([m.upper().replace('_', '-') for m in gen_metrics], fontsize=8)
        ax4.set_ylabel('Score')
        ax4.set_title('Generation Metrics')
        ax4.set_ylim(0, 1.1)

        # 5. QA metrics
        ax5 = fig.add_subplot(3, 3, 5)
        qa_metrics = ['exact_match', 'token_f1', 'semantic_similarity']
        x = np.arange(len(qa_metrics))
        width = 0.25

        for i, model in enumerate(models):
            values = [results[model].get(m, 0) for m in qa_metrics]
            ax5.bar(x + i*width, values, width, label=model, color=colors[i])

        ax5.set_xticks(x + width * len(models) / 2)
        ax5.set_xticklabels(['Exact\nMatch', 'Token\nF1', 'Semantic\nSim'], fontsize=8)
        ax5.set_ylabel('Score')
        ax5.set_title('QA Metrics')
        ax5.legend(fontsize=7)
        ax5.set_ylim(0, 1.1)

        # 6. Novel metrics comparison
        ax6 = fig.add_subplot(3, 3, 6)
        novel_metrics = ['grass_score', 'cares_score', 'rocket_score']
        x = np.arange(len(novel_metrics))
        width = 0.2

        for i, model in enumerate(models):
            values = [results[model].get(m, 0) for m in novel_metrics]
            ax6.bar(x + i*width, values, width, label=model, color=colors[i])

        ax6.set_xticks(x + width * len(models) / 2)
        ax6.set_xticklabels(['GRASS', 'CARES', 'ROCKET'], fontsize=10, fontweight='bold')
        ax6.set_ylabel('Score')
        ax6.set_title('Novel BioREASONC Metrics')
        ax6.legend(fontsize=7)
        ax6.set_ylim(0, 1.1)

        # 7-9. Summary text box (spans bottom row)
        ax7 = fig.add_subplot(3, 1, 3)
        ax7.axis('off')

        # Create summary text
        best_model = max(models, key=lambda m: results[m].get('rocket_score', 0))
        best_rocket = results[best_model].get('rocket_score', 0)

        summary_lines = [
            f"═══════════════════════════════════════════════════════════════════",
            f"                    EVALUATION SUMMARY",
            f"═══════════════════════════════════════════════════════════════════",
            f"",
            f"  Models Evaluated: {len(models)}",
            f"  Best Overall (ROCKET): {best_model} ({best_rocket:.3f})",
            f"",
            f"  Average Scores Across Models:",
        ]

        for metric in ['token_f1', 'rouge_l', 'grass_score', 'cares_score', 'rocket_score']:
            avg = np.mean([results[m].get(metric, 0) for m in models])
            summary_lines.append(f"    • {metric.replace('_', ' ').title()}: {avg:.3f}")

        summary_lines.extend([
            f"",
            f"═══════════════════════════════════════════════════════════════════",
        ])

        summary_text = '\n'.join(summary_lines)
        ax7.text(0.5, 0.5, summary_text, transform=ax7.transAxes,
                fontsize=11, fontfamily='monospace',
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle(title, fontsize=self.config.title_fontsize + 4, y=0.98, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        return self._save_figure(fig, filename)


# Convenience functions for quick plotting
def plot_score_distribution(scores: Dict[str, List[float]], output_dir: str = "plots", **kwargs) -> str:
    """Quick function to plot score distribution."""
    plotter = EvaluationPlotter(output_dir=output_dir)
    return plotter.plot_score_distribution(scores, **kwargs)


def plot_model_comparison(results: Dict[str, Dict[str, float]], output_dir: str = "plots", **kwargs) -> str:
    """Quick function to plot model comparison."""
    plotter = EvaluationPlotter(output_dir=output_dir)
    return plotter.plot_model_comparison(results, **kwargs)


def plot_metric_radar(results: Dict[str, Dict[str, float]], output_dir: str = "plots", **kwargs) -> str:
    """Quick function to plot metric radar chart."""
    plotter = EvaluationPlotter(output_dir=output_dir)
    return plotter.plot_metric_radar(results, **kwargs)


def plot_confidence_calibration(
    predictions: List[float],
    confidences: List[float],
    actuals: List[int],
    output_dir: str = "plots",
    **kwargs
) -> str:
    """Quick function to plot confidence calibration."""
    plotter = EvaluationPlotter(output_dir=output_dir)
    return plotter.plot_confidence_calibration(predictions, confidences, actuals, **kwargs)


def plot_hallucination_analysis(results: Dict[str, Dict[str, float]], output_dir: str = "plots", **kwargs) -> str:
    """Quick function to plot hallucination analysis."""
    plotter = EvaluationPlotter(output_dir=output_dir)
    return plotter.plot_hallucination_analysis(results, **kwargs)


def plot_causal_chain_accuracy(chain_results: Dict[str, List[Dict]], output_dir: str = "plots", **kwargs) -> str:
    """Quick function to plot causal chain accuracy."""
    plotter = EvaluationPlotter(output_dir=output_dir)
    return plotter.plot_causal_chain_accuracy(chain_results, **kwargs)


def plot_risk_score_correlation(
    predicted: List[float],
    ground_truth: List[float],
    output_dir: str = "plots",
    **kwargs
) -> str:
    """Quick function to plot risk score correlation."""
    plotter = EvaluationPlotter(output_dir=output_dir)
    return plotter.plot_risk_score_correlation(predicted, ground_truth, **kwargs)


def plot_knowledge_coverage(coverage_data: Dict[str, Dict[str, float]], output_dir: str = "plots", **kwargs) -> str:
    """Quick function to plot knowledge coverage."""
    plotter = EvaluationPlotter(output_dir=output_dir)
    return plotter.plot_knowledge_coverage(coverage_data, **kwargs)
