"""
BioREASONC-Bench Summary Generation

Generates evaluation reports, model cards, and benchmark summaries
for biomedical causal reasoning evaluation results.
"""

import os
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import csv


@dataclass
class ModelResult:
    """Container for a model's evaluation results."""
    model_name: str
    grass_score: float = 0.0
    cares_score: float = 0.0
    rocket_score: float = 0.0
    hallucination_rate: float = 0.0
    confidence: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class SummaryGenerator:
    """
    Generates evaluation summaries and reports.

    Supports multiple output formats:
    - Markdown reports
    - HTML reports
    - JSON exports
    - CSV tables
    - Model cards
    """

    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the summary generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_evaluation_report(
        self,
        results: Dict[str, ModelResult],
        benchmark_name: str = "BioREASONC-Bench",
        include_details: bool = True,
        format: str = "markdown"
    ) -> str:
        """
        Generate comprehensive evaluation report.

        Args:
            results: Dict mapping model names to ModelResult objects
            benchmark_name: Name of the benchmark
            include_details: Whether to include detailed breakdowns
            format: Output format ('markdown', 'html', 'json')

        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if format == "markdown":
            return self._generate_markdown_report(results, benchmark_name, timestamp, include_details)
        elif format == "html":
            return self._generate_html_report(results, benchmark_name, timestamp, include_details)
        elif format == "json":
            return self._generate_json_report(results, benchmark_name, timestamp)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _generate_markdown_report(
        self,
        results: Dict[str, ModelResult],
        benchmark_name: str,
        timestamp: str,
        include_details: bool
    ) -> str:
        """Generate markdown evaluation report."""
        lines = []

        # Header
        lines.append(f"# {benchmark_name} Evaluation Report")
        lines.append(f"\n**Generated:** {timestamp}\n")

        # Executive Summary
        lines.append("## Executive Summary\n")
        models = list(results.keys())
        lines.append(f"- **Models Evaluated:** {len(models)}")

        # Find best performing model
        best_model = max(results.keys(), key=lambda m: results[m].rocket_score)
        best_score = results[best_model].rocket_score
        lines.append(f"- **Best Performing Model:** {best_model} (ROCKET: {best_score:.3f})")

        avg_hallucination = sum(r.hallucination_rate for r in results.values()) / len(results)
        lines.append(f"- **Average Hallucination Rate:** {avg_hallucination:.1%}\n")

        # Overall Scores Table
        lines.append("## Overall Scores\n")
        lines.append("| Model | GRASS | CARES | ROCKET | Hallucination Rate |")
        lines.append("|-------|-------|-------|--------|-------------------|")

        for model_name, result in sorted(results.items(), key=lambda x: x[1].rocket_score, reverse=True):
            lines.append(
                f"| {model_name} | {result.grass_score:.3f} | {result.cares_score:.3f} | "
                f"{result.rocket_score:.3f} | {result.hallucination_rate:.1%} |"
            )

        # Score Interpretation
        lines.append("\n## Score Interpretation\n")
        lines.append("### GRASS (Gene Risk Aggregation Scoring System)")
        lines.append("Measures accuracy in aggregating SNP-level risk to gene-level predictions.")
        lines.append("- **0.8-1.0:** Excellent risk aggregation")
        lines.append("- **0.6-0.8:** Good performance")
        lines.append("- **0.4-0.6:** Moderate accuracy")
        lines.append("- **< 0.4:** Needs improvement\n")

        lines.append("### CARES (Causal-Aware Reasoning Evaluation Score)")
        lines.append("Evaluates causal reasoning with hallucination penalty.")
        lines.append("- Incorporates LLM-as-judge evaluation")
        lines.append("- Penalizes fabricated or unsupported claims\n")

        lines.append("### ROCKET (Risk-Omics Causal Knowledge Enrichment Trust)")
        lines.append("Comprehensive score combining:")
        lines.append("- **R**isk accuracy")
        lines.append("- **O**mics integration")
        lines.append("- **C**ausal reasoning")
        lines.append("- **K**nowledge coverage")
        lines.append("- **E**vidence quality")
        lines.append("- **T**rust calibration\n")

        if include_details:
            # Detailed breakdowns
            lines.append("## Detailed Results\n")

            for model_name, result in results.items():
                lines.append(f"### {model_name}\n")

                if 'grass_details' in result.details:
                    lines.append("#### GRASS Breakdown")
                    for key, value in result.details['grass_details'].items():
                        lines.append(f"- {key.replace('_', ' ').title()}: {value:.3f}")
                    lines.append("")

                if 'cares_details' in result.details:
                    lines.append("#### CARES Breakdown")
                    for key, value in result.details['cares_details'].items():
                        lines.append(f"- {key.replace('_', ' ').title()}: {value:.3f}")
                    lines.append("")

                if 'rocket_details' in result.details:
                    lines.append("#### ROCKET Components")
                    for key, value in result.details['rocket_details'].items():
                        lines.append(f"- {key}: {value:.3f}")
                    lines.append("")

        # Methodology
        lines.append("## Methodology\n")
        lines.append("This evaluation uses the BioREASONC-Bench framework for assessing")
        lines.append("biomedical causal reasoning capabilities in language models.\n")
        lines.append("**Knowledge Bases Consulted:**")
        lines.append("- ClinVar (clinical variant significance)")
        lines.append("- Open Targets (GWAS associations)")
        lines.append("- STRING-DB (protein interactions)")
        lines.append("- Ensembl (gene annotations)")
        lines.append("- Enrichr (pathway enrichment)")
        lines.append("- EpiGraphDB (Mendelian randomization evidence)\n")

        # Footer
        lines.append("---")
        lines.append(f"*Report generated by BioREASONC-Bench v1.0*")

        content = "\n".join(lines)
        filepath = self.output_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath.write_text(content)

        return str(filepath)

    def _generate_html_report(
        self,
        results: Dict[str, ModelResult],
        benchmark_name: str,
        timestamp: str,
        include_details: bool
    ) -> str:
        """Generate HTML evaluation report."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{benchmark_name} Evaluation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        h3 {{
            color: #7f8c8d;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .score-high {{
            color: #27ae60;
            font-weight: bold;
        }}
        .score-medium {{
            color: #f39c12;
            font-weight: bold;
        }}
        .score-low {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .summary-box {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .metric-card {{
            display: inline-block;
            background: white;
            padding: 15px 25px;
            margin: 10px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }}
        .metric-label {{
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{benchmark_name} Evaluation Report</h1>
        <p><strong>Generated:</strong> {timestamp}</p>

        <div class="summary-box">
            <h2>Executive Summary</h2>
            <div class="metric-card">
                <div class="metric-value">{len(results)}</div>
                <div class="metric-label">Models Evaluated</div>
            </div>
"""

        best_model = max(results.keys(), key=lambda m: results[m].rocket_score)
        best_score = results[best_model].rocket_score
        avg_hallucination = sum(r.hallucination_rate for r in results.values()) / len(results)

        html += f"""
            <div class="metric-card">
                <div class="metric-value">{best_score:.3f}</div>
                <div class="metric-label">Best ROCKET Score ({best_model})</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{avg_hallucination:.1%}</div>
                <div class="metric-label">Avg Hallucination Rate</div>
            </div>
        </div>

        <h2>Overall Scores</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>GRASS</th>
                    <th>CARES</th>
                    <th>ROCKET</th>
                    <th>Hallucination Rate</th>
                </tr>
            </thead>
            <tbody>
"""

        for model_name, result in sorted(results.items(), key=lambda x: x[1].rocket_score, reverse=True):
            rocket_class = "score-high" if result.rocket_score >= 0.7 else ("score-medium" if result.rocket_score >= 0.5 else "score-low")
            hall_class = "score-high" if result.hallucination_rate <= 0.1 else ("score-medium" if result.hallucination_rate <= 0.3 else "score-low")

            html += f"""
                <tr>
                    <td><strong>{model_name}</strong></td>
                    <td>{result.grass_score:.3f}</td>
                    <td>{result.cares_score:.3f}</td>
                    <td class="{rocket_class}">{result.rocket_score:.3f}</td>
                    <td class="{hall_class}">{result.hallucination_rate:.1%}</td>
                </tr>
"""

        html += """
            </tbody>
        </table>
"""

        if include_details:
            html += """
        <h2>Detailed Results</h2>
"""
            for model_name, result in results.items():
                html += f"""
        <h3>{model_name}</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
"""
                for key, value in result.details.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            html += f"<tr><td>{key} - {sub_key}</td><td>{sub_value:.3f}</td></tr>"
                    elif isinstance(value, (int, float)):
                        html += f"<tr><td>{key}</td><td>{value:.3f}</td></tr>"

                html += "</table>"

        html += f"""
        <div class="footer">
            <p>Report generated by BioREASONC-Bench v1.0</p>
        </div>
    </div>
</body>
</html>
"""

        filepath = self.output_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath.write_text(html)

        return str(filepath)

    def _generate_json_report(
        self,
        results: Dict[str, ModelResult],
        benchmark_name: str,
        timestamp: str
    ) -> str:
        """Generate JSON evaluation report."""
        report = {
            "benchmark": benchmark_name,
            "timestamp": timestamp,
            "models_evaluated": len(results),
            "results": {}
        }

        for model_name, result in results.items():
            report["results"][model_name] = {
                "grass_score": result.grass_score,
                "cares_score": result.cares_score,
                "rocket_score": result.rocket_score,
                "hallucination_rate": result.hallucination_rate,
                "confidence": result.confidence,
                "details": result.details
            }

        # Summary statistics
        scores = [r.rocket_score for r in results.values()]
        report["summary"] = {
            "best_model": max(results.keys(), key=lambda m: results[m].rocket_score),
            "best_rocket_score": max(scores),
            "average_rocket_score": sum(scores) / len(scores),
            "average_hallucination_rate": sum(r.hallucination_rate for r in results.values()) / len(results)
        }

        filepath = self.output_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath.write_text(json.dumps(report, indent=2))

        return str(filepath)

    def generate_model_card(
        self,
        model_name: str,
        result: ModelResult,
        model_description: str = "",
        intended_use: str = "",
        limitations: str = ""
    ) -> str:
        """
        Generate a model card for a specific model.

        Args:
            model_name: Name of the model
            result: ModelResult with evaluation scores
            model_description: Description of the model
            intended_use: Intended use cases
            limitations: Known limitations

        Returns:
            Path to generated model card
        """
        lines = []

        lines.append(f"# Model Card: {model_name}")
        lines.append(f"\n**Evaluated on:** BioREASONC-Bench")
        lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n")

        if model_description:
            lines.append("## Model Description\n")
            lines.append(model_description + "\n")

        lines.append("## Biomedical Causal Reasoning Performance\n")
        lines.append("### Overall Scores\n")
        lines.append(f"| Metric | Score |")
        lines.append(f"|--------|-------|")
        lines.append(f"| GRASS Score | {result.grass_score:.3f} |")
        lines.append(f"| CARES Score | {result.cares_score:.3f} |")
        lines.append(f"| ROCKET Score | {result.rocket_score:.3f} |")
        lines.append(f"| Hallucination Rate | {result.hallucination_rate:.1%} |")
        lines.append(f"| Confidence | {result.confidence:.3f} |")

        # Performance interpretation
        lines.append("\n### Performance Interpretation\n")

        if result.rocket_score >= 0.8:
            lines.append("**Overall:** Excellent performance in biomedical causal reasoning.")
        elif result.rocket_score >= 0.6:
            lines.append("**Overall:** Good performance with room for improvement.")
        elif result.rocket_score >= 0.4:
            lines.append("**Overall:** Moderate performance; may not be suitable for critical applications.")
        else:
            lines.append("**Overall:** Below baseline; significant improvements needed.")

        if result.hallucination_rate <= 0.05:
            lines.append("\n**Reliability:** Very low hallucination rate; highly reliable outputs.")
        elif result.hallucination_rate <= 0.15:
            lines.append("\n**Reliability:** Acceptable hallucination rate; verify critical claims.")
        else:
            lines.append("\n**Reliability:** High hallucination rate; outputs require verification.")

        if intended_use:
            lines.append("\n## Intended Use\n")
            lines.append(intended_use + "\n")

        if limitations:
            lines.append("## Limitations\n")
            lines.append(limitations + "\n")

        lines.append("## Evaluation Details\n")
        lines.append("### GRASS (Gene Risk Aggregation)")
        if 'grass_details' in result.details:
            for key, value in result.details['grass_details'].items():
                lines.append(f"- {key.replace('_', ' ').title()}: {value:.3f}")

        lines.append("\n### CARES (Causal-Aware Reasoning)")
        if 'cares_details' in result.details:
            for key, value in result.details['cares_details'].items():
                lines.append(f"- {key.replace('_', ' ').title()}: {value:.3f}")

        lines.append("\n### ROCKET Components")
        if 'rocket_details' in result.details:
            for key, value in result.details['rocket_details'].items():
                lines.append(f"- {key}: {value:.3f}")

        lines.append("\n---")
        lines.append("*Generated by BioREASONC-Bench*")

        content = "\n".join(lines)
        filepath = self.output_dir / f"model_card_{model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d')}.md"
        filepath.write_text(content)

        return str(filepath)

    def generate_benchmark_summary(
        self,
        results: Dict[str, ModelResult],
        benchmark_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a concise benchmark summary.

        Args:
            results: Dict mapping model names to ModelResult objects
            benchmark_config: Optional benchmark configuration details

        Returns:
            Path to generated summary
        """
        lines = []

        lines.append("# BioREASONC-Bench Summary\n")
        lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

        # Leaderboard
        lines.append("## Leaderboard\n")
        sorted_results = sorted(results.items(), key=lambda x: x[1].rocket_score, reverse=True)

        lines.append("| Rank | Model | ROCKET | GRASS | CARES |")
        lines.append("|------|-------|--------|-------|-------|")

        for i, (model_name, result) in enumerate(sorted_results, 1):
            medal = ""
            if i == 1:
                medal = " :1st_place_medal:"
            elif i == 2:
                medal = " :2nd_place_medal:"
            elif i == 3:
                medal = " :3rd_place_medal:"

            lines.append(
                f"| {i} | {model_name}{medal} | **{result.rocket_score:.3f}** | "
                f"{result.grass_score:.3f} | {result.cares_score:.3f} |"
            )

        # Key findings
        lines.append("\n## Key Findings\n")

        best_model, best_result = sorted_results[0]
        lines.append(f"- **Best Overall:** {best_model} achieves ROCKET score of {best_result.rocket_score:.3f}")

        # Best per metric
        best_grass = max(results.items(), key=lambda x: x[1].grass_score)
        best_cares = max(results.items(), key=lambda x: x[1].cares_score)
        lowest_hall = min(results.items(), key=lambda x: x[1].hallucination_rate)

        lines.append(f"- **Best GRASS:** {best_grass[0]} ({best_grass[1].grass_score:.3f})")
        lines.append(f"- **Best CARES:** {best_cares[0]} ({best_cares[1].cares_score:.3f})")
        lines.append(f"- **Lowest Hallucination:** {lowest_hall[0]} ({lowest_hall[1].hallucination_rate:.1%})")

        if benchmark_config:
            lines.append("\n## Benchmark Configuration\n")
            for key, value in benchmark_config.items():
                lines.append(f"- **{key}:** {value}")

        lines.append("\n---")
        lines.append("*BioREASONC-Bench: Biomedical Reasoning and Causal Inference Benchmark*")

        content = "\n".join(lines)
        filepath = self.output_dir / f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath.write_text(content)

        return str(filepath)

    def export_results_table(
        self,
        results: Dict[str, ModelResult],
        format: str = "csv",
        include_details: bool = False
    ) -> str:
        """
        Export results to tabular format.

        Args:
            results: Dict mapping model names to ModelResult objects
            format: Output format ('csv', 'tsv', 'latex')
            include_details: Whether to include detailed metrics

        Returns:
            Path to exported file
        """
        if format == "csv":
            return self._export_csv(results, include_details)
        elif format == "tsv":
            return self._export_csv(results, include_details, delimiter='\t', extension='tsv')
        elif format == "latex":
            return self._export_latex(results)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _export_csv(
        self,
        results: Dict[str, ModelResult],
        include_details: bool,
        delimiter: str = ',',
        extension: str = 'csv'
    ) -> str:
        """Export results to CSV/TSV format."""
        filepath = self.output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{extension}"

        headers = ['Model', 'GRASS', 'CARES', 'ROCKET', 'Hallucination_Rate', 'Confidence']

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=delimiter)
            writer.writerow(headers)

            for model_name, result in sorted(results.items(), key=lambda x: x[1].rocket_score, reverse=True):
                row = [
                    model_name,
                    f"{result.grass_score:.4f}",
                    f"{result.cares_score:.4f}",
                    f"{result.rocket_score:.4f}",
                    f"{result.hallucination_rate:.4f}",
                    f"{result.confidence:.4f}"
                ]
                writer.writerow(row)

        return str(filepath)

    def _export_latex(self, results: Dict[str, ModelResult]) -> str:
        """Export results to LaTeX table format."""
        lines = []

        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{BioREASONC-Bench Evaluation Results}")
        lines.append(r"\label{tab:bioreasonc-results}")
        lines.append(r"\begin{tabular}{lcccc}")
        lines.append(r"\toprule")
        lines.append(r"Model & GRASS & CARES & ROCKET & Hallucination \\")
        lines.append(r"\midrule")

        for model_name, result in sorted(results.items(), key=lambda x: x[1].rocket_score, reverse=True):
            # Escape underscores in model name
            safe_name = model_name.replace('_', r'\_')
            lines.append(
                f"{safe_name} & {result.grass_score:.3f} & {result.cares_score:.3f} & "
                f"\\textbf{{{result.rocket_score:.3f}}} & {result.hallucination_rate:.1%} \\\\"
            )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        content = "\n".join(lines)
        filepath = self.output_dir / f"results_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex"
        filepath.write_text(content)

        return str(filepath)


# Convenience functions
def generate_evaluation_report(
    results: Dict[str, Any],
    output_dir: str = "reports",
    format: str = "markdown",
    **kwargs
) -> str:
    """Quick function to generate evaluation report."""
    generator = SummaryGenerator(output_dir=output_dir)

    # Convert dict results to ModelResult objects if needed
    model_results = {}
    for model_name, data in results.items():
        if isinstance(data, ModelResult):
            model_results[model_name] = data
        else:
            model_results[model_name] = ModelResult(
                model_name=model_name,
                grass_score=data.get('grass_score', 0),
                cares_score=data.get('cares_score', 0),
                rocket_score=data.get('rocket_score', 0),
                hallucination_rate=data.get('hallucination_rate', 0),
                confidence=data.get('confidence', 0),
                details=data.get('details', {})
            )

    return generator.generate_evaluation_report(model_results, format=format, **kwargs)


def generate_model_card(
    model_name: str,
    result: Union[ModelResult, Dict[str, Any]],
    output_dir: str = "reports",
    **kwargs
) -> str:
    """Quick function to generate model card."""
    generator = SummaryGenerator(output_dir=output_dir)

    if not isinstance(result, ModelResult):
        result = ModelResult(
            model_name=model_name,
            grass_score=result.get('grass_score', 0),
            cares_score=result.get('cares_score', 0),
            rocket_score=result.get('rocket_score', 0),
            hallucination_rate=result.get('hallucination_rate', 0),
            confidence=result.get('confidence', 0),
            details=result.get('details', {})
        )

    return generator.generate_model_card(model_name, result, **kwargs)


def generate_benchmark_summary(
    results: Dict[str, Any],
    output_dir: str = "reports",
    **kwargs
) -> str:
    """Quick function to generate benchmark summary."""
    generator = SummaryGenerator(output_dir=output_dir)

    model_results = {}
    for model_name, data in results.items():
        if isinstance(data, ModelResult):
            model_results[model_name] = data
        else:
            model_results[model_name] = ModelResult(
                model_name=model_name,
                grass_score=data.get('grass_score', 0),
                cares_score=data.get('cares_score', 0),
                rocket_score=data.get('rocket_score', 0),
                hallucination_rate=data.get('hallucination_rate', 0),
                confidence=data.get('confidence', 0),
                details=data.get('details', {})
            )

    return generator.generate_benchmark_summary(model_results, **kwargs)


def export_results_table(
    results: Dict[str, Any],
    output_dir: str = "reports",
    format: str = "csv",
    **kwargs
) -> str:
    """Quick function to export results table."""
    generator = SummaryGenerator(output_dir=output_dir)

    model_results = {}
    for model_name, data in results.items():
        if isinstance(data, ModelResult):
            model_results[model_name] = data
        else:
            model_results[model_name] = ModelResult(
                model_name=model_name,
                grass_score=data.get('grass_score', 0),
                cares_score=data.get('cares_score', 0),
                rocket_score=data.get('rocket_score', 0),
                hallucination_rate=data.get('hallucination_rate', 0),
                confidence=data.get('confidence', 0),
                details=data.get('details', {})
            )

    return generator.export_results_table(model_results, format=format, **kwargs)
