"""
BioREASONC-Bench Interactive Dashboard

Generates interactive HTML dashboards for exploring evaluation results.
Uses Plotly for interactive visualizations.
"""

import os
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class DashboardConfig:
    """Configuration for dashboard generation."""
    title: str = "BioREASONC-Bench Dashboard"
    theme: str = "plotly_white"
    width: int = 1400
    height: int = 800
    show_details: bool = True
    export_static: bool = False


class DashboardGenerator:
    """
    Generates interactive HTML dashboards for evaluation results.

    Uses Plotly for interactive charts when available,
    falls back to static HTML tables otherwise.
    """

    def __init__(self, config: Optional[DashboardConfig] = None, output_dir: str = "dashboards"):
        """
        Initialize the dashboard generator.

        Args:
            config: Dashboard configuration
            output_dir: Directory to save dashboards
        """
        self.config = config or DashboardConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._check_plotly()

    def _check_plotly(self):
        """Check if Plotly is available."""
        try:
            import plotly
            self.has_plotly = True
        except ImportError:
            self.has_plotly = False

    def create_interactive_dashboard(
        self,
        results: Dict[str, Dict[str, Any]],
        benchmark_name: str = "BioREASONC-Bench"
    ) -> str:
        """
        Create a comprehensive interactive dashboard.

        Args:
            results: Dict mapping model names to evaluation results
            benchmark_name: Name of the benchmark

        Returns:
            Path to generated dashboard HTML
        """
        if self.has_plotly:
            return self._create_plotly_dashboard(results, benchmark_name)
        else:
            return self._create_static_dashboard(results, benchmark_name)

    def _create_plotly_dashboard(
        self,
        results: Dict[str, Dict[str, Any]],
        benchmark_name: str
    ) -> str:
        """Create dashboard using Plotly."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.express as px

        models = list(results.keys())
        metrics = ['grass_score', 'cares_score', 'rocket_score']

        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Overall Scores Comparison',
                'ROCKET Score Ranking',
                'Hallucination vs Performance',
                'Score Distribution',
                'Metric Radar Chart',
                'Confidence vs Accuracy'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "box"}],
                [{"type": "scatterpolar"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        colors = px.colors.qualitative.Set2[:len(models)]

        # 1. Overall Scores Comparison (Grouped Bar)
        for i, metric in enumerate(metrics):
            values = [results[m].get(metric, 0) for m in models]
            fig.add_trace(
                go.Bar(
                    name=metric.replace('_', ' ').title(),
                    x=models,
                    y=values,
                    text=[f'{v:.3f}' for v in values],
                    textposition='outside'
                ),
                row=1, col=1
            )

        # 2. ROCKET Score Ranking (Horizontal Bar)
        sorted_models = sorted(models, key=lambda m: results[m].get('rocket_score', 0))
        rocket_scores = [results[m].get('rocket_score', 0) for m in sorted_models]

        fig.add_trace(
            go.Bar(
                x=rocket_scores,
                y=sorted_models,
                orientation='h',
                marker_color=colors[:len(sorted_models)],
                text=[f'{s:.3f}' for s in rocket_scores],
                textposition='outside',
                showlegend=False
            ),
            row=1, col=2
        )

        # 3. Hallucination vs Performance (Scatter)
        hall_rates = [results[m].get('hallucination_rate', 0) for m in models]
        rocket_scores_all = [results[m].get('rocket_score', 0) for m in models]

        fig.add_trace(
            go.Scatter(
                x=hall_rates,
                y=rocket_scores_all,
                mode='markers+text',
                text=models,
                textposition='top center',
                marker=dict(size=15, color=colors[:len(models)]),
                showlegend=False
            ),
            row=2, col=1
        )

        # 4. Score Distribution (Box Plot)
        for metric in metrics:
            values = [results[m].get(metric, 0) for m in models]
            fig.add_trace(
                go.Box(
                    y=values,
                    name=metric.replace('_', ' ').title(),
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8
                ),
                row=2, col=2
            )

        # 5. Radar Chart
        radar_metrics = ['grass_score', 'cares_score', 'rocket_score', 'confidence']
        radar_labels = ['GRASS', 'CARES', 'ROCKET', 'Confidence']

        for i, model in enumerate(models[:4]):  # Limit to 4 models for readability
            values = [results[model].get(m, 0) for m in radar_metrics]
            values.append(values[0])  # Close the polygon

            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=radar_labels + [radar_labels[0]],
                    fill='toself',
                    name=model,
                    opacity=0.6
                ),
                row=3, col=1
            )

        # 6. Confidence vs Accuracy
        confidences = [results[m].get('confidence', 0) for m in models]
        cares_scores = [results[m].get('cares_score', 0) for m in models]

        fig.add_trace(
            go.Scatter(
                x=confidences,
                y=cares_scores,
                mode='markers+text',
                text=models,
                textposition='top center',
                marker=dict(
                    size=12,
                    color=rocket_scores_all,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='ROCKET')
                ),
                showlegend=False
            ),
            row=3, col=2
        )

        # Update layout
        fig.update_layout(
            title=dict(
                text=f'{benchmark_name} Evaluation Dashboard',
                font=dict(size=24)
            ),
            height=1200,
            width=self.config.width,
            template=self.config.theme,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Update axes labels
        fig.update_xaxes(title_text="Model", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=1, col=1)
        fig.update_xaxes(title_text="ROCKET Score", row=1, col=2)
        fig.update_xaxes(title_text="Hallucination Rate", row=2, col=1)
        fig.update_yaxes(title_text="ROCKET Score", row=2, col=1)
        fig.update_xaxes(title_text="Confidence", row=3, col=2)
        fig.update_yaxes(title_text="CARES Score", row=3, col=2)

        # Save to HTML
        filepath = self.output_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(str(filepath), include_plotlyjs='cdn')

        return str(filepath)

    def _create_static_dashboard(
        self,
        results: Dict[str, Dict[str, Any]],
        benchmark_name: str
    ) -> str:
        """Create static HTML dashboard without Plotly."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{benchmark_name} Dashboard</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .dashboard {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            background: white;
            padding: 30px;
            border-radius: 16px;
            margin-bottom: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            color: #333;
            font-size: 28px;
            margin-bottom: 10px;
        }}
        .header p {{
            color: #666;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .card {{
            background: white;
            padding: 25px;
            border-radius: 16px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            color: #333;
            font-size: 18px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }}
        .metric-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid #f0f0f0;
        }}
        .metric-row:last-child {{
            border-bottom: none;
        }}
        .metric-name {{
            font-weight: 500;
            color: #333;
        }}
        .metric-value {{
            font-size: 18px;
            font-weight: bold;
        }}
        .metric-value.high {{
            color: #27ae60;
        }}
        .metric-value.medium {{
            color: #f39c12;
        }}
        .metric-value.low {{
            color: #e74c3c;
        }}
        .bar-chart {{
            margin: 10px 0;
        }}
        .bar-row {{
            margin: 8px 0;
        }}
        .bar-label {{
            font-size: 14px;
            margin-bottom: 4px;
            display: flex;
            justify-content: space-between;
        }}
        .bar-container {{
            background: #e0e0e0;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
        }}
        .bar-fill {{
            height: 100%;
            border-radius: 10px;
            transition: width 0.5s ease;
        }}
        .bar-fill.grass {{
            background: linear-gradient(90deg, #43e97b 0%, #38f9d7 100%);
        }}
        .bar-fill.cares {{
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        }}
        .bar-fill.rocket {{
            background: linear-gradient(90deg, #fa709a 0%, #fee140 100%);
        }}
        .leaderboard {{
            width: 100%;
        }}
        .leaderboard-item {{
            display: flex;
            align-items: center;
            padding: 15px;
            margin: 10px 0;
            background: #f8f9fa;
            border-radius: 12px;
            transition: transform 0.2s;
        }}
        .leaderboard-item:hover {{
            transform: translateX(5px);
        }}
        .rank {{
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-right: 15px;
        }}
        .rank.gold {{
            background: linear-gradient(135deg, #ffd700 0%, #ffed4a 100%);
            color: #8b7500;
        }}
        .rank.silver {{
            background: linear-gradient(135deg, #c0c0c0 0%, #e8e8e8 100%);
            color: #555;
        }}
        .rank.bronze {{
            background: linear-gradient(135deg, #cd7f32 0%, #daa06d 100%);
            color: #5c3317;
        }}
        .rank.default {{
            background: #e0e0e0;
            color: #666;
        }}
        .model-name {{
            flex: 1;
            font-weight: 500;
        }}
        .model-score {{
            font-size: 20px;
            font-weight: bold;
            color: #667eea;
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
        }}
        .stat-box {{
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 12px;
        }}
        .stat-value {{
            font-size: 28px;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            margin-top: 5px;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: white;
            opacity: 0.8;
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>{benchmark_name} Dashboard</h1>
            <p>Generated: {timestamp}</p>
        </div>
"""

        # Summary statistics
        models = list(results.keys())
        best_model = max(models, key=lambda m: results[m].get('rocket_score', 0))
        avg_rocket = sum(results[m].get('rocket_score', 0) for m in models) / len(models)
        avg_hallucination = sum(results[m].get('hallucination_rate', 0) for m in models) / len(models)

        html += f"""
        <div class="card">
            <h2>Summary Statistics</h2>
            <div class="summary-stats">
                <div class="stat-box">
                    <div class="stat-value">{len(models)}</div>
                    <div class="stat-label">Models Evaluated</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{results[best_model].get('rocket_score', 0):.3f}</div>
                    <div class="stat-label">Best ROCKET Score</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{avg_rocket:.3f}</div>
                    <div class="stat-label">Average ROCKET</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{avg_hallucination:.1%}</div>
                    <div class="stat-label">Avg Hallucination</div>
                </div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h2>Leaderboard</h2>
                <div class="leaderboard">
"""

        # Leaderboard
        sorted_models = sorted(models, key=lambda m: results[m].get('rocket_score', 0), reverse=True)
        for i, model in enumerate(sorted_models):
            rank_class = ['gold', 'silver', 'bronze'][i] if i < 3 else 'default'
            score = results[model].get('rocket_score', 0)
            html += f"""
                    <div class="leaderboard-item">
                        <div class="rank {rank_class}">{i + 1}</div>
                        <div class="model-name">{model}</div>
                        <div class="model-score">{score:.3f}</div>
                    </div>
"""

        html += """
                </div>
            </div>

            <div class="card">
                <h2>Score Distribution</h2>
                <div class="bar-chart">
"""

        # Bar charts for each model
        for model in sorted_models:
            grass = results[model].get('grass_score', 0)
            cares = results[model].get('cares_score', 0)
            rocket = results[model].get('rocket_score', 0)

            html += f"""
                    <div class="bar-row">
                        <div class="bar-label">
                            <span>{model}</span>
                        </div>
                        <div class="bar-container">
                            <div class="bar-fill grass" style="width: {grass * 100}%"></div>
                        </div>
                        <div class="bar-container" style="margin-top: 4px">
                            <div class="bar-fill cares" style="width: {cares * 100}%"></div>
                        </div>
                        <div class="bar-container" style="margin-top: 4px">
                            <div class="bar-fill rocket" style="width: {rocket * 100}%"></div>
                        </div>
                    </div>
"""

        html += """
                </div>
                <div style="margin-top: 15px; font-size: 12px; color: #666;">
                    <span style="color: #43e97b;">&#9632;</span> GRASS
                    <span style="color: #4facfe; margin-left: 15px;">&#9632;</span> CARES
                    <span style="color: #fa709a; margin-left: 15px;">&#9632;</span> ROCKET
                </div>
            </div>
        </div>

        <div class="grid">
"""

        # Individual model cards
        for model in sorted_models[:4]:  # Show top 4
            r = results[model]
            rocket_class = 'high' if r.get('rocket_score', 0) >= 0.7 else ('medium' if r.get('rocket_score', 0) >= 0.5 else 'low')
            hall_class = 'high' if r.get('hallucination_rate', 0) <= 0.1 else ('medium' if r.get('hallucination_rate', 0) <= 0.3 else 'low')

            html += f"""
            <div class="card">
                <h2>{model}</h2>
                <div class="metric-row">
                    <span class="metric-name">GRASS Score</span>
                    <span class="metric-value">{r.get('grass_score', 0):.3f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-name">CARES Score</span>
                    <span class="metric-value">{r.get('cares_score', 0):.3f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-name">ROCKET Score</span>
                    <span class="metric-value {rocket_class}">{r.get('rocket_score', 0):.3f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-name">Hallucination Rate</span>
                    <span class="metric-value {hall_class}">{r.get('hallucination_rate', 0):.1%}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-name">Confidence</span>
                    <span class="metric-value">{r.get('confidence', 0):.3f}</span>
                </div>
            </div>
"""

        html += """
        </div>

        <div class="footer">
            <p>BioREASONC-Bench: Biomedical Reasoning and Causal Inference Benchmark</p>
        </div>
    </div>
</body>
</html>
"""

        filepath = self.output_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath.write_text(html)

        return str(filepath)

    def create_comparison_view(
        self,
        results: Dict[str, Dict[str, Any]],
        models_to_compare: Optional[List[str]] = None,
        metrics_to_compare: Optional[List[str]] = None
    ) -> str:
        """
        Create a side-by-side comparison view for selected models.

        Args:
            results: Dict mapping model names to evaluation results
            models_to_compare: List of models to include (default: all)
            metrics_to_compare: List of metrics to compare

        Returns:
            Path to generated comparison HTML
        """
        if models_to_compare is None:
            models_to_compare = list(results.keys())

        if metrics_to_compare is None:
            metrics_to_compare = ['grass_score', 'cares_score', 'rocket_score',
                                  'hallucination_rate', 'confidence']

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Comparison</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f7fa;
            padding: 40px;
            margin: 0;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: #2d3748;
            margin-bottom: 10px;
        }}
        .timestamp {{
            color: #718096;
            margin-bottom: 30px;
        }}
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }}
        .comparison-table th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: left;
            font-weight: 600;
        }}
        .comparison-table td {{
            padding: 16px 20px;
            border-bottom: 1px solid #edf2f7;
        }}
        .comparison-table tr:last-child td {{
            border-bottom: none;
        }}
        .comparison-table tr:hover td {{
            background: #f7fafc;
        }}
        .metric-name {{
            font-weight: 500;
            color: #4a5568;
        }}
        .score {{
            font-weight: bold;
            font-size: 16px;
        }}
        .best {{
            color: #38a169;
            background: #f0fff4;
            padding: 4px 12px;
            border-radius: 20px;
        }}
        .worst {{
            color: #e53e3e;
        }}
        .winner-badge {{
            display: inline-block;
            background: linear-gradient(135deg, #ffd700 0%, #ffed4a 100%);
            color: #8b7500;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            margin-left: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Model Comparison</h1>
        <p class="timestamp">Generated: {timestamp}</p>

        <table class="comparison-table">
            <thead>
                <tr>
                    <th>Metric</th>
"""

        for model in models_to_compare:
            html += f"                    <th>{model}</th>\n"

        html += """                </tr>
            </thead>
            <tbody>
"""

        # For each metric, find best and worst
        for metric in metrics_to_compare:
            values = {m: results[m].get(metric, 0) for m in models_to_compare}

            # For hallucination_rate, lower is better
            if 'hallucination' in metric.lower():
                best_model = min(values, key=values.get)
            else:
                best_model = max(values, key=values.get)

            metric_display = metric.replace('_', ' ').title()

            html += f"""                <tr>
                    <td class="metric-name">{metric_display}</td>
"""

            for model in models_to_compare:
                value = values[model]
                is_best = model == best_model

                if 'rate' in metric:
                    display_value = f"{value:.1%}"
                else:
                    display_value = f"{value:.3f}"

                if is_best:
                    html += f'                    <td><span class="score best">{display_value}</span><span class="winner-badge">Best</span></td>\n'
                else:
                    html += f'                    <td><span class="score">{display_value}</span></td>\n'

            html += "                </tr>\n"

        html += """            </tbody>
        </table>
    </div>
</body>
</html>
"""

        filepath = self.output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath.write_text(html)

        return str(filepath)


# Convenience functions
def create_interactive_dashboard(
    results: Dict[str, Dict[str, Any]],
    output_dir: str = "dashboards",
    **kwargs
) -> str:
    """Quick function to create interactive dashboard."""
    generator = DashboardGenerator(output_dir=output_dir)
    return generator.create_interactive_dashboard(results, **kwargs)


def create_comparison_view(
    results: Dict[str, Dict[str, Any]],
    output_dir: str = "dashboards",
    **kwargs
) -> str:
    """Quick function to create comparison view."""
    generator = DashboardGenerator(output_dir=output_dir)
    return generator.create_comparison_view(results, **kwargs)
