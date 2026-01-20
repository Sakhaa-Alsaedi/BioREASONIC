"""
Tests for the Enrichment Visualization Module.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional

# Import the visualizer
from enrichment.enrichment_visualizer import (
    EnrichmentVisualizer,
    VisualizerConfig,
)
from enrichment.enrichment_stats import EnrichmentScore


# Test fixtures
@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_enrichment_scores():
    """Create sample enrichment scores for testing."""
    return [
        EnrichmentScore(
            term="GO:0001234 - mitochondrial electron transport",
            p_value=0.0001,
            adjusted_p_value=0.001,
            fold_enrichment=5.2,
            combined_score=45.8,
            overlap_count=15,
            query_size=100,
            term_size=200,
            background_size=20000
        ),
        EnrichmentScore(
            term="GO:0002345 - cellular respiration",
            p_value=0.0005,
            adjusted_p_value=0.003,
            fold_enrichment=4.1,
            combined_score=32.5,
            overlap_count=12,
            query_size=100,
            term_size=180,
            background_size=20000
        ),
        EnrichmentScore(
            term="GO:0003456 - oxidative phosphorylation",
            p_value=0.001,
            adjusted_p_value=0.005,
            fold_enrichment=3.8,
            combined_score=28.3,
            overlap_count=10,
            query_size=100,
            term_size=150,
            background_size=20000
        ),
        EnrichmentScore(
            term="GO:0004567 - ATP synthesis coupled electron transport",
            p_value=0.002,
            adjusted_p_value=0.008,
            fold_enrichment=3.5,
            combined_score=22.1,
            overlap_count=8,
            query_size=100,
            term_size=120,
            background_size=20000
        ),
        EnrichmentScore(
            term="GO:0005678 - NADH dehydrogenase activity",
            p_value=0.005,
            adjusted_p_value=0.02,
            fold_enrichment=2.9,
            combined_score=15.6,
            overlap_count=6,
            query_size=100,
            term_size=100,
            background_size=20000
        ),
    ]


@pytest.fixture
def sample_gene_scores():
    """Create sample gene scores DataFrame."""
    np.random.seed(42)
    genes = ['NDUFS3', 'CELF1', 'MADD', 'PPP5C', 'MTCH2', 'JAZF1', 'ERCC1', 'HERC1']

    return pd.DataFrame({
        'target': genes,
        'combined_risk_score': np.random.uniform(0.3, 0.9, len(genes)),
        'combined_causal_score': np.random.uniform(0.2, 0.8, len(genes)),
        'MR_score': np.random.uniform(0.1, 0.7, len(genes)),
        'evidence_score': np.random.uniform(0.2, 0.9, len(genes)),
        'causal_confidence_score': np.random.uniform(0.1, 0.6, len(genes)),
    })


@pytest.fixture
def sample_network():
    """Create sample NetworkX graph."""
    try:
        import networkx as nx
        G = nx.Graph()
        genes = ['NDUFS3', 'CELF1', 'MADD', 'PPP5C', 'MTCH2']
        for gene in genes:
            G.add_node(gene)

        # Add edges
        G.add_edge('NDUFS3', 'CELF1', score=0.8)
        G.add_edge('NDUFS3', 'MADD', score=0.6)
        G.add_edge('CELF1', 'PPP5C', score=0.7)
        G.add_edge('PPP5C', 'MTCH2', score=0.5)
        G.add_edge('MADD', 'MTCH2', score=0.4)

        return G
    except ImportError:
        return None


class TestVisualizerConfig:
    """Test VisualizerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VisualizerConfig()
        assert config.figsize == (10, 8)
        assert config.dpi == 150
        assert config.style == "whitegrid"
        assert config.max_terms == 20
        assert config.fdr_threshold == 0.05

    def test_custom_config(self):
        """Test custom configuration."""
        config = VisualizerConfig(
            figsize=(12, 10),
            dpi=300,
            max_terms=30
        )
        assert config.figsize == (12, 10)
        assert config.dpi == 300
        assert config.max_terms == 30


class TestEnrichmentVisualizer:
    """Test EnrichmentVisualizer class."""

    def test_init(self, temp_output_dir):
        """Test visualizer initialization."""
        viz = EnrichmentVisualizer(output_dir=temp_output_dir)
        assert viz.output_dir == temp_output_dir
        assert viz.config is not None
        assert os.path.exists(temp_output_dir)

    def test_init_with_config(self, temp_output_dir):
        """Test initialization with custom config."""
        config = VisualizerConfig(dpi=300)
        viz = EnrichmentVisualizer(output_dir=temp_output_dir, config=config)
        assert viz.config.dpi == 300


class TestEnrichmentBarPlot:
    """Test enrichment bar plot."""

    def test_enrichment_bar_basic(self, temp_output_dir, sample_enrichment_scores):
        """Test basic enrichment bar plot."""
        viz = EnrichmentVisualizer(output_dir=temp_output_dir)
        path = viz.plot_enrichment_bar(sample_enrichment_scores, "GO_BP")

        assert path != ""
        assert os.path.exists(path)
        assert "go_bp_bar" in path.lower()

    def test_enrichment_bar_custom_title(self, temp_output_dir, sample_enrichment_scores):
        """Test bar plot with custom title."""
        viz = EnrichmentVisualizer(output_dir=temp_output_dir)
        path = viz.plot_enrichment_bar(
            sample_enrichment_scores,
            "KEGG",
            title="Custom KEGG Enrichment"
        )
        assert os.path.exists(path)

    def test_enrichment_bar_top_n(self, temp_output_dir, sample_enrichment_scores):
        """Test bar plot with top_n limit."""
        viz = EnrichmentVisualizer(output_dir=temp_output_dir)
        path = viz.plot_enrichment_bar(sample_enrichment_scores, "GO_BP", top_n=3)
        assert os.path.exists(path)

    def test_enrichment_bar_empty(self, temp_output_dir):
        """Test bar plot with empty scores."""
        viz = EnrichmentVisualizer(output_dir=temp_output_dir)
        path = viz.plot_enrichment_bar([], "GO_BP")
        assert path == ""


class TestEnrichmentDotPlot:
    """Test enrichment dot plot."""

    def test_enrichment_dot_basic(self, temp_output_dir, sample_enrichment_scores):
        """Test basic enrichment dot plot."""
        viz = EnrichmentVisualizer(output_dir=temp_output_dir)
        path = viz.plot_enrichment_dot(sample_enrichment_scores, "GO_BP")

        assert path != ""
        assert os.path.exists(path)
        assert "dot" in path.lower()

    def test_enrichment_dot_empty(self, temp_output_dir):
        """Test dot plot with empty scores."""
        viz = EnrichmentVisualizer(output_dir=temp_output_dir)
        path = viz.plot_enrichment_dot([], "GO_BP")
        assert path == ""


class TestRadarPlot:
    """Test radar/spider plot."""

    def test_radar_basic(self, temp_output_dir, sample_gene_scores):
        """Test basic radar plot."""
        viz = EnrichmentVisualizer(output_dir=temp_output_dir)
        genes = ['NDUFS3', 'CELF1', 'MADD']
        path = viz.plot_radar_scores(sample_gene_scores, genes)

        assert path != ""
        assert os.path.exists(path)
        assert "radar" in path.lower()

    def test_radar_empty_df(self, temp_output_dir):
        """Test radar with empty DataFrame."""
        viz = EnrichmentVisualizer(output_dir=temp_output_dir)
        path = viz.plot_radar_scores(pd.DataFrame(), ['NDUFS3'])
        assert path == ""

    def test_radar_missing_genes(self, temp_output_dir, sample_gene_scores):
        """Test radar with non-existent genes."""
        viz = EnrichmentVisualizer(output_dir=temp_output_dir)
        path = viz.plot_radar_scores(sample_gene_scores, ['NONEXISTENT1', 'NONEXISTENT2'])
        assert path == ""


class TestVolcanoPlot:
    """Test volcano plot."""

    def test_volcano_basic(self, temp_output_dir, sample_gene_scores):
        """Test basic volcano plot."""
        viz = EnrichmentVisualizer(output_dir=temp_output_dir)
        path = viz.plot_volcano(
            sample_gene_scores,
            x_col="causal_confidence_score",
            y_col="combined_risk_score"
        )

        assert path != ""
        assert os.path.exists(path)
        assert "volcano" in path.lower()

    def test_volcano_empty(self, temp_output_dir):
        """Test volcano with empty DataFrame."""
        viz = EnrichmentVisualizer(output_dir=temp_output_dir)
        path = viz.plot_volcano(pd.DataFrame())
        assert path == ""


class TestScoreDistribution:
    """Test score distribution plot."""

    def test_distribution_basic(self, temp_output_dir, sample_gene_scores):
        """Test basic distribution plot."""
        viz = EnrichmentVisualizer(output_dir=temp_output_dir)
        path = viz.plot_score_distribution(sample_gene_scores)

        assert path != ""
        assert os.path.exists(path)
        assert "distribution" in path.lower()


class TestNetworkPlot:
    """Test PPI network plot."""

    def test_network_basic(self, temp_output_dir, sample_network):
        """Test basic network plot."""
        if sample_network is None:
            pytest.skip("networkx not available")

        viz = EnrichmentVisualizer(output_dir=temp_output_dir)
        path = viz.plot_ppi_network(sample_network, hub_genes=['NDUFS3'])

        assert path != ""
        assert os.path.exists(path)
        assert "network" in path.lower()

    def test_network_empty(self, temp_output_dir):
        """Test network with None."""
        viz = EnrichmentVisualizer(output_dir=temp_output_dir)
        path = viz.plot_ppi_network(None)
        assert path == ""


class TestVennDiagram:
    """Test Venn diagram."""

    def test_venn_two_sets(self, temp_output_dir):
        """Test Venn with two sets."""
        viz = EnrichmentVisualizer(output_dir=temp_output_dir)
        sets = {
            "AD": {"NDUFS3", "CELF1", "MADD", "PPP5C"},
            "T2D": {"MADD", "PPP5C", "JAZF1", "MTCH2"}
        }
        path = viz.plot_venn(sets)

        assert path != ""
        assert os.path.exists(path)
        assert "venn" in path.lower()

    def test_venn_three_sets(self, temp_output_dir):
        """Test Venn with three sets."""
        viz = EnrichmentVisualizer(output_dir=temp_output_dir)
        sets = {
            "AD": {"NDUFS3", "CELF1", "MADD"},
            "T2D": {"MADD", "PPP5C", "JAZF1"},
            "PD": {"CELF1", "PPP5C", "HERC1"}
        }
        path = viz.plot_venn(sets)
        assert os.path.exists(path)

    def test_venn_insufficient_sets(self, temp_output_dir):
        """Test Venn with single set."""
        viz = EnrichmentVisualizer(output_dir=temp_output_dir)
        sets = {"AD": {"NDUFS3", "CELF1"}}
        path = viz.plot_venn(sets)
        assert path == ""


class TestManhattanPlot:
    """Test Manhattan-style plot."""

    def test_manhattan_basic(self, temp_output_dir, sample_gene_scores):
        """Test basic Manhattan plot."""
        viz = EnrichmentVisualizer(output_dir=temp_output_dir)
        path = viz.plot_gene_manhattan(sample_gene_scores)

        assert path != ""
        assert os.path.exists(path)
        assert "manhattan" in path.lower()

    def test_manhattan_with_threshold(self, temp_output_dir, sample_gene_scores):
        """Test Manhattan with threshold line."""
        viz = EnrichmentVisualizer(output_dir=temp_output_dir)
        path = viz.plot_gene_manhattan(
            sample_gene_scores,
            threshold=0.5,
            highlight_genes=['NDUFS3']
        )
        assert os.path.exists(path)


class TestSummaryDashboard:
    """Test summary dashboard."""

    @pytest.fixture
    def mock_pipeline_results(self, sample_enrichment_scores, sample_gene_scores, sample_network):
        """Create mock pipeline results."""
        # Import necessary types
        from enrichment.enrichment_pipeline import (
            PipelineResults,
            SharedGeneAnalysis,
            EnrichmentResults,
            DrugRepurposingResults,
        )
        from enrichment.disease_similarity import DiseaseSimilarity
        from enrichment.open_targets_client import DrugInfo

        shared_analysis = SharedGeneAnalysis(
            common_genes=['MADD', 'PPP5C', 'NDUFS3'],
            disease_specific_genes={
                'Alzheimer Disease': ['CELF1', 'ERCC1'],
                'Type 2 Diabetes': ['JAZF1', 'MTCH2']
            },
            hub_genes=['NDUFS3', 'MADD'],
            bridging_genes=[],
            gene_scores=sample_gene_scores
        )

        enrichment = EnrichmentResults(
            go_bp=sample_enrichment_scores,
            go_mf=sample_enrichment_scores[:3],
            go_cc=sample_enrichment_scores[:2],
            kegg=sample_enrichment_scores[:3],
            reactome=sample_enrichment_scores[:2],
            disease=[],
            gwas=[]
        )

        disease_sim = DiseaseSimilarity(
            disease_a="Alzheimer Disease",
            disease_b="Diabetes Mellitus, Type 2",
            jaccard_genes=0.15,
            jaccard_snps=0.05,
            pathway_overlap=0.1,
            network_proximity=0.2,
            combined_score=0.1,
            shared_genes=['MADD', 'PPP5C', 'NDUFS3']
        )

        drug_results = DrugRepurposingResults(
            ad_drugs=[
                DrugInfo(
                    drug_id="CHEMBL502",
                    drug_name="DONEPEZIL",
                    drug_type="Small molecule",
                    mechanism_of_action="AChE inhibitor",
                    max_clinical_phase=4,
                    target_genes=["ACHE"],
                    is_approved=True
                )
            ],
            t2d_drugs=[
                DrugInfo(
                    drug_id="CHEMBL1431",
                    drug_name="METFORMIN",
                    drug_type="Small molecule",
                    mechanism_of_action="AMPK activator",
                    max_clinical_phase=4,
                    target_genes=["PRKAB1"],
                    is_approved=True
                )
            ],
            shared_targets=[],
            repurposing_candidates=[]
        )

        return PipelineResults(
            diseases=["Alzheimer Disease", "Diabetes Mellitus, Type 2"],
            total_genes=1853,
            total_edges=153611,
            shared_analysis=shared_analysis,
            ppi_network=sample_network,
            network_stats={'nodes': 5, 'edges': 5, 'density': 0.5},
            enrichment=enrichment,
            disease_similarity=disease_sim,
            drug_results=drug_results
        )

    def test_dashboard_basic(self, temp_output_dir, mock_pipeline_results):
        """Test basic dashboard creation."""
        viz = EnrichmentVisualizer(output_dir=temp_output_dir)
        path = viz.create_summary_dashboard(mock_pipeline_results)

        assert path != ""
        assert os.path.exists(path)
        assert "dashboard" in path.lower()

    def test_dashboard_none_results(self, temp_output_dir):
        """Test dashboard with None results."""
        viz = EnrichmentVisualizer(output_dir=temp_output_dir)
        path = viz.create_summary_dashboard(None)
        assert path == ""


class TestExportAllPlots:
    """Test export_all_plots functionality."""

    @pytest.fixture
    def simple_pipeline_results(self, sample_enrichment_scores, sample_gene_scores):
        """Create simple pipeline results for export test."""
        from enrichment.enrichment_pipeline import (
            PipelineResults,
            SharedGeneAnalysis,
            EnrichmentResults,
        )

        shared_analysis = SharedGeneAnalysis(
            common_genes=['MADD', 'PPP5C'],
            disease_specific_genes={
                'Alzheimer Disease': ['CELF1'],
                'Type 2 Diabetes': ['JAZF1']
            },
            hub_genes=['MADD'],
            bridging_genes=[],
            gene_scores=sample_gene_scores
        )

        enrichment = EnrichmentResults(
            go_bp=sample_enrichment_scores[:3],
            go_mf=[],
            go_cc=[],
            kegg=sample_enrichment_scores[:2],
            reactome=[],
            disease=[],
            gwas=[]
        )

        return PipelineResults(
            diseases=["AD", "T2D"],
            total_genes=100,
            total_edges=500,
            shared_analysis=shared_analysis,
            ppi_network=None,
            network_stats={},
            enrichment=enrichment,
            disease_similarity=None,
            drug_results=None
        )

    def test_export_all(self, temp_output_dir, simple_pipeline_results):
        """Test exporting all plots."""
        viz = EnrichmentVisualizer(output_dir=temp_output_dir)
        paths = viz.export_all_plots(simple_pipeline_results)

        assert len(paths) > 0
        for path in paths:
            assert os.path.exists(path)

    def test_export_none_results(self, temp_output_dir):
        """Test export with None results."""
        viz = EnrichmentVisualizer(output_dir=temp_output_dir)
        paths = viz.export_all_plots(None)
        assert paths == []


class TestSimilarityPlots:
    """Test disease similarity plots."""

    def test_similarity_chord(self, temp_output_dir):
        """Test similarity chord diagram."""
        from enrichment.disease_similarity import DiseaseSimilarity

        disease_sim = DiseaseSimilarity(
            disease_a="Alzheimer Disease",
            disease_b="Diabetes Mellitus, Type 2",
            jaccard_genes=0.15,
            jaccard_snps=0.05,
            pathway_overlap=0.1,
            network_proximity=0.2,
            combined_score=0.1,
            shared_genes=['MADD', 'PPP5C', 'NDUFS3']
        )

        viz = EnrichmentVisualizer(output_dir=temp_output_dir)
        path = viz.plot_similarity_chord(disease_sim, ['MADD', 'PPP5C', 'NDUFS3'])

        assert path != ""
        assert os.path.exists(path)

    def test_similarity_heatmap(self, temp_output_dir):
        """Test similarity heatmap."""
        viz = EnrichmentVisualizer(output_dir=temp_output_dir)
        similarity_data = [
            {'disease_a': 'AD', 'disease_b': 'T2D', 'similarity': 0.15},
            {'disease_a': 'AD', 'disease_b': 'PD', 'similarity': 0.08},
            {'disease_a': 'T2D', 'disease_b': 'PD', 'similarity': 0.05},
        ]
        path = viz.plot_similarity_heatmap(similarity_data)

        assert path != ""
        assert os.path.exists(path)


class TestDrugPlots:
    """Test drug repurposing plots."""

    @pytest.fixture
    def sample_drug_results(self):
        """Create sample drug results."""
        from enrichment.enrichment_pipeline import DrugRepurposingResults
        from enrichment.open_targets_client import DrugInfo

        return DrugRepurposingResults(
            ad_drugs=[
                DrugInfo(
                    drug_id="CHEMBL502",
                    drug_name="DONEPEZIL",
                    drug_type="Small molecule",
                    mechanism_of_action="AChE inhibitor",
                    max_clinical_phase=4,
                    target_genes=["ACHE"],
                    is_approved=True
                ),
                DrugInfo(
                    drug_id="CHEMBL807",
                    drug_name="MEMANTINE",
                    drug_type="Small molecule",
                    mechanism_of_action="NMDA antagonist",
                    max_clinical_phase=4,
                    target_genes=["GRIN1", "GRIN2A"],
                    is_approved=True
                )
            ],
            t2d_drugs=[
                DrugInfo(
                    drug_id="CHEMBL1431",
                    drug_name="METFORMIN",
                    drug_type="Small molecule",
                    mechanism_of_action="AMPK activator",
                    max_clinical_phase=4,
                    target_genes=["PRKAB1"],
                    is_approved=True
                )
            ],
            shared_targets=[{'gene': 'ACHE', 'drugs': ['DONEPEZIL'], 'drug_count': 1}],
            repurposing_candidates=[]
        )

    def test_drug_target_network(self, temp_output_dir, sample_drug_results):
        """Test drug-target network plot."""
        viz = EnrichmentVisualizer(output_dir=temp_output_dir)
        path = viz.plot_drug_target_network(sample_drug_results)

        assert path != ""
        assert os.path.exists(path)

    def test_clinical_phase_bar(self, temp_output_dir, sample_drug_results):
        """Test clinical phase bar plot."""
        viz = EnrichmentVisualizer(output_dir=temp_output_dir)
        path = viz.plot_clinical_phase_bar(sample_drug_results)

        assert path != ""
        assert os.path.exists(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
