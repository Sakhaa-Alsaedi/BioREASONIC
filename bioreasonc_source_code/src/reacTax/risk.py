"""
Risk-Aware Reasoning Module (R)

Implements genetic risk scoring and statistical analysis:
- GWAS beta/OR interpretation
- Risk stratification
- Cumulative/aggregate risk scores
- Risk comparison and ranking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats

from ..schema import RiskGene, BenchmarkItem, RiskLabel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Risk level categories"""
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    PROTECTIVE = "protective"
    NEUTRAL = "neutral"


@dataclass
class RiskScore:
    """Represents a genetic risk score"""
    gene: str
    score: float
    score_type: str  # 'or', 'beta', 'aggregate'
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    risk_level: Optional[RiskLevel] = None


@dataclass
class AggregateRiskScore:
    """Represents an aggregate/cumulative risk score"""
    genes: List[str]
    score: float
    method: str  # 'sum', 'weighted', 'product'
    weights: Optional[List[float]] = None
    individual_scores: Optional[List[float]] = None


class RiskClassifier:
    """Classifies genetic risk based on OR/beta values"""

    def __init__(self,
                 high_or_threshold: float = 1.5,
                 moderate_or_threshold: float = 1.2,
                 protective_or_threshold: float = 0.8,
                 significance_threshold: float = 0.05):
        self.high_or = high_or_threshold
        self.moderate_or = moderate_or_threshold
        self.protective_or = protective_or_threshold
        self.significance = significance_threshold

    def classify_or(self, odds_ratio: float,
                    p_value: Optional[float] = None) -> RiskLevel:
        """Classify risk level based on odds ratio"""
        # Check significance
        if p_value is not None and p_value > self.significance:
            return RiskLevel.NEUTRAL

        if odds_ratio >= self.high_or:
            return RiskLevel.HIGH
        elif odds_ratio >= self.moderate_or:
            return RiskLevel.MODERATE
        elif odds_ratio <= self.protective_or:
            return RiskLevel.PROTECTIVE
        else:
            return RiskLevel.LOW

    def classify_beta(self, beta: float,
                      p_value: Optional[float] = None) -> RiskLevel:
        """Classify risk level based on beta coefficient"""
        # Convert beta to approximate OR
        approx_or = np.exp(beta)
        return self.classify_or(approx_or, p_value)

    def get_risk_interpretation(self, odds_ratio: float,
                                 gene: str = "gene") -> str:
        """Generate human-readable risk interpretation"""
        risk_level = self.classify_or(odds_ratio)

        interpretations = {
            RiskLevel.HIGH: f"{gene} confers HIGH risk (OR={odds_ratio:.2f}). Individuals with risk alleles have {odds_ratio:.1f}x increased odds of disease.",
            RiskLevel.MODERATE: f"{gene} confers MODERATE risk (OR={odds_ratio:.2f}). Risk alleles moderately increase disease odds.",
            RiskLevel.LOW: f"{gene} confers LOW risk (OR={odds_ratio:.2f}). Effect size is small but may be significant in aggregate.",
            RiskLevel.PROTECTIVE: f"{gene} is PROTECTIVE (OR={odds_ratio:.2f}). Risk alleles reduce disease odds by {(1-odds_ratio)*100:.0f}%.",
            RiskLevel.NEUTRAL: f"{gene} shows NEUTRAL effect (OR={odds_ratio:.2f}). No significant association detected."
        }

        return interpretations[risk_level]


class RiskScoreCalculator:
    """Calculates various genetic risk scores"""

    def __init__(self):
        self.classifier = RiskClassifier()

    def calculate_individual_score(self, gene: RiskGene) -> RiskScore:
        """Calculate risk score for a single gene"""
        if gene.odds_ratio:
            score = gene.odds_ratio
            score_type = 'or'
        elif gene.beta:
            score = np.exp(gene.beta)  # Convert to OR
            score_type = 'beta_converted'
        else:
            score = 1.0
            score_type = 'unknown'

        risk_level = self.classifier.classify_or(score, gene.p_value)

        return RiskScore(
            gene=gene.symbol,
            score=score,
            score_type=score_type,
            p_value=gene.p_value,
            risk_level=risk_level
        )

    def calculate_aggregate_score(self, genes: List[RiskGene],
                                   method: str = 'weighted_sum') -> AggregateRiskScore:
        """
        Calculate aggregate risk score across multiple genes

        Methods:
        - 'sum': Simple sum of log(OR)
        - 'weighted_sum': Sum weighted by -log10(p-value)
        - 'product': Product of ORs
        - 'mean': Average OR
        """
        if not genes:
            return AggregateRiskScore([], 0.0, method)

        individual_scores = []
        weights = []

        for gene in genes:
            if gene.odds_ratio and gene.odds_ratio > 0:
                individual_scores.append(gene.odds_ratio)
                # Weight by significance
                if gene.p_value and gene.p_value > 0:
                    weights.append(-np.log10(gene.p_value))
                else:
                    weights.append(1.0)
            else:
                individual_scores.append(1.0)
                weights.append(0.0)

        # Normalize weights
        if sum(weights) > 0:
            weights = [w / sum(weights) for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)

        # Calculate aggregate
        if method == 'sum':
            score = sum(np.log(s) for s in individual_scores)
        elif method == 'weighted_sum':
            score = sum(w * np.log(s) for w, s in zip(weights, individual_scores))
        elif method == 'product':
            score = np.prod(individual_scores)
        elif method == 'mean':
            score = np.mean(individual_scores)
        elif method == 'max':
            score = max(individual_scores)
        else:
            score = sum(np.log(s) for s in individual_scores)

        return AggregateRiskScore(
            genes=[g.symbol for g in genes],
            score=score,
            method=method,
            weights=weights,
            individual_scores=individual_scores
        )

    def rank_genes_by_risk(self, genes: List[RiskGene],
                           ascending: bool = False) -> List[Tuple[str, float, RiskLevel]]:
        """Rank genes by their risk contribution"""
        scored_genes = []

        for gene in genes:
            if gene.odds_ratio:
                risk_level = self.classifier.classify_or(gene.odds_ratio, gene.p_value)
                scored_genes.append((gene.symbol, gene.odds_ratio, risk_level))

        # Sort by OR
        scored_genes.sort(key=lambda x: x[1], reverse=not ascending)

        return scored_genes

    def compare_risks(self, gene1: RiskGene, gene2: RiskGene) -> Dict:
        """Compare risk between two genes"""
        or1 = gene1.odds_ratio or 1.0
        or2 = gene2.odds_ratio or 1.0

        ratio = or1 / or2
        log_ratio = np.log(or1) - np.log(or2)

        # Determine which is higher risk
        if ratio > 1.1:
            comparison = f"{gene1.symbol} confers higher risk than {gene2.symbol}"
            higher_risk = gene1.symbol
        elif ratio < 0.9:
            comparison = f"{gene2.symbol} confers higher risk than {gene1.symbol}"
            higher_risk = gene2.symbol
        else:
            comparison = f"{gene1.symbol} and {gene2.symbol} have similar risk"
            higher_risk = "similar"

        return {
            "gene1": gene1.symbol,
            "gene2": gene2.symbol,
            "or1": or1,
            "or2": or2,
            "ratio": ratio,
            "log_ratio": log_ratio,
            "comparison": comparison,
            "higher_risk": higher_risk
        }


class GWASStatistics:
    """GWAS statistical analysis utilities"""

    @staticmethod
    def beta_to_or(beta: float, se: float = None) -> Tuple[float, Optional[Tuple[float, float]]]:
        """Convert beta coefficient to odds ratio with CI"""
        odds_ratio = np.exp(beta)

        if se is not None:
            ci_low = np.exp(beta - 1.96 * se)
            ci_high = np.exp(beta + 1.96 * se)
            return odds_ratio, (ci_low, ci_high)

        return odds_ratio, None

    @staticmethod
    def or_to_beta(odds_ratio: float) -> float:
        """Convert odds ratio to beta coefficient"""
        return np.log(odds_ratio)

    @staticmethod
    def calculate_effect_allele_frequency(maf: float, effect_allele_is_minor: bool = True) -> float:
        """Calculate effect allele frequency from MAF"""
        if effect_allele_is_minor:
            return maf
        return 1 - maf

    @staticmethod
    def calculate_population_attributable_risk(or_value: float, eaf: float) -> float:
        """
        Calculate Population Attributable Risk (PAR)

        PAR = EAF * (OR - 1) / (1 + EAF * (OR - 1))
        """
        if or_value <= 0:
            return 0.0
        return eaf * (or_value - 1) / (1 + eaf * (or_value - 1))

    @staticmethod
    def calculate_variance_explained(beta: float, maf: float) -> float:
        """
        Calculate variance explained by a variant

        R² ≈ 2 * beta² * MAF * (1 - MAF)
        """
        return 2 * beta**2 * maf * (1 - maf)


class RiskReasoning:
    """Main class for Risk-Aware reasoning"""

    def __init__(self):
        self.classifier = RiskClassifier()
        self.calculator = RiskScoreCalculator()
        self.gwas_stats = GWASStatistics()
        self.risk_scores: List[RiskScore] = []

    def analyze_genes(self, genes: List[RiskGene]) -> Dict:
        """Comprehensive risk analysis of genes"""
        results = {
            "individual_scores": [],
            "ranked_genes": [],
            "risk_distribution": {},
            "aggregate_scores": {},
            "summary_stats": {}
        }

        # Calculate individual scores
        for gene in genes:
            score = self.calculator.calculate_individual_score(gene)
            results["individual_scores"].append(score)
            self.risk_scores.append(score)

        # Rank genes
        results["ranked_genes"] = self.calculator.rank_genes_by_risk(genes)

        # Risk distribution
        risk_counts = {level: 0 for level in RiskLevel}
        for score in results["individual_scores"]:
            if score.risk_level:
                risk_counts[score.risk_level] += 1
        results["risk_distribution"] = {k.value: v for k, v in risk_counts.items()}

        # Calculate aggregate scores with different methods
        for method in ['sum', 'weighted_sum', 'product', 'mean', 'max']:
            agg = self.calculator.calculate_aggregate_score(genes, method)
            results["aggregate_scores"][method] = agg.score

        # Summary statistics
        or_values = [g.odds_ratio for g in genes if g.odds_ratio]
        if or_values:
            results["summary_stats"] = {
                "mean_or": np.mean(or_values),
                "median_or": np.median(or_values),
                "max_or": max(or_values),
                "min_or": min(or_values),
                "std_or": np.std(or_values),
                "n_genes": len(or_values)
            }

        return results

    def generate_risk_questions(self, genes: List[RiskGene]) -> List[BenchmarkItem]:
        """Generate risk-aware benchmark questions"""
        questions = []

        # Analyze genes first
        analysis = self.analyze_genes(genes)

        # R-RISK-LEVEL questions
        for gene in genes[:30]:
            if gene.odds_ratio:
                risk_level = self.classifier.classify_or(gene.odds_ratio, gene.p_value)
                interpretation = self.classifier.get_risk_interpretation(
                    gene.odds_ratio, gene.symbol
                )

                q = BenchmarkItem(
                    id=f"R-{len(questions):04d}",
                    taxonomy="R",
                    label=RiskLabel.RISK_LEVEL.value,
                    template_id="R-RISK-LEVEL-01",
                    question=f"What is the risk level of {gene.symbol} for COVID-19 severity based on OR={gene.odds_ratio:.2f}?",
                    answer=f"{risk_level.value.upper()} risk. {interpretation}",
                    explanation=f"Risk classification: HIGH (OR≥1.5), MODERATE (1.2≤OR<1.5), LOW (0.8<OR<1.2), PROTECTIVE (OR≤0.8).",
                    source_genes=[gene.symbol],
                    source_data={"or": gene.odds_ratio, "p_value": gene.p_value},
                    algorithm_used="risk_classification"
                )
                questions.append(q)

        # R-RISK-COMPARE questions
        genes_with_or = [g for g in genes if g.odds_ratio]
        for i in range(min(15, len(genes_with_or) - 1)):
            gene1, gene2 = genes_with_or[i], genes_with_or[i + 1]
            comparison = self.calculator.compare_risks(gene1, gene2)

            q = BenchmarkItem(
                id=f"R-{len(questions):04d}",
                taxonomy="R",
                label=RiskLabel.RISK_COMPARE.value,
                template_id="R-RISK-COMPARE-01",
                question=f"Compare the risk contribution of {gene1.symbol} (OR={gene1.odds_ratio:.2f}) vs {gene2.symbol} (OR={gene2.odds_ratio:.2f}) for COVID-19.",
                answer=comparison["comparison"],
                explanation=f"OR ratio: {comparison['ratio']:.2f}. Higher OR indicates greater risk contribution.",
                source_genes=[gene1.symbol, gene2.symbol],
                source_data=comparison,
                algorithm_used="risk_comparison"
            )
            questions.append(q)

        # R-RISK-RANK questions
        if len(genes_with_or) >= 3:
            top_genes = analysis["ranked_genes"][:5]
            gene_names = [g[0] for g in top_genes]
            q = BenchmarkItem(
                id=f"R-{len(questions):04d}",
                taxonomy="R",
                label=RiskLabel.RISK_RANK.value,
                template_id="R-RISK-RANK-01",
                question=f"Rank the following genes by their COVID-19 risk contribution: {', '.join(gene_names)}",
                answer=f"Ranking (highest to lowest risk): {' > '.join([f'{g[0]} (OR={g[1]:.2f})' for g in top_genes])}",
                explanation="Genes ranked by odds ratio. Higher OR = greater risk contribution.",
                source_genes=gene_names,
                algorithm_used="risk_ranking"
            )
            questions.append(q)

        # R-RISK-AGGREGATE questions
        if len(genes_with_or) >= 3:
            subset_genes = genes_with_or[:5]
            agg_score = self.calculator.calculate_aggregate_score(subset_genes, 'weighted_sum')

            q = BenchmarkItem(
                id=f"R-{len(questions):04d}",
                taxonomy="R",
                label=RiskLabel.RISK_AGGREGATE.value,
                template_id="R-RISK-AGGREGATE-01",
                question=f"Calculate the cumulative weighted risk score for genes: {', '.join(agg_score.genes)}",
                answer=f"Aggregate risk score (weighted sum of log-OR): {agg_score.score:.3f}",
                explanation=f"Individual ORs: {[f'{g}:{s:.2f}' for g, s in zip(agg_score.genes, agg_score.individual_scores)]}. Weighted by -log10(p-value).",
                source_genes=agg_score.genes,
                source_data={"score": agg_score.score, "method": agg_score.method},
                algorithm_used="aggregate_scoring"
            )
            questions.append(q)

        # R-BETA-INTERPRET questions
        for gene in genes[:10]:
            if gene.odds_ratio:
                beta = self.gwas_stats.or_to_beta(gene.odds_ratio)
                maf = gene.maf or 0.1

                par = self.gwas_stats.calculate_population_attributable_risk(
                    gene.odds_ratio, maf
                )

                q = BenchmarkItem(
                    id=f"R-{len(questions):04d}",
                    taxonomy="R",
                    label=RiskLabel.BETA_INTERPRET.value,
                    template_id="R-BETA-INTERPRET-01",
                    question=f"Interpret the GWAS statistics for {gene.symbol}: OR={gene.odds_ratio:.2f}, MAF={maf:.2f}",
                    answer=f"Beta coefficient: {beta:.3f}. Population Attributable Risk (PAR): {par:.1%}",
                    explanation=f"Beta = ln(OR). PAR indicates the proportion of disease cases attributable to this variant if causal.",
                    source_genes=[gene.symbol],
                    source_data={"or": gene.odds_ratio, "beta": beta, "maf": maf, "par": par},
                    algorithm_used="gwas_interpretation"
                )
                questions.append(q)

        return questions


# Factory function
def create_risk_module() -> RiskReasoning:
    """Create Risk-Aware reasoning module"""
    return RiskReasoning()


if __name__ == "__main__":
    # Test
    module = create_risk_module()

    # Create test genes
    test_genes = [
        RiskGene(symbol="ACE2", odds_ratio=2.1, p_value=1e-10, maf=0.15),
        RiskGene(symbol="TYK2", odds_ratio=1.6, p_value=2.3e-8, maf=0.03),
        RiskGene(symbol="ABO", odds_ratio=1.3, p_value=1e-5, maf=0.35),
        RiskGene(symbol="IFNAR2", odds_ratio=0.8, p_value=1e-16, maf=0.66),
    ]

    # Analyze
    results = module.analyze_genes(test_genes)
    print("Risk Distribution:", results["risk_distribution"])
    print("Ranked Genes:", results["ranked_genes"])
    print("Aggregate Scores:", results["aggregate_scores"])
    print("Summary Stats:", results["summary_stats"])

    # Generate questions
    questions = module.generate_risk_questions(test_genes)
    print(f"\nGenerated {len(questions)} questions")
    for q in questions[:3]:
        print(f"\n{q.id}: {q.question}")
        print(f"Answer: {q.answer}")
