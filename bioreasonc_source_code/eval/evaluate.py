#!/usr/bin/env python3
"""
BioREASONC-Bench Evaluation Runner

Evaluates LLM responses on the benchmark using GRASS, CARES, and ROCKET scores.

Usage:
    python eval/evaluate.py --benchmark outputs/bioreasonc_bench_v1.jsonl \
                           --responses results/model_responses.jsonl \
                           --output results/evaluation_report.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from grass import GRASSCalculator, SNPRisk, GeneScore
from cares import CARESCalculator, CARESConfig, ApplicationDomain, score_llm_response
from rocket import ROCKETCalculator, ComponentScores, NetworkCentrality

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """
    Comprehensive benchmark evaluator using GRASS, CARES, and ROCKET scores
    """

    def __init__(
        self,
        domain: str = 'default',
        cares_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize evaluator

        Args:
            domain: Application domain for CARES scoring
            cares_weights: Custom weights for CARES categories
        """
        # CARES configuration
        domain_map = {
            'drug_interaction': ApplicationDomain.DRUG_INTERACTION,
            'clinical_decision': ApplicationDomain.CLINICAL_DECISION,
            'literature_summary': ApplicationDomain.LITERATURE_SUMMARY,
            'research_exploration': ApplicationDomain.RESEARCH_EXPLORATION,
            'default': ApplicationDomain.DEFAULT
        }

        cares_config = CARESConfig(
            domain=domain_map.get(domain, ApplicationDomain.DEFAULT)
        )

        if cares_weights:
            cares_config.weight_s = cares_weights.get('S', 0.25)
            cares_config.weight_c = cares_weights.get('C', 0.25)
            cares_config.weight_r = cares_weights.get('R', 0.25)
            cares_config.weight_m = cares_weights.get('M', 0.25)

        self.cares_calculator = CARESCalculator(cares_config)
        self.grass_calculator = GRASSCalculator()
        self.rocket_calculator = ROCKETCalculator()

        # Store benchmark items
        self.benchmark_items: Dict[str, Dict] = {}
        self.model_responses: Dict[str, Dict] = {}

    def load_benchmark(self, filepath: str):
        """Load benchmark items from JSONL file"""
        logger.info(f"Loading benchmark from {filepath}")

        with open(filepath, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                self.benchmark_items[item['id']] = item

        logger.info(f"Loaded {len(self.benchmark_items)} benchmark items")

    def load_responses(self, filepath: str):
        """
        Load model responses from JSONL file

        Expected format:
        {
            "id": "R-0001",
            "model_response": "...",
            "confidence": 0.85  # Optional
        }
        """
        logger.info(f"Loading responses from {filepath}")

        with open(filepath, 'r') as f:
            for line in f:
                response = json.loads(line.strip())
                self.model_responses[response['id']] = response

        logger.info(f"Loaded {len(self.model_responses)} model responses")

    def evaluate_response(
        self,
        item_id: str,
        model_response: str,
        confidence: float = 0.5
    ) -> Dict:
        """
        Evaluate a single response

        Args:
            item_id: Benchmark item ID
            model_response: Model's response text
            confidence: Model's confidence score

        Returns:
            Evaluation result dictionary
        """
        if item_id not in self.benchmark_items:
            return {'error': f'Item {item_id} not found in benchmark'}

        item = self.benchmark_items[item_id]
        category = item.get('taxonomy', item['id'][0])  # Extract from ID if not present
        gold_answer = item.get('answer', '')

        # Score the response
        score, estimated_confidence = score_llm_response(
            response=model_response,
            gold_answer=gold_answer,
            category=category,
            question_text=item.get('question', '')
        )

        # Use provided confidence or estimated
        final_confidence = confidence if confidence != 0.5 else estimated_confidence

        # Add to CARES calculator
        self.cares_calculator.add_score(
            question_id=item_id,
            category=category,
            score=score,
            confidence=final_confidence,
            question_text=item.get('question'),
            model_response=model_response,
            gold_answer=gold_answer
        )

        return {
            'id': item_id,
            'category': category,
            'score': score,
            'confidence': final_confidence,
            'is_correct': score >= 4,
            'is_hallucination': score <= 1 and final_confidence > 0.7
        }

    def evaluate_all_responses(self) -> List[Dict]:
        """
        Evaluate all loaded responses

        Returns:
            List of evaluation results
        """
        results = []

        for item_id, response_data in self.model_responses.items():
            result = self.evaluate_response(
                item_id=item_id,
                model_response=response_data.get('model_response', ''),
                confidence=response_data.get('confidence', 0.5)
            )
            results.append(result)

        return results

    def compute_grass_scores(
        self,
        snp_data: Optional[List[Dict]] = None,
        gene_data: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Compute GRASS/WGRS scores from genetic data

        Args:
            snp_data: List of SNP annotations
            gene_data: Gene metadata (length, GWAS scores)

        Returns:
            Dictionary of gene to WGRS score
        """
        if snp_data:
            for snp in snp_data:
                self.grass_calculator.add_snp(
                    gene_symbol=snp['gene'],
                    rsid=snp.get('rsid', 'unknown'),
                    chromosome=snp.get('chromosome', ''),
                    position=snp.get('position', 0),
                    maf=snp.get('maf', 0.01),
                    odds_ratio=snp.get('odds_ratio'),
                    beta=snp.get('beta'),
                    p_value=snp.get('p_value'),
                    functional_impact=snp.get('impact'),
                    gene_length=snp.get('gene_length', 10000)
                )

        if gene_data:
            for gene, data in gene_data.items():
                if 'gwas_score' in data:
                    self.grass_calculator.set_gwas_score(gene, data['gwas_score'])

        return self.grass_calculator.compute_all_scores()

    def compute_rocket_scores(
        self,
        wgrs_scores: Dict[str, float],
        network_data: Optional[Dict] = None,
        enrichment_data: Optional[Dict] = None,
        semantic_data: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Compute ROCKET scores from multi-omics data

        Args:
            wgrs_scores: Gene to WGRS mapping
            network_data: Network centrality per gene
            enrichment_data: Enrichment scores per gene
            semantic_data: Semantic scores per gene

        Returns:
            Dictionary of gene to ROCKET score
        """
        # Normalize WGRS
        if wgrs_scores:
            wgrs_min = min(wgrs_scores.values())
            wgrs_max = max(wgrs_scores.values())
        else:
            return {}

        for gene, wgrs in wgrs_scores.items():
            s_r = self.rocket_calculator.compute_risk_score(wgrs, wgrs_min, wgrs_max)

            # Structure score from network
            s_s = 0.0
            if network_data and gene in network_data:
                centrality = NetworkCentrality(**network_data[gene])
                s_s = centrality.compute_structure_score()

            # Enrichment score
            s_e = enrichment_data.get(gene, 0.0) if enrichment_data else 0.0

            # Semantic score
            s_m = semantic_data.get(gene, 0.0) if semantic_data else 0.0

            self.rocket_calculator.set_gene_scores(gene, s_r, s_s, s_e, s_m)

        return self.rocket_calculator.compute_all_rockets()

    def generate_report(self) -> Dict:
        """
        Generate comprehensive evaluation report

        Returns:
            Complete evaluation report dictionary
        """
        # CARES report
        cares_report = self.cares_calculator.get_detailed_report()

        # GRASS summary
        grass_scores = self.grass_calculator.compute_all_scores()
        grass_summary = {
            'total_genes': len(grass_scores),
            'top_genes': self.grass_calculator.get_ranked_genes(top_n=10),
            'score_distribution': {
                'min': min(grass_scores.values()) if grass_scores else 0,
                'max': max(grass_scores.values()) if grass_scores else 0,
                'mean': sum(grass_scores.values()) / len(grass_scores) if grass_scores else 0
            }
        }

        # ROCKET summary
        rocket_scores = self.rocket_calculator.compute_all_rockets()
        rocket_targets = self.rocket_calculator.get_ranked_targets(threshold=0.5)
        rocket_summary = {
            'total_genes': len(rocket_scores),
            'high_confidence_targets': len([t for t in rocket_targets if t[1] >= 0.85]),
            'moderate_confidence_targets': len([t for t in rocket_targets if 0.70 <= t[1] < 0.85]),
            'top_targets': [
                {'gene': g, 'score': s, 'interpretation': i}
                for g, s, i in rocket_targets[:10]
            ]
        }

        return {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'benchmark_items': len(self.benchmark_items),
                'evaluated_responses': len(self.model_responses)
            },
            'cares': cares_report,
            'grass': grass_summary,
            'rocket': rocket_summary,
            'summary': {
                'overall_cares_score': cares_report['cares_score'],
                'hallucination_rate': cares_report['hallucination_rate'],
                'calibration_error': cares_report['calibration_error'],
                'top_risk_genes': [g for g, _ in grass_summary['top_genes'][:5]],
                'therapeutic_targets': [t['gene'] for t in rocket_summary['top_targets'][:5]]
            }
        }

    def export_report(self, filepath: str):
        """Export evaluation report to JSON file"""
        report = self.generate_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report exported to {filepath}")


def evaluate_benchmark_responses(
    benchmark_path: str,
    responses_path: str,
    output_path: str,
    domain: str = 'default',
    genetic_data_path: Optional[str] = None
):
    """
    Main evaluation function

    Args:
        benchmark_path: Path to benchmark JSONL
        responses_path: Path to model responses JSONL
        output_path: Path for output report
        domain: Application domain for scoring
        genetic_data_path: Optional path to genetic data for GRASS/ROCKET
    """
    evaluator = BenchmarkEvaluator(domain=domain)

    # Load data
    evaluator.load_benchmark(benchmark_path)
    evaluator.load_responses(responses_path)

    # Evaluate all responses
    logger.info("Evaluating responses...")
    results = evaluator.evaluate_all_responses()

    # Load genetic data if provided
    if genetic_data_path and Path(genetic_data_path).exists():
        logger.info(f"Loading genetic data from {genetic_data_path}")
        with open(genetic_data_path, 'r') as f:
            genetic_data = json.load(f)

        snp_data = genetic_data.get('snps', [])
        gene_data = genetic_data.get('genes', {})

        wgrs = evaluator.compute_grass_scores(snp_data, gene_data)
        evaluator.compute_rocket_scores(
            wgrs,
            network_data=genetic_data.get('network'),
            enrichment_data=genetic_data.get('enrichment'),
            semantic_data=genetic_data.get('semantic')
        )

    # Generate and export report
    evaluator.export_report(output_path)

    # Print summary
    report = evaluator.generate_report()
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"CARES Score: {report['summary']['overall_cares_score']:.4f}")
    print(f"Hallucination Rate: {report['summary']['hallucination_rate']:.2%}")
    print(f"Calibration Error: {report['summary']['calibration_error']:.4f}")
    print(f"\nTop Risk Genes: {', '.join(report['summary']['top_risk_genes'])}")
    print(f"Therapeutic Targets: {', '.join(report['summary']['therapeutic_targets'])}")
    print("="*60)


def create_sample_responses(benchmark_path: str, output_path: str):
    """Create sample response file for testing"""
    responses = []

    with open(benchmark_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            # Create mock response (use actual answer for high score)
            responses.append({
                'id': item['id'],
                'model_response': item['answer'],  # Perfect response for testing
                'confidence': 0.85
            })

    with open(output_path, 'w') as f:
        for resp in responses:
            f.write(json.dumps(resp) + '\n')

    print(f"Created sample responses at {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="BioREASONC-Bench Evaluator"
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model responses')
    eval_parser.add_argument('--benchmark', '-b', required=True,
                            help='Path to benchmark JSONL file')
    eval_parser.add_argument('--responses', '-r', required=True,
                            help='Path to model responses JSONL file')
    eval_parser.add_argument('--output', '-o', default='evaluation_report.json',
                            help='Output path for evaluation report')
    eval_parser.add_argument('--domain', '-d', default='default',
                            choices=['drug_interaction', 'clinical_decision',
                                    'literature_summary', 'research_exploration', 'default'],
                            help='Application domain for CARES scoring')
    eval_parser.add_argument('--genetic-data', '-g',
                            help='Path to genetic data JSON for GRASS/ROCKET')

    # Create sample command
    sample_parser = subparsers.add_parser('create-sample',
                                          help='Create sample response file')
    sample_parser.add_argument('--benchmark', '-b', required=True,
                              help='Path to benchmark JSONL file')
    sample_parser.add_argument('--output', '-o', default='sample_responses.jsonl',
                              help='Output path for sample responses')

    args = parser.parse_args()

    if args.command == 'evaluate':
        evaluate_benchmark_responses(
            benchmark_path=args.benchmark,
            responses_path=args.responses,
            output_path=args.output,
            domain=args.domain,
            genetic_data_path=args.genetic_data
        )
    elif args.command == 'create-sample':
        create_sample_responses(args.benchmark, args.output)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
