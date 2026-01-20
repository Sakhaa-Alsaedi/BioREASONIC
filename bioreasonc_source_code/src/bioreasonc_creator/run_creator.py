"""
BioREASONC-Creator Pipeline Orchestrator

Complete pipeline for generating the BioREASONC benchmark:
1. Generate Q&A from source data with ground truth
2. Paraphrase questions (optional)
3. Add explanations
4. Multi-LLM validation
5. Filter and export valid items
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from .generator import QuestionGenerator, GeneratedItem
from .paraphraser import QuestionParaphraser
from .explainer import ExplanationGenerator
from .validator import MultiLLMValidator, ValidatorConfig

# Sample data for synthetic generation
SAMPLE_GWAS_DATA = [
    {"rsid": "rs10490770", "gene": "LZTFL1", "or_value": 1.6, "p_value": 2.3e-10, "chromosome": "3"},
    {"rsid": "rs11385942", "gene": "SLC6A20", "or_value": 1.8, "p_value": 1.1e-15, "chromosome": "3"},
    {"rsid": "rs657152", "gene": "ABO", "or_value": 1.1, "p_value": 4.9e-8, "chromosome": "9"},
    {"rsid": "rs2236757", "gene": "IFNAR2", "or_value": 1.3, "p_value": 5.0e-8, "chromosome": "21"},
    {"rsid": "rs74956615", "gene": "NOTCH4", "or_value": 1.5, "p_value": 2.3e-8, "chromosome": "6"},
    {"rsid": "rs2109069", "gene": "DPP9", "or_value": 1.4, "p_value": 3.1e-12, "chromosome": "19"},
    {"rsid": "rs9380142", "gene": "HLA-G", "or_value": 0.9, "p_value": 1.5e-9, "chromosome": "6"},
    {"rsid": "rs12329760", "gene": "TMPRSS2", "or_value": 0.85, "p_value": 7.1e-6, "chromosome": "21"},
    {"rsid": "rs13050728", "gene": "IFIH1", "or_value": 1.2, "p_value": 2.0e-7, "chromosome": "2"},
    {"rsid": "rs1886814", "gene": "FOXP4", "or_value": 1.25, "p_value": 9.8e-9, "chromosome": "6"},
]


@dataclass
class PipelineConfig:
    """Configuration for the BioREASONC-Creator pipeline."""
    # Data sources
    data_dir: str = "data"
    output_dir: str = "outputs"

    # Generation settings
    items_per_category: int = 100
    taxonomies: List[str] = None  # None means all: S, C, R, M

    # Paraphrasing settings
    enable_paraphrasing: bool = True
    num_paraphrases: int = 2

    # Validation settings
    validation_threshold: float = 4.0
    require_majority: bool = True

    # API keys
    openai_api_key: Optional[str] = "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"
    anthropic_api_key: Optional[str] = "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"
    gemini_api_key: Optional[str] = None

    def __post_init__(self):
        if self.taxonomies is None:
            self.taxonomies = ['S', 'C', 'R', 'M']

        # Try to load API keys from environment if not provided
        if not self.openai_api_key:
            self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        if not self.anthropic_api_key:
            self.anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not self.gemini_api_key:
            self.gemini_api_key = os.environ.get('GEMINI_API_KEY')


class BioREASONCCreator:
    """
    Main pipeline orchestrator for creating the BioREASONC benchmark.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the BioREASONC-Creator pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.stats = {}

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize pipeline components."""
        # Generator
        self.generator = QuestionGenerator()

        # Paraphraser
        self.paraphraser = QuestionParaphraser(
            openai_api_key=self.config.openai_api_key,
            anthropic_api_key=self.config.anthropic_api_key,
            num_paraphrases=self.config.num_paraphrases,
            use_llm=True
        )

        # Explainer
        self.explainer = ExplanationGenerator(
            openai_api_key=self.config.openai_api_key,
            anthropic_api_key=self.config.anthropic_api_key,
            use_llm=True
        )

        # Validator
        validator_config = ValidatorConfig(
            passing_threshold=self.config.validation_threshold,
            require_majority=self.config.require_majority
        )
        self.validator = MultiLLMValidator(
            openai_api_key=self.config.openai_api_key,
            anthropic_api_key=self.config.anthropic_api_key,
            gemini_api_key=self.config.gemini_api_key,
            config=validator_config
        )

    def _progress_callback(self, step: str):
        """Create a progress callback for a step."""
        def callback(current, total):
            print(f"  [{step}] {current}/{total} ({100*current/total:.1f}%)", end='\r')
            if current == total:
                print()
        return callback

    def _find_data_files(self) -> Dict[str, str]:
        """Find available data files."""
        data_path = Path(self.config.data_dir)
        files = {}

        # Look for CSV files
        for csv_file in data_path.glob("*.csv"):
            name = csv_file.stem.lower()
            files[name] = str(csv_file)

        return files

    def run_generation(self, data_files: Optional[Dict[str, str]] = None) -> List[Dict]:
        """
        Step 1: Generate Q&A items from source data.

        Args:
            data_files: Optional dict of {name: path} or list of {path, disease}

        Returns:
            List of generated items
        """
        print("\n=== Step 1: Generation ===")

        if data_files is None:
            data_files = self._find_data_files()

        items = []
        if not data_files:
            print("No data files found, using sample GWAS data")
            # Use sample data with pandas DataFrame
            import pandas as pd
            df = pd.DataFrame(SAMPLE_GWAS_DATA)
            items = self.generator.generate_from_dataframe(
                df,
                disease="COVID-19 severity",
                max_per_template=self.config.items_per_category
            )
        else:
            print(f"Found data files: {list(data_files.keys())}")
            # Build file configs - map filename to disease
            disease_mapping = {
                'gwas': 'COVID-19 severity',
                'covid': 'COVID-19',
                'hgi': 'COVID-19',
                'breast_cancer': 'Breast Cancer',
                'diabetes': 'Type 2 Diabetes',
            }

            file_configs = []
            for name, path in data_files.items():
                disease = disease_mapping.get(name.lower(), 'Disease')
                for key, dis in disease_mapping.items():
                    if key in name.lower():
                        disease = dis
                        break
                file_configs.append({'path': path, 'disease': disease})

            items = self.generator.generate_from_multiple_files(
                file_configs,
                max_per_template=self.config.items_per_category
            )

        # Convert GeneratedItem objects to dicts
        item_dicts = [item.to_dict() for item in items]

        self.stats['generation'] = {
            'total_items': len(item_dicts),
            'by_taxonomy': self._count_by_taxonomy(item_dicts)
        }

        print(f"Generated {len(item_dicts)} items")
        return item_dicts

    def run_paraphrasing(self, items: List[Dict]) -> List[Dict]:
        """
        Step 2: Generate paraphrases (optional).

        Args:
            items: List of generated items

        Returns:
            List of items with paraphrases
        """
        if not self.config.enable_paraphrasing:
            print("\n=== Step 2: Paraphrasing (SKIPPED) ===")
            return items

        print("\n=== Step 2: Paraphrasing ===")

        paraphrased = self.paraphraser.paraphrase_batch(
            items,
            progress_callback=self._progress_callback("Paraphrase")
        )

        self.stats['paraphrasing'] = self.paraphraser.get_stats(paraphrased)

        print(f"Generated {len(paraphrased)} items (including paraphrases)")
        return paraphrased

    def run_explanation(self, items: List[Dict]) -> List[Dict]:
        """
        Step 3: Add explanations.

        Args:
            items: List of items

        Returns:
            List of items with explanations
        """
        print("\n=== Step 3: Explanation Generation ===")

        explained = self.explainer.add_explanations(
            items,
            progress_callback=self._progress_callback("Explain")
        )

        self.stats['explanation'] = self.explainer.get_stats(explained)

        print(f"Added explanations to {len(explained)} items")
        return explained

    def run_validation(self, items: List[Dict]) -> tuple:
        """
        Step 4: Multi-LLM validation.

        Args:
            items: List of items to validate

        Returns:
            Tuple of (valid_items, invalid_items)
        """
        print("\n=== Step 4: Multi-LLM Validation ===")

        valid, invalid = self.validator.validate_batch(
            items,
            progress_callback=self._progress_callback("Validate")
        )

        self.stats['validation'] = self.validator.get_stats(valid, invalid)

        print(f"Validation complete: {len(valid)} valid, {len(invalid)} invalid")
        return valid, invalid

    def export_results(
        self,
        valid_items: List[Dict],
        invalid_items: List[Dict],
        filename: str = "bioreasonc_bench"
    ) -> Dict[str, str]:
        """
        Step 5: Export results to files.

        Args:
            valid_items: List of valid items
            invalid_items: List of invalid items
            filename: Base filename for outputs

        Returns:
            Dict of output file paths
        """
        print("\n=== Step 5: Export Results ===")

        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        output_files = {}

        # Export valid items as JSONL (main benchmark)
        jsonl_path = output_path / f"{filename}_v1.jsonl"
        with open(jsonl_path, 'w') as f:
            for item in valid_items:
                # Clean up item for export
                export_item = self._prepare_export_item(item)
                f.write(json.dumps(export_item) + '\n')
        output_files['benchmark'] = str(jsonl_path)

        # Export invalid items for review
        if invalid_items:
            invalid_path = output_path / f"{filename}_invalid.jsonl"
            with open(invalid_path, 'w') as f:
                for item in invalid_items:
                    f.write(json.dumps(item) + '\n')
            output_files['invalid'] = str(invalid_path)

        # Export summary statistics
        summary_path = output_path / "summary.json"
        summary = {
            'version': '1.0.0',
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'items_per_category': self.config.items_per_category,
                'total_items': len(valid_items),
                'paraphrased': self.config.enable_paraphrasing,
                'validated': True
            },
            'statistics': {
                'total_items': len(valid_items),
                'by_taxonomy': self._count_by_taxonomy(valid_items),
                'by_label': self._count_by_label(valid_items),
                'avg_validation_score': sum(i.get('avg_score', 0) for i in valid_items) / len(valid_items) if valid_items else 0,
                'valid_items': len(valid_items)
            }
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        output_files['summary'] = str(summary_path)

        print(f"Exported files:")
        for name, path in output_files.items():
            print(f"  - {name}: {path}")

        return output_files

    def _item_to_dict(self, item: GeneratedItem) -> Dict:
        """Convert GeneratedItem to dictionary."""
        return item.to_dict()

    def _prepare_export_item(self, item: Dict) -> Dict:
        """Prepare item for export (remove internal fields)."""
        export_fields = [
            'id', 'taxonomy', 'label', 'template_id',
            'question', 'answer', 'explanation', 'difficulty',
            'source_genes', 'source_variants', 'source_data',
            'validation_scores', 'avg_score', 'is_valid',
            'paraphrased', 'original_question', 'algorithm_used'
        ]
        return {k: item.get(k) for k in export_fields if k in item}

    def _count_by_taxonomy(self, items: List[Dict]) -> Dict[str, int]:
        """Count items by taxonomy."""
        counts = {}
        for item in items:
            tax = item.get('taxonomy', 'U')
            counts[tax] = counts.get(tax, 0) + 1
        return counts

    def _count_by_label(self, items: List[Dict]) -> Dict[str, int]:
        """Count items by label."""
        counts = {}
        for item in items:
            label = item.get('label', 'UNKNOWN')
            counts[label] = counts.get(label, 0) + 1
        return counts

    def run(
        self,
        data_files: Optional[Dict[str, str]] = None,
        output_filename: str = "bioreasonc_bench"
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline.

        Args:
            data_files: Optional dict of data file paths
            output_filename: Base filename for outputs

        Returns:
            Dict with pipeline results and statistics
        """
        print("=" * 60)
        print("BioREASONC-Creator Pipeline")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  - Items per category: {self.config.items_per_category}")
        print(f"  - Taxonomies: {self.config.taxonomies}")
        print(f"  - Paraphrasing: {self.config.enable_paraphrasing}")
        print(f"  - Validation threshold: {self.config.validation_threshold}")

        start_time = datetime.now()

        # Step 1: Generate
        items = self.run_generation(data_files)

        # Step 2: Paraphrase
        items = self.run_paraphrasing(items)

        # Step 3: Explain
        items = self.run_explanation(items)

        # Step 4: Validate
        valid_items, invalid_items = self.run_validation(items)

        # Step 5: Export
        output_files = self.export_results(valid_items, invalid_items, output_filename)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        print(f"Duration: {duration:.1f} seconds")
        print(f"Valid items: {len(valid_items)}")
        print(f"Invalid items: {len(invalid_items)}")
        print(f"Output files: {output_files}")

        return {
            'valid_items': len(valid_items),
            'invalid_items': len(invalid_items),
            'output_files': output_files,
            'statistics': self.stats,
            'duration_seconds': duration
        }


def main():
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description='BioREASONC-Creator Pipeline')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--output-dir', default='outputs', help='Output directory')
    parser.add_argument('--items', type=int, default=100, help='Items per category')
    parser.add_argument('--no-paraphrase', action='store_true', help='Disable paraphrasing')
    parser.add_argument('--threshold', type=float, default=4.0, help='Validation threshold')

    args = parser.parse_args()

    config = PipelineConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        items_per_category=args.items,
        enable_paraphrasing=not args.no_paraphrase,
        validation_threshold=args.threshold
    )

    creator = BioREASONCCreator(config)
    results = creator.run()

    return results


if __name__ == '__main__':
    main()
