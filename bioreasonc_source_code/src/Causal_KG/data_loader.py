"""
Data Ingestion Module for BioREASONC-Bench

Handles loading and preprocessing of:
- COVID-19 genetic risk data
- RA genetic risk data
- SNP annotations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from ..schema import GeneticVariant, RiskGene

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestor:
    """Loads and preprocesses genetic risk data"""

    def __init__(self, data_dir: str = "../Data"):
        self.data_dir = Path(data_dir)
        self.covid_genes: List[RiskGene] = []
        self.ra_genes: List[RiskGene] = []
        self.variants: List[GeneticVariant] = []
        self.gene_variant_map: Dict[str, List[str]] = {}
        self.variant_gene_map: Dict[str, List[str]] = {}

    def load_all(self) -> Dict:
        """Load all available data"""
        logger.info("Loading all genetic risk data...")

        results = {
            "covid_genes": self.load_covid_risk_genes(),
            "covid_variants": self.load_covid_risk_variants(),
            "ra_genes": self.load_ra_risk_genes(),
            "ra_snps": self.load_ra_snp_factors()
        }

        self._build_mappings()
        logger.info(f"Loaded {len(self.covid_genes)} COVID genes, "
                   f"{len(self.ra_genes)} RA genes, "
                   f"{len(self.variants)} variants")

        return results

    def load_covid_risk_genes(self) -> pd.DataFrame:
        """Load COVID-19 risk gene data"""
        filepath = self.data_dir / "COVID-19_risk_gene.csv"
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()

        df = pd.read_csv(filepath, skiprows=1)
        logger.info(f"Loaded COVID-19 risk genes: {len(df)} rows")

        # Parse into RiskGene objects
        for _, row in df.iterrows():
            gene = RiskGene(
                symbol=str(row.get('Risk gene', '')).strip(),
                odds_ratio=self._safe_float(row.get(' OR')),
                p_value=self._safe_float(row.get('P-Value')),
                maf=self._safe_float(row.get('MAF')),
                associated_diseases=['COVID-19']
            )
            if gene.symbol and gene.symbol != 'nan':
                self.covid_genes.append(gene)

        return df

    def load_covid_risk_variants(self) -> pd.DataFrame:
        """Load COVID-19 risk variant data"""
        filepath = self.data_dir / "COCID-19_risk variants.csv"
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()

        df = pd.read_csv(filepath)
        logger.info(f"Loaded COVID-19 risk variants: {len(df)} rows")

        return df

    def load_ra_risk_genes(self) -> pd.DataFrame:
        """Load RA risk gene data"""
        filepath = self.data_dir / "RA_risk genes.csv"
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()

        df = pd.read_csv(filepath, skiprows=1)
        logger.info(f"Loaded RA risk genes: {len(df)} rows")

        # Parse into RiskGene objects
        for _, row in df.iterrows():
            gene = RiskGene(
                symbol=str(row.get('Symbol', '')).strip(),
                ensembl_id=str(row.get('Ensembl Gene ID', '')),
                gene_type=str(row.get('Type', '')),
                chromosome=str(row.get('Chr', '')),
                position=self._safe_float(row.get('Position (Mbp)')),
                associated_diseases=['Rheumatoid Arthritis']
            )
            if gene.symbol and gene.symbol != 'nan':
                self.ra_genes.append(gene)

        return df

    def load_ra_snp_factors(self) -> pd.DataFrame:
        """Load RA SNP risk factor data"""
        filepath = self.data_dir / "RA_SNP risk factors.csv"
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()

        df = pd.read_csv(filepath, skiprows=1)
        logger.info(f"Loaded RA SNP factors: {len(df)} rows")

        # Parse into GeneticVariant objects
        for _, row in df.iterrows():
            variant = GeneticVariant(
                rsid=str(row.get('dbsnp.rsid', '')).strip(),
                chromosome=str(row.get('dbsnp.chrom', '')),
                position=self._safe_int(row.get('dbsnp.hg38.start')),
                ref_allele=str(row.get('dbsnp.ref', '')),
                alt_allele=str(row.get('dbsnp.alt', ''))
            )
            if variant.rsid and variant.rsid != 'nan':
                self.variants.append(variant)

                # Build variant-gene mapping
                gene_symbol = str(row.get('dbsnp.gene.symbol', '')).strip()
                if gene_symbol and gene_symbol != 'nan' and gene_symbol != 'N.A':
                    if variant.rsid not in self.variant_gene_map:
                        self.variant_gene_map[variant.rsid] = []
                    self.variant_gene_map[variant.rsid].append(gene_symbol)

                    if gene_symbol not in self.gene_variant_map:
                        self.gene_variant_map[gene_symbol] = []
                    self.gene_variant_map[gene_symbol].append(variant.rsid)

        return df

    def _build_mappings(self):
        """Build gene-variant mappings from loaded data"""
        # Update RA genes with their variants
        for gene in self.ra_genes:
            if gene.symbol in self.gene_variant_map:
                gene.associated_variants = self.gene_variant_map[gene.symbol]

    def get_all_genes(self) -> List[RiskGene]:
        """Get all unique genes"""
        all_genes = {}
        for gene in self.covid_genes + self.ra_genes:
            if gene.symbol not in all_genes:
                all_genes[gene.symbol] = gene
        return list(all_genes.values())

    def get_shared_genes(self) -> List[str]:
        """Get genes shared between COVID-19 and RA"""
        covid_symbols = {g.symbol.upper() for g in self.covid_genes}
        ra_symbols = {g.symbol.upper() for g in self.ra_genes}
        return list(covid_symbols & ra_symbols)

    def get_genes_with_or(self, min_or: float = 1.0) -> List[RiskGene]:
        """Get genes with OR above threshold"""
        return [g for g in self.covid_genes
                if g.odds_ratio and g.odds_ratio >= min_or]

    def get_high_risk_genes(self, or_threshold: float = 1.5) -> List[RiskGene]:
        """Get high-risk genes based on OR"""
        return self.get_genes_with_or(or_threshold)

    def get_gene_by_symbol(self, symbol: str) -> Optional[RiskGene]:
        """Get gene by symbol"""
        symbol_upper = symbol.upper()
        for gene in self.covid_genes + self.ra_genes:
            if gene.symbol.upper() == symbol_upper:
                return gene
        return None

    def get_variants_for_gene(self, gene_symbol: str) -> List[str]:
        """Get variants associated with a gene"""
        return self.gene_variant_map.get(gene_symbol, [])

    def get_genes_for_variant(self, rsid: str) -> List[str]:
        """Get genes associated with a variant"""
        return self.variant_gene_map.get(rsid, [])

    def to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """Convert loaded data to DataFrames"""
        covid_df = pd.DataFrame([g.to_dict() for g in self.covid_genes])
        ra_df = pd.DataFrame([g.to_dict() for g in self.ra_genes])
        variants_df = pd.DataFrame([v.to_dict() for v in self.variants])

        return {
            "covid_genes": covid_df,
            "ra_genes": ra_df,
            "variants": variants_df
        }

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Safely convert to float"""
        try:
            if pd.isna(value) or value == 'N.A' or value == '':
                return None
            return float(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_int(value) -> Optional[int]:
        """Safely convert to int"""
        try:
            if pd.isna(value) or value == 'N.A' or value == '':
                return None
            return int(float(value))
        except (ValueError, TypeError):
            return None


def load_data(data_dir: str = "../Data") -> DataIngestor:
    """Convenience function to load all data"""
    ingestor = DataIngestor(data_dir)
    ingestor.load_all()
    return ingestor


if __name__ == "__main__":
    # Test loading
    ingestor = load_data()
    print(f"COVID genes: {len(ingestor.covid_genes)}")
    print(f"RA genes: {len(ingestor.ra_genes)}")
    print(f"Variants: {len(ingestor.variants)}")
    print(f"Shared genes: {ingestor.get_shared_genes()}")
