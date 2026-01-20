"""
Ensembl REST API Client

Fetches gene annotations, coordinates, and variant information from Ensembl.

API Documentation: https://rest.ensembl.org/
"""

import requests
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GeneInfo:
    """Gene information from Ensembl"""
    gene_id: str           # Ensembl gene ID (ENSG...)
    gene_symbol: str       # HGNC symbol
    description: str
    biotype: str           # protein_coding, lncRNA, etc.
    chromosome: str
    start: int
    end: int
    strand: int            # 1 or -1
    length: int            # Gene length in bp

    @property
    def length_kb(self) -> float:
        """Gene length in kilobases"""
        return self.length / 1000.0


@dataclass
class TranscriptInfo:
    """Transcript information"""
    transcript_id: str
    gene_id: str
    biotype: str
    is_canonical: bool
    length: int
    exon_count: int


@dataclass
class VariantInfo:
    """Variant information from Ensembl"""
    rsid: str
    chromosome: str
    position: int
    alleles: str
    ancestral_allele: Optional[str]
    minor_allele: Optional[str]
    maf: Optional[float]
    consequence_type: Optional[str]
    gene_symbol: Optional[str]
    gene_id: Optional[str]


class EnsemblClient:
    """
    Ensembl REST API Client

    Provides access to gene annotations, coordinates, and variant data.
    """

    BASE_URL = "https://rest.ensembl.org"

    def __init__(
        self,
        species: str = "homo_sapiens",
        rate_limit: float = 0.067  # 15 requests per second max
    ):
        """
        Initialize Ensembl client

        Args:
            species: Species name (default: homo_sapiens)
            rate_limit: Minimum seconds between requests
        """
        self.species = species
        self.rate_limit = rate_limit
        self._last_request_time = 0
        self._cache: Dict[str, Any] = {}

    def _rate_limit_wait(self):
        """Wait to respect rate limits"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        method: str = "GET",
        data: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Make rate-limited API request"""
        self._rate_limit_wait()

        url = f"{self.BASE_URL}{endpoint}"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        try:
            if method == "GET":
                response = requests.get(url, params=params, headers=headers, timeout=30)
            else:
                response = requests.post(url, params=params, json=data, headers=headers, timeout=30)

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 1))
                logger.warning(f"Rate limited, waiting {retry_after}s")
                time.sleep(retry_after)
                return self._make_request(endpoint, params, method, data)

            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            logger.error(f"Ensembl API request failed: {e}")
            return None

    def get_gene_by_symbol(self, symbol: str) -> Optional[GeneInfo]:
        """
        Get gene information by HGNC symbol

        Args:
            symbol: Gene symbol (e.g., "TYK2")

        Returns:
            GeneInfo object or None
        """
        cache_key = f"gene_symbol_{symbol}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Use xrefs to find gene by symbol
        endpoint = f"/xrefs/symbol/{self.species}/{symbol}"
        params = {"object_type": "gene"}

        data = self._make_request(endpoint, params)
        if not data:
            return None

        # Find the gene entry
        gene_id = None
        for entry in data:
            if entry.get("type") == "gene":
                gene_id = entry.get("id")
                break

        if not gene_id:
            logger.warning(f"Gene not found: {symbol}")
            return None

        # Get full gene info
        gene_info = self.get_gene_by_id(gene_id)
        if gene_info:
            self._cache[cache_key] = gene_info

        return gene_info

    def get_gene_by_id(self, gene_id: str) -> Optional[GeneInfo]:
        """
        Get gene information by Ensembl ID

        Args:
            gene_id: Ensembl gene ID (e.g., "ENSG00000105397")

        Returns:
            GeneInfo object or None
        """
        cache_key = f"gene_id_{gene_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        endpoint = f"/lookup/id/{gene_id}"
        params = {"expand": 1}

        data = self._make_request(endpoint, params)
        if not data:
            return None

        gene_info = GeneInfo(
            gene_id=data.get("id", ""),
            gene_symbol=data.get("display_name", ""),
            description=data.get("description", ""),
            biotype=data.get("biotype", ""),
            chromosome=str(data.get("seq_region_name", "")),
            start=data.get("start", 0),
            end=data.get("end", 0),
            strand=data.get("strand", 1),
            length=data.get("end", 0) - data.get("start", 0) + 1
        )

        self._cache[cache_key] = gene_info
        return gene_info

    def get_gene_length(self, symbol: str) -> int:
        """
        Get gene length in base pairs

        Args:
            symbol: Gene symbol

        Returns:
            Gene length in bp (0 if not found)
        """
        gene_info = self.get_gene_by_symbol(symbol)
        return gene_info.length if gene_info else 0

    def get_transcripts(self, gene_id: str) -> List[TranscriptInfo]:
        """
        Get transcripts for a gene

        Args:
            gene_id: Ensembl gene ID

        Returns:
            List of TranscriptInfo objects
        """
        endpoint = f"/lookup/id/{gene_id}"
        params = {"expand": 1, "utr": 1}

        data = self._make_request(endpoint, params)
        if not data or "Transcript" not in data:
            return []

        transcripts = []
        for t in data.get("Transcript", []):
            transcript = TranscriptInfo(
                transcript_id=t.get("id", ""),
                gene_id=gene_id,
                biotype=t.get("biotype", ""),
                is_canonical=t.get("is_canonical", 0) == 1,
                length=t.get("length", 0),
                exon_count=len(t.get("Exon", []))
            )
            transcripts.append(transcript)

        return transcripts

    def get_variant_by_rsid(self, rsid: str) -> Optional[VariantInfo]:
        """
        Get variant information by rsID

        Args:
            rsid: dbSNP rsID (e.g., "rs12345")

        Returns:
            VariantInfo object or None
        """
        # Normalize rsid
        if not rsid.startswith("rs"):
            rsid = f"rs{rsid}"

        cache_key = f"variant_{rsid}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        endpoint = f"/variation/{self.species}/{rsid}"

        data = self._make_request(endpoint)
        if not data:
            return None

        # Get mapping info
        mappings = data.get("mappings", [])
        if not mappings:
            return None

        mapping = mappings[0]

        # Get consequence
        consequence = None
        gene_symbol = None
        gene_id = None

        if "most_severe_consequence" in data:
            consequence = data["most_severe_consequence"]

        # Get MAF
        maf = None
        minor_allele = data.get("minor_allele")
        if "MAF" in data:
            maf = data["MAF"]

        variant_info = VariantInfo(
            rsid=rsid,
            chromosome=mapping.get("seq_region_name", ""),
            position=mapping.get("start", 0),
            alleles=mapping.get("allele_string", ""),
            ancestral_allele=data.get("ancestral_allele"),
            minor_allele=minor_allele,
            maf=maf,
            consequence_type=consequence,
            gene_symbol=gene_symbol,
            gene_id=gene_id
        )

        self._cache[cache_key] = variant_info
        return variant_info

    def get_variant_consequences(self, rsid: str) -> List[Dict]:
        """
        Get variant consequences (VEP-like annotation)

        Args:
            rsid: dbSNP rsID

        Returns:
            List of consequence dictionaries
        """
        if not rsid.startswith("rs"):
            rsid = f"rs{rsid}"

        endpoint = f"/vep/{self.species}/id/{rsid}"

        data = self._make_request(endpoint)
        if not data or not isinstance(data, list):
            return []

        consequences = []
        for entry in data:
            for tc in entry.get("transcript_consequences", []):
                consequences.append({
                    "gene_symbol": tc.get("gene_symbol"),
                    "gene_id": tc.get("gene_id"),
                    "transcript_id": tc.get("transcript_id"),
                    "consequence_terms": tc.get("consequence_terms", []),
                    "impact": tc.get("impact"),
                    "biotype": tc.get("biotype"),
                    "sift_prediction": tc.get("sift_prediction"),
                    "polyphen_prediction": tc.get("polyphen_prediction")
                })

        return consequences

    def get_genes_in_region(
        self,
        chromosome: str,
        start: int,
        end: int
    ) -> List[GeneInfo]:
        """
        Get genes in a genomic region

        Args:
            chromosome: Chromosome name (e.g., "1", "X")
            start: Start position
            end: End position

        Returns:
            List of GeneInfo objects
        """
        endpoint = f"/overlap/region/{self.species}/{chromosome}:{start}-{end}"
        params = {"feature": "gene"}

        data = self._make_request(endpoint, params)
        if not data:
            return []

        genes = []
        for entry in data:
            if entry.get("feature_type") == "gene":
                gene = GeneInfo(
                    gene_id=entry.get("gene_id", ""),
                    gene_symbol=entry.get("external_name", ""),
                    description=entry.get("description", ""),
                    biotype=entry.get("biotype", ""),
                    chromosome=chromosome,
                    start=entry.get("start", 0),
                    end=entry.get("end", 0),
                    strand=entry.get("strand", 1),
                    length=entry.get("end", 0) - entry.get("start", 0) + 1
                )
                genes.append(gene)

        return genes

    def batch_get_genes(self, symbols: List[str]) -> Dict[str, GeneInfo]:
        """
        Get multiple genes by symbols (uses POST for efficiency)

        Args:
            symbols: List of gene symbols

        Returns:
            Dictionary mapping symbol to GeneInfo
        """
        endpoint = f"/lookup/symbol/{self.species}"
        data = {"symbols": symbols}

        result = self._make_request(endpoint, method="POST", data=data)
        if not result:
            return {}

        genes = {}
        for symbol, info in result.items():
            if info and isinstance(info, dict):
                gene = GeneInfo(
                    gene_id=info.get("id", ""),
                    gene_symbol=symbol,
                    description=info.get("description", ""),
                    biotype=info.get("biotype", ""),
                    chromosome=str(info.get("seq_region_name", "")),
                    start=info.get("start", 0),
                    end=info.get("end", 0),
                    strand=info.get("strand", 1),
                    length=info.get("end", 0) - info.get("start", 0) + 1
                )
                genes[symbol] = gene

        return genes


# Convenience functions
def get_gene_length(symbol: str) -> int:
    """Quick lookup of gene length"""
    client = EnsemblClient()
    return client.get_gene_length(symbol)


def get_variant_info(rsid: str) -> Optional[VariantInfo]:
    """Quick lookup of variant info"""
    client = EnsemblClient()
    return client.get_variant_by_rsid(rsid)


# Example usage
if __name__ == "__main__":
    print("Ensembl REST API Client")
    print("=" * 50)

    client = EnsemblClient()

    # Test genes
    test_genes = ["TYK2", "ACE2", "HLA-DRB1", "BRCA1"]

    print("\nGene Information:")
    for symbol in test_genes:
        gene = client.get_gene_by_symbol(symbol)
        if gene:
            print(f"\n{symbol}:")
            print(f"  Ensembl ID: {gene.gene_id}")
            print(f"  Chromosome: {gene.chromosome}")
            print(f"  Position: {gene.start:,}-{gene.end:,}")
            print(f"  Length: {gene.length:,} bp ({gene.length_kb:.1f} kb)")
            print(f"  Biotype: {gene.biotype}")
        else:
            print(f"\n{symbol}: Not found")

    # Test variants
    print("\n" + "=" * 50)
    print("\nVariant Information:")
    test_variants = ["rs34536443", "rs12720356", "rs2476601"]

    for rsid in test_variants:
        variant = client.get_variant_by_rsid(rsid)
        if variant:
            print(f"\n{rsid}:")
            print(f"  Position: chr{variant.chromosome}:{variant.position:,}")
            print(f"  Alleles: {variant.alleles}")
            print(f"  MAF: {variant.maf}")
            print(f"  Consequence: {variant.consequence_type}")
        else:
            print(f"\n{rsid}: Not found")
