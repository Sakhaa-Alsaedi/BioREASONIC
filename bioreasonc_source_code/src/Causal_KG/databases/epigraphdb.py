"""
EpiGraphDB API Client

Fetches Mendelian Randomization evidence and causal relationships from EpiGraphDB.

API Documentation: https://docs.epigraphdb.org/api/
"""

import requests
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MREvidence:
    """Mendelian Randomization evidence from EpiGraphDB"""
    exposure: str  # Exposure trait
    outcome: str   # Outcome trait
    exposure_id: str
    outcome_id: str
    beta: float    # Effect size
    se: float      # Standard error
    p_value: float
    method: str    # MR method (IVW, MR-Egger, etc.)
    nsnp: int      # Number of SNPs used
    selection: str  # Selection method
    consortium: str = ""
    pmid: str = ""

    @property
    def is_significant(self) -> bool:
        """Check if association is significant at p < 0.05"""
        return self.p_value < 0.05

    @property
    def effect_direction(self) -> str:
        """Get effect direction"""
        if self.beta > 0:
            return "positive"
        elif self.beta < 0:
            return "negative"
        return "null"

    def get_causal_score(self) -> float:
        """
        Compute causal evidence score for ROCKET S_C

        Considers p-value significance and number of SNPs
        """
        import math

        # Base score from -log10(p-value), capped at 10
        if self.p_value <= 0:
            log_p = 10.0
        else:
            log_p = min(-math.log10(max(self.p_value, 1e-300)), 10.0)

        # Normalize to 0-1
        p_score = log_p / 10.0

        # Boost for more SNPs (more robust)
        snp_factor = min(1.0, math.log10(max(self.nsnp, 1) + 1) / 2.0)

        return 0.7 * p_score + 0.3 * snp_factor


@dataclass
class GeneToDisease:
    """Gene-disease relationship from EpiGraphDB"""
    gene_symbol: str
    gene_id: str
    disease_name: str
    disease_id: str  # EFO or other ontology ID
    score: float     # Association score
    source: str      # Data source
    pmid: Optional[str] = None


@dataclass
class PheWASResult:
    """PheWAS result from EpiGraphDB"""
    rsid: str
    trait: str
    trait_id: str
    p_value: float
    beta: float
    se: float
    ancestry: str = ""
    source: str = ""


@dataclass
class DrugTarget:
    """Drug-target relationship from EpiGraphDB"""
    drug_name: str
    drug_id: str
    gene_symbol: str
    gene_id: str
    action_type: str
    source: str


class EpiGraphDBClient:
    """
    EpiGraphDB API Client

    Provides access to Mendelian Randomization evidence,
    GWAS results, and drug-target relationships.
    """

    BASE_URL = "https://api.epigraphdb.org"

    def __init__(self, timeout: int = 60):
        """
        Initialize EpiGraphDB client

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self._cache: Dict[str, Any] = {}

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        method: str = "GET"
    ) -> Optional[Dict]:
        """Make API request"""
        url = f"{self.BASE_URL}{endpoint}"

        try:
            if method == "GET":
                response = requests.get(url, params=params, timeout=self.timeout)
            else:
                response = requests.post(url, json=params, timeout=self.timeout)

            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            logger.error(f"EpiGraphDB API request failed: {e}")
            return None

    def get_mr_evidence(
        self,
        exposure: str,
        outcome: Optional[str] = None,
        p_threshold: float = 0.05
    ) -> List[MREvidence]:
        """
        Get Mendelian Randomization evidence for an exposure

        Args:
            exposure: Exposure trait name
            outcome: Optional outcome trait to filter
            p_threshold: P-value threshold

        Returns:
            List of MREvidence objects
        """
        endpoint = "/mr"
        params = {
            "exposure_trait": exposure,
            "pval_threshold": p_threshold
        }

        if outcome:
            params["outcome_trait"] = outcome

        data = self._make_request(endpoint, params)
        if not data or "results" not in data:
            return []

        results = []
        for entry in data["results"]:
            mr = MREvidence(
                exposure=entry.get("exposure.trait", ""),
                outcome=entry.get("outcome.trait", ""),
                exposure_id=entry.get("exposure.id", ""),
                outcome_id=entry.get("outcome.id", ""),
                beta=entry.get("mr.b", 0.0),
                se=entry.get("mr.se", 0.0),
                p_value=entry.get("mr.pval", 1.0),
                method=entry.get("mr.method", ""),
                nsnp=entry.get("mr.nsnp", 0),
                selection=entry.get("mr.selection", ""),
                consortium=entry.get("mr.consortium", ""),
                pmid=entry.get("mr.pmid", "")
            )
            results.append(mr)

        return results

    def get_mr_eve_evidence(
        self,
        gene_symbol: str,
        trait: Optional[str] = None
    ) -> List[MREvidence]:
        """
        Get MR-EvE (MR using cis-eQTLs) evidence for gene-trait

        This is particularly useful for drug target validation.

        Args:
            gene_symbol: Gene symbol (e.g., "TYK2")
            trait: Optional trait to filter

        Returns:
            List of MREvidence objects
        """
        endpoint = "/mr-eve"
        params = {
            "gene_name": gene_symbol
        }

        if trait:
            params["outcome_trait"] = trait

        data = self._make_request(endpoint, params)
        if not data or "results" not in data:
            return []

        results = []
        for entry in data["results"]:
            mr = MREvidence(
                exposure=gene_symbol,  # Gene expression as exposure
                outcome=entry.get("outcome.trait", ""),
                exposure_id=entry.get("gene.ensembl_id", ""),
                outcome_id=entry.get("outcome.id", ""),
                beta=entry.get("mr.b", 0.0),
                se=entry.get("mr.se", 0.0),
                p_value=entry.get("mr.pval", 1.0),
                method=entry.get("mr.method", "MR-EvE"),
                nsnp=entry.get("mr.nsnp", 0),
                selection=entry.get("mr.selection", "cis-eQTL"),
                consortium=entry.get("mr.consortium", "")
            )
            results.append(mr)

        return results

    def get_gene_disease_associations(
        self,
        gene_symbol: str,
        source: str = "all"
    ) -> List[GeneToDisease]:
        """
        Get gene-disease associations

        Args:
            gene_symbol: Gene symbol
            source: Data source filter ("all", "disgenet", "opentargets", etc.)

        Returns:
            List of GeneToDisease objects
        """
        # Try literature-based associations
        endpoint = "/gene-disease"
        params = {
            "gene_name": gene_symbol
        }

        data = self._make_request(endpoint, params)
        if not data or "results" not in data:
            return []

        results = []
        for entry in data["results"]:
            assoc = GeneToDisease(
                gene_symbol=entry.get("gene.name", gene_symbol),
                gene_id=entry.get("gene.ensembl_id", ""),
                disease_name=entry.get("disease.label", ""),
                disease_id=entry.get("disease.id", ""),
                score=entry.get("r.score", 0.0),
                source=entry.get("r.source", ""),
                pmid=entry.get("r.pmid")
            )
            results.append(assoc)

        return results

    def get_phewas(
        self,
        rsid: str,
        p_threshold: float = 1e-5
    ) -> List[PheWASResult]:
        """
        Get PheWAS results for a variant

        Args:
            rsid: SNP rsID
            p_threshold: P-value threshold

        Returns:
            List of PheWASResult objects
        """
        # Normalize rsid
        if not rsid.startswith("rs"):
            rsid = f"rs{rsid}"

        endpoint = "/phewas"
        params = {
            "variant": rsid,
            "pval_threshold": p_threshold
        }

        data = self._make_request(endpoint, params)
        if not data or "results" not in data:
            return []

        results = []
        for entry in data["results"]:
            phewas = PheWASResult(
                rsid=rsid,
                trait=entry.get("gwas.trait", ""),
                trait_id=entry.get("gwas.id", ""),
                p_value=entry.get("assoc.pval", 1.0),
                beta=entry.get("assoc.beta", 0.0),
                se=entry.get("assoc.se", 0.0),
                ancestry=entry.get("gwas.ancestry", ""),
                source=entry.get("gwas.source", "")
            )
            results.append(phewas)

        # Sort by p-value
        results.sort(key=lambda x: x.p_value)

        return results

    def get_drug_targets(
        self,
        gene_symbol: str
    ) -> List[DrugTarget]:
        """
        Get drugs targeting a gene

        Args:
            gene_symbol: Gene symbol

        Returns:
            List of DrugTarget objects
        """
        endpoint = "/drugs/risk-factors"
        params = {
            "gene_name": gene_symbol
        }

        data = self._make_request(endpoint, params)
        if not data or "results" not in data:
            return []

        results = []
        for entry in data["results"]:
            drug = DrugTarget(
                drug_name=entry.get("drug.label", ""),
                drug_id=entry.get("drug.id", ""),
                gene_symbol=gene_symbol,
                gene_id=entry.get("gene.ensembl_id", ""),
                action_type=entry.get("r.action_type", ""),
                source=entry.get("r.source", "")
            )
            results.append(drug)

        return results

    def get_pathway_evidence(
        self,
        pathway: str
    ) -> List[Dict]:
        """
        Get genes in a pathway

        Args:
            pathway: Pathway name (e.g., "JAK-STAT signaling")

        Returns:
            List of gene dictionaries
        """
        endpoint = "/pathway"
        params = {
            "pathway_name": pathway
        }

        data = self._make_request(endpoint, params)
        if not data or "results" not in data:
            return []

        return data["results"]

    def get_causal_evidence_score(
        self,
        gene_symbol: str,
        disease_keywords: List[str]
    ) -> float:
        """
        Compute causal evidence score for gene-disease relationship

        Uses MR evidence to compute score for ROCKET S_C

        Args:
            gene_symbol: Gene symbol
            disease_keywords: Keywords to match diseases

        Returns:
            Causal evidence score (0-1)
        """
        # Get MR-EvE evidence (gene expression -> disease)
        mr_evidence = self.get_mr_eve_evidence(gene_symbol)

        if not mr_evidence:
            # Try traditional MR with gene name as exposure
            mr_evidence = self.get_mr_evidence(gene_symbol)

        if not mr_evidence:
            return 0.0

        # Filter for relevant diseases
        relevant_evidence = []
        for mr in mr_evidence:
            outcome_lower = mr.outcome.lower()
            for keyword in disease_keywords:
                if keyword.lower() in outcome_lower:
                    relevant_evidence.append(mr)
                    break

        if not relevant_evidence:
            # Return average score of all evidence if no keyword match
            scores = [mr.get_causal_score() for mr in mr_evidence]
            return sum(scores) / len(scores) * 0.5  # Discount for non-specific match

        # Return best score from relevant evidence
        scores = [mr.get_causal_score() for mr in relevant_evidence]
        return max(scores)

    def get_variant_trait_associations(
        self,
        rsid: str,
        trait: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive variant-trait associations

        Args:
            rsid: SNP rsID
            trait: Optional trait to filter

        Returns:
            Dictionary with PheWAS and coloc results
        """
        # Get PheWAS
        phewas_results = self.get_phewas(rsid)

        if trait:
            trait_lower = trait.lower()
            phewas_results = [
                p for p in phewas_results
                if trait_lower in p.trait.lower()
            ]

        return {
            "rsid": rsid,
            "phewas_count": len(phewas_results),
            "phewas_results": phewas_results[:20],  # Top 20
            "most_significant_trait": phewas_results[0].trait if phewas_results else None,
            "most_significant_pval": phewas_results[0].p_value if phewas_results else None
        }


# Alternative: MR-Base API (for additional MR evidence)
class MRBaseClient:
    """
    MR-Base API Client

    Alternative source for MR evidence.
    API: https://gwas.mrcieu.ac.uk/
    """

    BASE_URL = "https://api.opengwas.io/api"

    def __init__(self, timeout: int = 60):
        self.timeout = timeout

    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request"""
        url = f"{self.BASE_URL}{endpoint}"

        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"MR-Base API request failed: {e}")
            return None

    def search_gwas(self, trait: str) -> List[Dict]:
        """
        Search for GWAS studies by trait

        Args:
            trait: Trait name to search

        Returns:
            List of GWAS study info
        """
        endpoint = "/gwasinfo"
        params = {"trait": trait}

        data = self._make_request(endpoint, params)
        return data if data else []

    def get_associations(
        self,
        gwas_id: str,
        p_threshold: float = 5e-8
    ) -> List[Dict]:
        """
        Get genome-wide significant associations

        Args:
            gwas_id: GWAS study ID
            p_threshold: P-value threshold

        Returns:
            List of associations
        """
        endpoint = f"/associations/{gwas_id}"
        params = {"pval": p_threshold}

        data = self._make_request(endpoint, params)
        return data if data else []

    def get_variant_associations(
        self,
        rsid: str,
        p_threshold: float = 1e-5
    ) -> List[Dict]:
        """
        Get associations for a specific variant

        Args:
            rsid: SNP rsID
            p_threshold: P-value threshold

        Returns:
            List of associations
        """
        if not rsid.startswith("rs"):
            rsid = f"rs{rsid}"

        endpoint = f"/phewas/{rsid}"
        params = {"pval": p_threshold}

        data = self._make_request(endpoint, params)
        return data if data else []


# Convenience functions
def get_mr_evidence(exposure: str, outcome: str = None) -> List[MREvidence]:
    """Quick lookup of MR evidence"""
    client = EpiGraphDBClient()
    return client.get_mr_evidence(exposure, outcome)


def get_causal_score(gene: str, disease_keywords: List[str]) -> float:
    """Quick causal evidence score for ROCKET"""
    client = EpiGraphDBClient()
    return client.get_causal_evidence_score(gene, disease_keywords)


def get_phewas_results(rsid: str) -> List[PheWASResult]:
    """Quick PheWAS lookup"""
    client = EpiGraphDBClient()
    return client.get_phewas(rsid)


# Example usage
if __name__ == "__main__":
    print("EpiGraphDB API Client")
    print("=" * 50)

    client = EpiGraphDBClient()

    # Test MR evidence
    print("\nMR Evidence for 'Body mass index':")
    mr_results = client.get_mr_evidence("Body mass index", outcome="Type 2 diabetes")
    for mr in mr_results[:5]:
        print(f"  {mr.exposure} -> {mr.outcome}")
        print(f"    Beta: {mr.beta:.3f}, P: {mr.p_value:.2e}")
        print(f"    Method: {mr.method}, SNPs: {mr.nsnp}")
        print(f"    Causal Score: {mr.get_causal_score():.3f}")

    # Test gene-disease
    print("\n" + "=" * 50)
    print("\nGene-Disease Associations for TYK2:")
    gene_disease = client.get_gene_disease_associations("TYK2")
    for assoc in gene_disease[:5]:
        print(f"  {assoc.gene_symbol} - {assoc.disease_name}")
        print(f"    Score: {assoc.score:.3f}, Source: {assoc.source}")

    # Test MR-EvE (gene expression MR)
    print("\n" + "=" * 50)
    print("\nMR-EvE Evidence for TYK2:")
    mr_eve = client.get_mr_eve_evidence("TYK2")
    for mr in mr_eve[:5]:
        print(f"  {mr.exposure} -> {mr.outcome}")
        print(f"    Beta: {mr.beta:.3f}, P: {mr.p_value:.2e}")
        print(f"    Causal Score: {mr.get_causal_score():.3f}")

    # Test causal score
    print("\n" + "=" * 50)
    score = client.get_causal_evidence_score("TYK2", ["rheumatoid", "arthritis", "autoimmune"])
    print(f"\nCausal Evidence Score for TYK2 (autoimmune): {score:.4f}")
