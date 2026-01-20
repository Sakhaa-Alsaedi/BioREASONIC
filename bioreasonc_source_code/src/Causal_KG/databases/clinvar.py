"""
ClinVar API Client

Fetches clinical significance annotations for genetic variants from NCBI ClinVar.

API Documentation: https://www.ncbi.nlm.nih.gov/clinvar/docs/maintenance_use/
"""

import requests
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class ClinicalSignificance(Enum):
    """ClinVar clinical significance categories"""
    PATHOGENIC = "pathogenic"
    LIKELY_PATHOGENIC = "likely_pathogenic"
    UNCERTAIN_SIGNIFICANCE = "uncertain_significance"
    LIKELY_BENIGN = "likely_benign"
    BENIGN = "benign"
    CONFLICTING = "conflicting_interpretations"
    NOT_PROVIDED = "not_provided"
    UNKNOWN = "unknown"


# Mapping from ClinVar strings to enum
CLINVAR_SIGNIFICANCE_MAP = {
    "pathogenic": ClinicalSignificance.PATHOGENIC,
    "likely pathogenic": ClinicalSignificance.LIKELY_PATHOGENIC,
    "pathogenic/likely pathogenic": ClinicalSignificance.PATHOGENIC,
    "uncertain significance": ClinicalSignificance.UNCERTAIN_SIGNIFICANCE,
    "likely benign": ClinicalSignificance.LIKELY_BENIGN,
    "benign": ClinicalSignificance.BENIGN,
    "benign/likely benign": ClinicalSignificance.BENIGN,
    "conflicting interpretations of pathogenicity": ClinicalSignificance.CONFLICTING,
    "not provided": ClinicalSignificance.NOT_PROVIDED,
}


@dataclass
class ClinVarVariant:
    """ClinVar variant annotation"""
    rsid: str
    clinvar_id: Optional[str] = None
    gene_symbol: Optional[str] = None
    clinical_significance: ClinicalSignificance = ClinicalSignificance.UNKNOWN
    review_status: Optional[str] = None
    condition: Optional[str] = None
    molecular_consequence: Optional[str] = None
    hgvs: Optional[str] = None
    chromosome: Optional[str] = None
    position: Optional[int] = None
    ref_allele: Optional[str] = None
    alt_allele: Optional[str] = None

    def get_significance_score(self) -> float:
        """Get numeric score for clinical significance"""
        scores = {
            ClinicalSignificance.PATHOGENIC: 1.0,
            ClinicalSignificance.LIKELY_PATHOGENIC: 0.8,
            ClinicalSignificance.UNCERTAIN_SIGNIFICANCE: 0.3,
            ClinicalSignificance.CONFLICTING: 0.4,
            ClinicalSignificance.LIKELY_BENIGN: 0.1,
            ClinicalSignificance.BENIGN: 0.0,
            ClinicalSignificance.NOT_PROVIDED: 0.2,
            ClinicalSignificance.UNKNOWN: 0.2,
        }
        return scores.get(self.clinical_significance, 0.2)


class ClinVarClient:
    """
    ClinVar API Client

    Uses NCBI E-utilities to query ClinVar database.
    """

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    CLINVAR_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    CLINVAR_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    def __init__(
        self,
        api_key: Optional[str] = None,
        email: Optional[str] = None,
        rate_limit: float = 0.34  # 3 requests per second without API key
    ):
        """
        Initialize ClinVar client

        Args:
            api_key: NCBI API key (optional, increases rate limit to 10/sec)
            email: Email for NCBI (required by NCBI terms of service)
            rate_limit: Seconds between requests
        """
        self.api_key = api_key
        self.email = email or "user@example.com"
        self.rate_limit = rate_limit if not api_key else 0.1
        self._last_request_time = 0

        # Cache results
        self._cache: Dict[str, ClinVarVariant] = {}

    def _rate_limit_wait(self):
        """Wait to respect rate limits"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    def _make_request(self, url: str, params: Dict) -> Optional[requests.Response]:
        """Make rate-limited request"""
        self._rate_limit_wait()

        # Add common parameters
        params['email'] = self.email
        if self.api_key:
            params['api_key'] = self.api_key

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"ClinVar API request failed: {e}")
            return None

    def search_by_rsid(self, rsid: str) -> Optional[List[str]]:
        """
        Search ClinVar for a variant by rsID

        Args:
            rsid: dbSNP rsID (e.g., "rs12345" or "12345")

        Returns:
            List of ClinVar IDs
        """
        # Normalize rsid
        if not rsid.startswith('rs'):
            rsid = f'rs{rsid}'

        params = {
            'db': 'clinvar',
            'term': f'{rsid}[rs]',
            'retmode': 'json',
            'retmax': 10
        }

        response = self._make_request(self.CLINVAR_SEARCH_URL, params)
        if not response:
            return None

        try:
            data = response.json()
            id_list = data.get('esearchresult', {}).get('idlist', [])
            return id_list
        except Exception as e:
            logger.error(f"Failed to parse ClinVar search response: {e}")
            return None

    def get_variant_summary(self, clinvar_ids: List[str]) -> List[Dict]:
        """
        Get variant summaries from ClinVar IDs

        Args:
            clinvar_ids: List of ClinVar IDs

        Returns:
            List of variant summary dictionaries
        """
        if not clinvar_ids:
            return []

        params = {
            'db': 'clinvar',
            'id': ','.join(clinvar_ids),
            'retmode': 'json'
        }

        response = self._make_request(self.CLINVAR_SUMMARY_URL, params)
        if not response:
            return []

        try:
            data = response.json()
            result = data.get('result', {})

            summaries = []
            for uid in clinvar_ids:
                if uid in result and uid != 'uids':
                    summaries.append(result[uid])

            return summaries
        except Exception as e:
            logger.error(f"Failed to parse ClinVar summary response: {e}")
            return []

    def get_variant_by_rsid(self, rsid: str) -> Optional[ClinVarVariant]:
        """
        Get variant annotation by rsID

        Args:
            rsid: dbSNP rsID

        Returns:
            ClinVarVariant object or None
        """
        # Check cache
        if rsid in self._cache:
            return self._cache[rsid]

        # Normalize rsid
        if not rsid.startswith('rs'):
            rsid = f'rs{rsid}'

        # Search for ClinVar IDs
        clinvar_ids = self.search_by_rsid(rsid)
        if not clinvar_ids:
            logger.info(f"No ClinVar entry found for {rsid}")
            return None

        # Get summary
        summaries = self.get_variant_summary(clinvar_ids[:1])  # Get first match
        if not summaries:
            return None

        # Parse summary
        summary = summaries[0]
        variant = self._parse_summary(rsid, summary)

        # Cache result
        self._cache[rsid] = variant

        return variant

    def _parse_summary(self, rsid: str, summary: Dict) -> ClinVarVariant:
        """Parse ClinVar summary into ClinVarVariant"""
        # Get clinical significance
        clin_sig_str = summary.get('clinical_significance', {})
        if isinstance(clin_sig_str, dict):
            clin_sig_str = clin_sig_str.get('description', 'unknown')
        clin_sig_str = str(clin_sig_str).lower()

        clinical_significance = CLINVAR_SIGNIFICANCE_MAP.get(
            clin_sig_str,
            ClinicalSignificance.UNKNOWN
        )

        # Get gene info
        genes = summary.get('genes', [])
        gene_symbol = genes[0].get('symbol') if genes else None

        # Get variation info
        variation_set = summary.get('variation_set', [])
        hgvs = None
        if variation_set:
            hgvs = variation_set[0].get('canonical_spdi')

        return ClinVarVariant(
            rsid=rsid,
            clinvar_id=str(summary.get('uid', '')),
            gene_symbol=gene_symbol,
            clinical_significance=clinical_significance,
            review_status=summary.get('review_status'),
            condition=summary.get('trait_set', [{}])[0].get('trait_name') if summary.get('trait_set') else None,
            hgvs=hgvs
        )

    def get_variants_batch(self, rsids: List[str]) -> Dict[str, ClinVarVariant]:
        """
        Get annotations for multiple variants

        Args:
            rsids: List of rsIDs

        Returns:
            Dictionary mapping rsID to ClinVarVariant
        """
        results = {}

        for rsid in rsids:
            variant = self.get_variant_by_rsid(rsid)
            if variant:
                results[rsid] = variant

        return results

    def search_by_gene(self, gene_symbol: str, limit: int = 100) -> List[ClinVarVariant]:
        """
        Search for pathogenic variants in a gene

        Args:
            gene_symbol: Gene symbol (e.g., "BRCA1")
            limit: Maximum number of results

        Returns:
            List of ClinVarVariant objects
        """
        params = {
            'db': 'clinvar',
            'term': f'{gene_symbol}[gene] AND pathogenic[clinical significance]',
            'retmode': 'json',
            'retmax': limit
        }

        response = self._make_request(self.CLINVAR_SEARCH_URL, params)
        if not response:
            return []

        try:
            data = response.json()
            id_list = data.get('esearchresult', {}).get('idlist', [])

            if not id_list:
                return []

            # Get summaries
            summaries = self.get_variant_summary(id_list)

            variants = []
            for summary in summaries:
                # Get rsID from summary
                rsid = None
                for xref in summary.get('supporting_submissions', {}).get('rsid', []):
                    rsid = f"rs{xref}"
                    break

                if rsid:
                    variant = self._parse_summary(rsid, summary)
                    variants.append(variant)

            return variants

        except Exception as e:
            logger.error(f"Failed to search gene {gene_symbol}: {e}")
            return []


# Convenience function
def get_clinical_significance(rsid: str, api_key: Optional[str] = None) -> Optional[ClinVarVariant]:
    """
    Quick lookup of clinical significance for a variant

    Args:
        rsid: dbSNP rsID
        api_key: Optional NCBI API key

    Returns:
        ClinVarVariant or None
    """
    client = ClinVarClient(api_key=api_key)
    return client.get_variant_by_rsid(rsid)


# Example usage
if __name__ == "__main__":
    print("ClinVar API Client")
    print("=" * 50)

    # Create client
    client = ClinVarClient()

    # Test variants
    test_rsids = [
        "rs34536443",  # TYK2 protective variant
        "rs12720356",  # TYK2 variant
        "rs2476601",   # PTPN22 - RA risk
    ]

    print("\nFetching variant annotations...")
    for rsid in test_rsids:
        print(f"\n{rsid}:")
        variant = client.get_variant_by_rsid(rsid)
        if variant:
            print(f"  Gene: {variant.gene_symbol}")
            print(f"  Clinical Significance: {variant.clinical_significance.value}")
            print(f"  Review Status: {variant.review_status}")
            print(f"  Score: {variant.get_significance_score()}")
        else:
            print("  Not found in ClinVar")
