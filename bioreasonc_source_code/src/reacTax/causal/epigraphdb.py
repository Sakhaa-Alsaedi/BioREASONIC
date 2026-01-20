"""
EpiGraphDB API Client

Client for querying EpiGraphDB (https://epigraphdb.org) for
Mendelian Randomization results and gene-disease associations.
"""

import requests
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class EpiGraphDBClient:
    """
    Client for EpiGraphDB API

    EpiGraphDB aggregates epidemiological and biomedical data,
    including pre-computed MR results from large-scale analyses.
    """

    BASE_URL = "https://api.epigraphdb.org"

    def __init__(self, timeout: int = 30):
        """
        Initialize EpiGraphDB client

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout

    def get_mr_results(self, exposure: str, outcome: str) -> List[Dict]:
        """
        Get MR results for exposure-outcome pair

        Args:
            exposure: Exposure trait name
            outcome: Outcome trait name

        Returns:
            List of MR results with estimates and statistics
        """
        try:
            url = f"{self.BASE_URL}/mr"
            params = {
                "exposure_trait": exposure,
                "outcome_trait": outcome
            }

            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            return data.get("results", [])

        except requests.RequestException as e:
            logger.warning(f"EpiGraphDB MR request failed: {e}")
            return []

    def search_traits(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for traits in EpiGraphDB

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of matching traits
        """
        try:
            url = f"{self.BASE_URL}/meta/nodes/Gwas/search"
            params = {"name": query}

            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            return results[:limit]

        except requests.RequestException as e:
            logger.warning(f"EpiGraphDB trait search failed: {e}")
            return []

    def get_gene_to_disease(self, gene: str) -> List[Dict]:
        """
        Get gene-disease associations

        Args:
            gene: Gene symbol (e.g., 'ACE2')

        Returns:
            List of disease associations
        """
        try:
            url = f"{self.BASE_URL}/gene/druggability/ppi"
            params = {"gene_name": gene}

            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            return data.get("results", [])

        except requests.RequestException as e:
            logger.warning(f"EpiGraphDB gene-disease request failed: {e}")
            return []

    def get_literature_evidence(self, gene: str, disease: str) -> List[Dict]:
        """
        Get literature evidence for gene-disease relationship

        Args:
            gene: Gene symbol
            disease: Disease name

        Returns:
            List of literature evidence
        """
        try:
            url = f"{self.BASE_URL}/literature/gene"
            params = {
                "gene_name": gene,
                "object_name": disease
            }

            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            return data.get("results", [])

        except requests.RequestException as e:
            logger.warning(f"EpiGraphDB literature request failed: {e}")
            return []

    def get_pathway_evidence(self, gene: str) -> List[Dict]:
        """
        Get pathway information for a gene

        Args:
            gene: Gene symbol

        Returns:
            List of pathway associations
        """
        try:
            url = f"{self.BASE_URL}/mappings/gene-to-protein"
            params = {"gene_name": gene}

            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            return data.get("results", [])

        except requests.RequestException as e:
            logger.warning(f"EpiGraphDB pathway request failed: {e}")
            return []
