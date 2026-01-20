"""
Open Targets API Client

Fetches gene-disease associations, GWAS data, and drug targets from Open Targets.

APIs:
- Open Targets Platform: https://platform.opentargets.org/
- Open Targets Genetics: https://genetics.opentargets.org/

GraphQL API Documentation: https://platform-docs.opentargets.org/
"""

import requests
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GeneDiseaseAssociation:
    """Gene-disease association from Open Targets"""
    gene_id: str  # Ensembl gene ID
    gene_symbol: str
    disease_id: str  # EFO ID
    disease_name: str
    overall_score: float  # 0-1 association score
    genetic_association_score: float = 0.0
    literature_score: float = 0.0
    known_drug_score: float = 0.0
    affected_pathway_score: float = 0.0
    datasource_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class GWASAssociation:
    """GWAS association from Open Targets Genetics"""
    rsid: str
    gene_symbol: str
    gene_id: str
    trait_id: str
    trait_name: str
    p_value: float
    beta: Optional[float] = None
    odds_ratio: Optional[float] = None
    confidence_interval: Optional[tuple] = None
    study_id: str = ""
    sample_size: int = 0


@dataclass
class DrugTarget:
    """Drug target information"""
    gene_id: str
    gene_symbol: str
    drug_id: str
    drug_name: str
    mechanism_of_action: str
    action_type: str
    phase: int  # Clinical trial phase
    approved: bool = False


class OpenTargetsClient:
    """
    Open Targets Platform and Genetics API Client

    Uses GraphQL API for efficient queries.
    """

    PLATFORM_API = "https://api.platform.opentargets.org/api/v4/graphql"
    GENETICS_API = "https://api.genetics.opentargets.org/graphql"

    def __init__(self, timeout: int = 30):
        """
        Initialize Open Targets client

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self._cache: Dict[str, Any] = {}

    def _graphql_query(self, api_url: str, query: str, variables: Dict = None) -> Optional[Dict]:
        """Execute GraphQL query"""
        try:
            response = requests.post(
                api_url,
                json={"query": query, "variables": variables or {}},
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            if "errors" in data:
                logger.error(f"GraphQL errors: {data['errors']}")
                return None

            return data.get("data")

        except requests.RequestException as e:
            logger.error(f"Open Targets API request failed: {e}")
            return None

    def get_gene_info(self, gene_symbol: str) -> Optional[Dict]:
        """
        Get gene information by symbol

        Args:
            gene_symbol: Gene symbol (e.g., "TYK2")

        Returns:
            Gene information dictionary
        """
        # First search for the gene ID
        search_query = """
        query SearchGene($symbol: String!) {
            search(queryString: $symbol, entityNames: ["target"], page: {size: 10, index: 0}) {
                total
                hits {
                    id
                    name
                    entity
                    description
                }
            }
        }
        """

        data = self._graphql_query(
            self.PLATFORM_API,
            search_query,
            {"symbol": gene_symbol}
        )

        if not data or not data.get("search", {}).get("hits"):
            return None

        # Find the exact match
        gene_id = None
        for hit in data["search"]["hits"]:
            if hit.get("name", "").upper() == gene_symbol.upper():
                gene_id = hit.get("id")
                break

        if not gene_id and data["search"]["hits"]:
            # Use first result if no exact match
            gene_id = data["search"]["hits"][0].get("id")

        if not gene_id:
            return None

        # Get detailed gene info
        target_query = """
        query TargetInfo($ensemblId: String!) {
            target(ensemblId: $ensemblId) {
                id
                approvedSymbol
                approvedName
                biotype
            }
        }
        """

        target_data = self._graphql_query(
            self.PLATFORM_API,
            target_query,
            {"ensemblId": gene_id}
        )

        if target_data and target_data.get("target"):
            return target_data["target"]

        return {"id": gene_id, "approvedSymbol": gene_symbol}

    def get_gene_disease_associations(
        self,
        gene_symbol: str,
        limit: int = 25
    ) -> List[GeneDiseaseAssociation]:
        """
        Get disease associations for a gene

        Args:
            gene_symbol: Gene symbol
            limit: Maximum number of associations

        Returns:
            List of GeneDiseaseAssociation objects
        """
        # First get gene ID
        gene_info = self.get_gene_info(gene_symbol)
        if not gene_info:
            logger.warning(f"Gene not found: {gene_symbol}")
            return []

        gene_id = gene_info.get("id")

        query = """
        query GeneAssociations($geneId: String!, $size: Int!) {
            target(ensemblId: $geneId) {
                id
                approvedSymbol
                associatedDiseases(page: {size: $size, index: 0}) {
                    count
                    rows {
                        disease {
                            id
                            name
                        }
                        score
                        datatypeScores {
                            id
                            score
                        }
                    }
                }
            }
        }
        """

        data = self._graphql_query(
            self.PLATFORM_API,
            query,
            {"geneId": gene_id, "size": limit}
        )

        if not data or not data.get("target"):
            return []

        associations = []
        for row in data["target"].get("associatedDiseases", {}).get("rows", []):
            # Parse datatype scores
            datatype_scores = {
                ds["id"]: ds["score"]
                for ds in row.get("datatypeScores", [])
            }

            assoc = GeneDiseaseAssociation(
                gene_id=gene_id,
                gene_symbol=gene_symbol,
                disease_id=row["disease"]["id"],
                disease_name=row["disease"]["name"],
                overall_score=row["score"],
                genetic_association_score=datatype_scores.get("genetic_association", 0.0),
                literature_score=datatype_scores.get("literature", 0.0),
                known_drug_score=datatype_scores.get("known_drug", 0.0),
                affected_pathway_score=datatype_scores.get("affected_pathway", 0.0)
            )
            associations.append(assoc)

        return associations

    def get_disease_associations_for_gene(
        self,
        gene_symbol: str,
        disease_name: str
    ) -> Optional[GeneDiseaseAssociation]:
        """
        Get association between a specific gene and disease

        Args:
            gene_symbol: Gene symbol
            disease_name: Disease name to search for

        Returns:
            GeneDiseaseAssociation or None
        """
        associations = self.get_gene_disease_associations(gene_symbol, limit=100)

        disease_name_lower = disease_name.lower()
        for assoc in associations:
            if disease_name_lower in assoc.disease_name.lower():
                return assoc

        return None

    def get_gwas_variants(
        self,
        gene_symbol: str,
        limit: int = 50
    ) -> List[GWASAssociation]:
        """
        Get GWAS variants associated with a gene from Open Targets Genetics

        Args:
            gene_symbol: Gene symbol
            limit: Maximum number of variants

        Returns:
            List of GWASAssociation objects
        """
        query = """
        query GeneVariants($symbol: String!) {
            geneInfo(geneId: $symbol) {
                id
                symbol
            }
            studiesAndLeadVariantsForGene(geneId: $symbol) {
                study {
                    studyId
                    traitReported
                    traitEfos
                    nTotal
                }
                variant {
                    id
                    rsId
                }
                pval
                beta
                oddsRatio
                betaCILower
                betaCIUpper
            }
        }
        """

        # Open Targets Genetics uses Ensembl ID, need to convert
        gene_info = self.get_gene_info(gene_symbol)
        if not gene_info:
            return []

        gene_id = gene_info.get("id")

        data = self._graphql_query(
            self.GENETICS_API,
            query,
            {"symbol": gene_id}
        )

        if not data:
            return []

        variants = []
        for row in data.get("studiesAndLeadVariantsForGene", [])[:limit]:
            study = row.get("study", {})
            variant = row.get("variant", {})

            ci = None
            if row.get("betaCILower") and row.get("betaCIUpper"):
                ci = (row["betaCILower"], row["betaCIUpper"])

            gwas = GWASAssociation(
                rsid=variant.get("rsId", ""),
                gene_symbol=gene_symbol,
                gene_id=gene_id,
                trait_id=study.get("traitEfos", [""])[0] if study.get("traitEfos") else "",
                trait_name=study.get("traitReported", ""),
                p_value=row.get("pval", 1.0),
                beta=row.get("beta"),
                odds_ratio=row.get("oddsRatio"),
                confidence_interval=ci,
                study_id=study.get("studyId", ""),
                sample_size=study.get("nTotal", 0)
            )
            variants.append(gwas)

        return variants

    def get_drug_targets(self, gene_symbol: str) -> List[DrugTarget]:
        """
        Get drugs targeting a gene

        Args:
            gene_symbol: Gene symbol

        Returns:
            List of DrugTarget objects
        """
        gene_info = self.get_gene_info(gene_symbol)
        if not gene_info:
            return []

        gene_id = gene_info.get("id")

        query = """
        query DrugTargets($geneId: String!) {
            target(ensemblId: $geneId) {
                id
                approvedSymbol
                knownDrugs {
                    rows {
                        drug {
                            id
                            name
                        }
                        mechanismOfAction
                        actionType
                        phase
                        status
                    }
                }
            }
        }
        """

        data = self._graphql_query(
            self.PLATFORM_API,
            query,
            {"geneId": gene_id}
        )

        if not data or not data.get("target"):
            return []

        drugs = []
        for row in data["target"].get("knownDrugs", {}).get("rows", []):
            drug = DrugTarget(
                gene_id=gene_id,
                gene_symbol=gene_symbol,
                drug_id=row["drug"]["id"],
                drug_name=row["drug"]["name"],
                mechanism_of_action=row.get("mechanismOfAction", ""),
                action_type=row.get("actionType", ""),
                phase=row.get("phase", 0),
                approved=row.get("status") == "Approved"
            )
            drugs.append(drug)

        return drugs

    def get_gwas_score_for_disease(
        self,
        gene_symbol: str,
        disease_keywords: List[str]
    ) -> float:
        """
        Get GWAS-based association score for gene-disease pair

        Args:
            gene_symbol: Gene symbol
            disease_keywords: Keywords to match disease names

        Returns:
            Association score (0-1)
        """
        associations = self.get_gene_disease_associations(gene_symbol)

        best_score = 0.0
        for assoc in associations:
            disease_lower = assoc.disease_name.lower()
            for keyword in disease_keywords:
                if keyword.lower() in disease_lower:
                    if assoc.genetic_association_score > best_score:
                        best_score = assoc.genetic_association_score
                    break

        return best_score


# Convenience functions
def get_gene_disease_score(gene_symbol: str, disease_name: str) -> float:
    """
    Quick lookup of gene-disease association score

    Args:
        gene_symbol: Gene symbol
        disease_name: Disease name

    Returns:
        Association score (0-1)
    """
    client = OpenTargetsClient()
    assoc = client.get_disease_associations_for_gene(gene_symbol, disease_name)
    return assoc.overall_score if assoc else 0.0


def get_gwas_associations(gene_symbol: str) -> List[GWASAssociation]:
    """Quick lookup of GWAS associations"""
    client = OpenTargetsClient()
    return client.get_gwas_variants(gene_symbol)


# Example usage
if __name__ == "__main__":
    print("Open Targets API Client")
    print("=" * 50)

    client = OpenTargetsClient()

    # Test genes
    test_genes = ["TYK2", "ACE2", "HLA-DRB1"]

    for gene in test_genes:
        print(f"\n{gene}:")

        # Get gene info
        info = client.get_gene_info(gene)
        if info:
            print(f"  Ensembl ID: {info.get('id')}")

            # Get top disease associations
            associations = client.get_gene_disease_associations(gene, limit=5)
            print(f"  Top disease associations:")
            for assoc in associations[:3]:
                print(f"    - {assoc.disease_name}: {assoc.overall_score:.3f}")
                print(f"      (genetic: {assoc.genetic_association_score:.3f})")

            # Get drug targets
            drugs = client.get_drug_targets(gene)
            if drugs:
                print(f"  Drugs targeting this gene:")
                for drug in drugs[:3]:
                    print(f"    - {drug.drug_name} (Phase {drug.phase})")
