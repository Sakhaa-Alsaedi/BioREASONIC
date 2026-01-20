"""
STRING-DB API Client

Fetches protein-protein interaction networks and computes network centrality measures.

API Documentation: https://string-db.org/cgi/help?subpage=api
"""

import requests
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class ProteinInteraction:
    """Protein-protein interaction from STRING"""
    protein1: str
    protein2: str
    gene1: str
    gene2: str
    combined_score: int  # 0-1000
    experimental_score: int = 0
    database_score: int = 0
    textmining_score: int = 0
    coexpression_score: int = 0

    @property
    def confidence(self) -> float:
        """Confidence score normalized to 0-1"""
        return self.combined_score / 1000.0


@dataclass
class NetworkCentrality:
    """Network centrality measures for a protein/gene"""
    protein_id: str
    gene_symbol: str
    degree: float = 0.0           # Degree centrality (normalized)
    betweenness: float = 0.0      # Betweenness centrality
    closeness: float = 0.0        # Closeness centrality
    pagerank: float = 0.0         # PageRank
    eigenvector: float = 0.0      # Eigenvector centrality
    clustering: float = 0.0       # Local clustering coefficient

    def to_dict(self) -> Dict[str, float]:
        return {
            "degree": self.degree,
            "betweenness": self.betweenness,
            "closeness": self.closeness,
            "pagerank": self.pagerank,
            "eigenvector": self.eigenvector,
            "clustering": self.clustering
        }

    def compute_structure_score(
        self,
        alpha: float = 0.25,
        beta: float = 0.25,
        gamma: float = 0.25,
        delta: float = 0.25
    ) -> float:
        """Compute weighted structure score for ROCKET"""
        total = alpha + beta + gamma + delta
        return (
            (alpha / total) * self.degree +
            (beta / total) * self.betweenness +
            (gamma / total) * self.closeness +
            (delta / total) * self.pagerank
        )


@dataclass
class FunctionalEnrichment:
    """Functional enrichment result from STRING"""
    term_id: str
    term_name: str
    category: str  # GO Process, GO Function, GO Component, KEGG, etc.
    p_value: float
    fdr: float
    gene_count: int
    genes: List[str] = field(default_factory=list)


class StringDBClient:
    """
    STRING-DB API Client

    Provides access to protein-protein interaction networks
    and network analysis functions.
    """

    BASE_URL = "https://string-db.org/api"
    VERSION = "11.5"

    def __init__(
        self,
        species: int = 9606,  # Human NCBI taxonomy ID
        score_threshold: int = 400,  # Medium confidence
        network_type: str = "physical"  # "physical" or "functional"
    ):
        """
        Initialize STRING-DB client

        Args:
            species: NCBI taxonomy ID (9606 for human)
            score_threshold: Minimum combined score (0-1000)
            network_type: Type of network ("physical" or "functional")
        """
        self.species = species
        self.score_threshold = score_threshold
        self.network_type = network_type

        # Network cache
        self._network: Optional[nx.Graph] = None
        self._protein_to_gene: Dict[str, str] = {}
        self._gene_to_protein: Dict[str, str] = {}

    def _make_request(
        self,
        endpoint: str,
        params: Dict,
        output_format: str = "json"
    ) -> Optional[List[Dict]]:
        """Make API request"""
        url = f"{self.BASE_URL}/{output_format}/{endpoint}"

        try:
            response = requests.post(url, data=params, timeout=60)
            response.raise_for_status()

            if output_format == "json":
                return response.json()
            else:
                return response.text

        except requests.RequestException as e:
            logger.error(f"STRING-DB API request failed: {e}")
            return None

    def get_string_ids(self, identifiers: List[str]) -> Dict[str, str]:
        """
        Map gene symbols to STRING protein IDs

        Args:
            identifiers: List of gene symbols

        Returns:
            Dictionary mapping gene symbol to STRING ID
        """
        params = {
            "identifiers": "\r".join(identifiers),
            "species": self.species,
            "limit": 1,
            "echo_query": 1
        }

        data = self._make_request("get_string_ids", params)
        if not data:
            return {}

        mapping = {}
        for entry in data:
            query = entry.get("queryItem", "")
            string_id = entry.get("stringId", "")
            preferred_name = entry.get("preferredName", "")

            if query and string_id:
                mapping[query] = string_id
                self._protein_to_gene[string_id] = preferred_name
                self._gene_to_protein[preferred_name] = string_id

        return mapping

    def get_interactions(
        self,
        proteins: List[str],
        add_nodes: int = 0
    ) -> List[ProteinInteraction]:
        """
        Get protein-protein interactions

        Args:
            proteins: List of STRING protein IDs or gene symbols
            add_nodes: Number of additional interactors to add

        Returns:
            List of ProteinInteraction objects
        """
        # Map gene symbols to STRING IDs if needed
        if proteins and not proteins[0].startswith(str(self.species)):
            mapping = self.get_string_ids(proteins)
            proteins = list(mapping.values())

        if not proteins:
            return []

        params = {
            "identifiers": "\r".join(proteins),
            "species": self.species,
            "required_score": self.score_threshold,
            "network_type": self.network_type,
            "add_nodes": add_nodes
        }

        data = self._make_request("network", params)
        if not data:
            return []

        interactions = []
        for entry in data:
            interaction = ProteinInteraction(
                protein1=entry.get("stringId_A", ""),
                protein2=entry.get("stringId_B", ""),
                gene1=entry.get("preferredName_A", ""),
                gene2=entry.get("preferredName_B", ""),
                combined_score=entry.get("score", 0),
                experimental_score=entry.get("escore", 0),
                database_score=entry.get("dscore", 0),
                textmining_score=entry.get("tscore", 0),
                coexpression_score=entry.get("ascore", 0)
            )
            interactions.append(interaction)

            # Update mappings
            self._protein_to_gene[interaction.protein1] = interaction.gene1
            self._protein_to_gene[interaction.protein2] = interaction.gene2

        return interactions

    def build_network(
        self,
        genes: List[str],
        add_interactors: int = 10
    ) -> nx.Graph:
        """
        Build a NetworkX graph from STRING interactions

        Args:
            genes: Seed genes
            add_interactors: Number of additional interactors

        Returns:
            NetworkX Graph object
        """
        interactions = self.get_interactions(genes, add_nodes=add_interactors)

        G = nx.Graph()

        for interaction in interactions:
            G.add_edge(
                interaction.gene1,
                interaction.gene2,
                weight=interaction.confidence,
                combined_score=interaction.combined_score
            )

        self._network = G
        return G

    def compute_centrality(
        self,
        gene: str,
        network: Optional[nx.Graph] = None
    ) -> Optional[NetworkCentrality]:
        """
        Compute network centrality measures for a gene

        Args:
            gene: Gene symbol
            network: NetworkX graph (uses cached network if None)

        Returns:
            NetworkCentrality object or None
        """
        G = network or self._network
        if G is None or gene not in G.nodes():
            return None

        try:
            # Compute centralities
            degree_cent = nx.degree_centrality(G)
            betweenness_cent = nx.betweenness_centrality(G)
            closeness_cent = nx.closeness_centrality(G)
            pagerank = nx.pagerank(G)

            # Eigenvector centrality may fail on disconnected graphs
            try:
                eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000)
            except nx.NetworkXException:
                eigenvector_cent = {n: 0.0 for n in G.nodes()}

            clustering = nx.clustering(G)

            return NetworkCentrality(
                protein_id=self._gene_to_protein.get(gene, ""),
                gene_symbol=gene,
                degree=degree_cent.get(gene, 0.0),
                betweenness=betweenness_cent.get(gene, 0.0),
                closeness=closeness_cent.get(gene, 0.0),
                pagerank=pagerank.get(gene, 0.0),
                eigenvector=eigenvector_cent.get(gene, 0.0),
                clustering=clustering.get(gene, 0.0)
            )

        except Exception as e:
            logger.error(f"Failed to compute centrality for {gene}: {e}")
            return None

    def compute_all_centralities(
        self,
        genes: List[str],
        add_interactors: int = 20
    ) -> Dict[str, NetworkCentrality]:
        """
        Compute centrality measures for multiple genes

        Args:
            genes: List of gene symbols
            add_interactors: Number of additional interactors for context

        Returns:
            Dictionary mapping gene to NetworkCentrality
        """
        # Build network
        G = self.build_network(genes, add_interactors=add_interactors)

        results = {}
        for gene in genes:
            centrality = self.compute_centrality(gene, G)
            if centrality:
                results[gene] = centrality

        return results

    def get_functional_enrichment(
        self,
        genes: List[str]
    ) -> List[FunctionalEnrichment]:
        """
        Get functional enrichment analysis

        Args:
            genes: List of gene symbols

        Returns:
            List of FunctionalEnrichment results
        """
        # Map to STRING IDs
        mapping = self.get_string_ids(genes)
        string_ids = list(mapping.values())

        if not string_ids:
            return []

        params = {
            "identifiers": "\r".join(string_ids),
            "species": self.species
        }

        data = self._make_request("enrichment", params)
        if not data:
            return []

        enrichments = []
        for entry in data:
            enrichment = FunctionalEnrichment(
                term_id=entry.get("term", ""),
                term_name=entry.get("description", ""),
                category=entry.get("category", ""),
                p_value=entry.get("p_value", 1.0),
                fdr=entry.get("fdr", 1.0),
                gene_count=entry.get("number_of_genes", 0),
                genes=entry.get("inputGenes", "").split(",") if entry.get("inputGenes") else []
            )
            enrichments.append(enrichment)

        # Sort by FDR
        enrichments.sort(key=lambda x: x.fdr)

        return enrichments

    def get_network_image_url(
        self,
        genes: List[str],
        add_nodes: int = 0
    ) -> str:
        """
        Get URL for network visualization image

        Args:
            genes: Gene symbols
            add_nodes: Additional interactors

        Returns:
            URL string
        """
        mapping = self.get_string_ids(genes)
        string_ids = list(mapping.values())

        if not string_ids:
            return ""

        params = "&".join([
            f"identifiers={','.join(string_ids)}",
            f"species={self.species}",
            f"add_white_nodes={add_nodes}",
            f"network_flavor={self.network_type}"
        ])

        return f"{self.BASE_URL}/image/network?{params}"

    def get_neighbors(self, gene: str, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Get top interacting neighbors for a gene

        Args:
            gene: Gene symbol
            limit: Maximum number of neighbors

        Returns:
            List of (neighbor_gene, confidence) tuples
        """
        interactions = self.get_interactions([gene], add_nodes=limit)

        neighbors = []
        for interaction in interactions:
            if interaction.gene1 == gene:
                neighbors.append((interaction.gene2, interaction.confidence))
            elif interaction.gene2 == gene:
                neighbors.append((interaction.gene1, interaction.confidence))

        # Sort by confidence and return top
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:limit]


# Convenience functions
def get_network_centrality(gene: str, context_genes: Optional[List[str]] = None) -> Optional[NetworkCentrality]:
    """
    Quick lookup of network centrality for a gene

    Args:
        gene: Gene symbol
        context_genes: Additional genes for network context

    Returns:
        NetworkCentrality or None
    """
    client = StringDBClient()
    genes = [gene] + (context_genes or [])
    centralities = client.compute_all_centralities(genes)
    return centralities.get(gene)


def get_ppi_network(genes: List[str]) -> nx.Graph:
    """Quick lookup to build PPI network"""
    client = StringDBClient()
    return client.build_network(genes)


# Example usage
if __name__ == "__main__":
    print("STRING-DB API Client")
    print("=" * 50)

    client = StringDBClient(score_threshold=400)

    # Test genes (RA and COVID-19 related)
    test_genes = ["TYK2", "JAK1", "JAK2", "STAT3", "ACE2", "TMPRSS2"]

    print(f"\nBuilding PPI network for: {', '.join(test_genes)}")

    # Get interactions
    interactions = client.get_interactions(test_genes, add_nodes=5)
    print(f"\nFound {len(interactions)} interactions")

    # Show top interactions
    print("\nTop interactions by score:")
    sorted_interactions = sorted(interactions, key=lambda x: x.combined_score, reverse=True)
    for i, interaction in enumerate(sorted_interactions[:10]):
        print(f"  {interaction.gene1} -- {interaction.gene2}: {interaction.confidence:.2f}")

    # Compute centralities
    print("\n" + "=" * 50)
    print("\nNetwork Centrality Measures:")
    centralities = client.compute_all_centralities(test_genes, add_interactors=20)

    for gene in test_genes:
        if gene in centralities:
            c = centralities[gene]
            print(f"\n{gene}:")
            print(f"  Degree: {c.degree:.4f}")
            print(f"  Betweenness: {c.betweenness:.4f}")
            print(f"  Closeness: {c.closeness:.4f}")
            print(f"  PageRank: {c.pagerank:.4f}")
            print(f"  Structure Score: {c.compute_structure_score():.4f}")

    # Functional enrichment
    print("\n" + "=" * 50)
    print("\nFunctional Enrichment (top 5):")
    enrichments = client.get_functional_enrichment(test_genes)
    for e in enrichments[:5]:
        print(f"  [{e.category}] {e.term_name}")
        print(f"    FDR: {e.fdr:.2e}, Genes: {e.gene_count}")
