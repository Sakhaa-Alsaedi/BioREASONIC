"""
Unified Biomedical API Client

Provides a single interface to all biomedical databases for the evaluation pipeline.
Combines: ClinVar, Open Targets, Ensembl, STRING-DB, Enrichr, EpiGraphDB

This module is used by GRASS, CARES, and ROCKET evaluators.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import all API clients
from .clinvar import ClinVarClient, ClinVarVariant, ClinicalSignificance
from .open_targets import OpenTargetsClient, GeneDiseaseAssociation, GWASAssociation, DrugTarget
from .ensembl import EnsemblClient, GeneInfo, VariantInfo
from .string_db import StringDBClient, NetworkCentrality, ProteinInteraction
from .epigraphdb import EpiGraphDBClient, MREvidence

# Enrichr moved to enrichment/ module
from enrichment import EnrichrClient, EnrichmentResult

logger = logging.getLogger(__name__)


@dataclass
class VariantAnnotation:
    """Complete variant annotation from multiple sources"""
    rsid: str
    gene_symbol: Optional[str] = None
    gene_id: Optional[str] = None
    chromosome: Optional[str] = None
    position: Optional[int] = None
    alleles: Optional[str] = None

    # Clinical annotation (ClinVar)
    clinical_significance: Optional[ClinicalSignificance] = None
    clinvar_review_status: Optional[str] = None
    clinvar_condition: Optional[str] = None

    # Functional annotation (Ensembl)
    consequence_type: Optional[str] = None
    sift_prediction: Optional[str] = None
    polyphen_prediction: Optional[str] = None
    maf: Optional[float] = None

    # PheWAS
    associated_traits: List[str] = field(default_factory=list)
    best_gwas_pvalue: Optional[float] = None

    def get_risk_score(self) -> float:
        """
        Compute aggregate risk score for GRASS

        Returns:
            Risk score (0-1)
        """
        score = 0.0
        weights_sum = 0.0

        # ClinVar contribution (weight: 0.4)
        if self.clinical_significance:
            clin_scores = {
                ClinicalSignificance.PATHOGENIC: 1.0,
                ClinicalSignificance.LIKELY_PATHOGENIC: 0.8,
                ClinicalSignificance.CONFLICTING: 0.4,
                ClinicalSignificance.UNCERTAIN_SIGNIFICANCE: 0.3,
                ClinicalSignificance.LIKELY_BENIGN: 0.1,
                ClinicalSignificance.BENIGN: 0.0,
            }
            score += 0.4 * clin_scores.get(self.clinical_significance, 0.2)
            weights_sum += 0.4

        # Consequence contribution (weight: 0.3)
        if self.consequence_type:
            consequence_scores = {
                "stop_gained": 1.0,
                "frameshift_variant": 1.0,
                "splice_acceptor_variant": 0.9,
                "splice_donor_variant": 0.9,
                "start_lost": 0.9,
                "stop_lost": 0.9,
                "missense_variant": 0.6,
                "inframe_insertion": 0.5,
                "inframe_deletion": 0.5,
                "splice_region_variant": 0.4,
                "synonymous_variant": 0.1,
                "intron_variant": 0.05,
                "intergenic_variant": 0.01,
            }
            score += 0.3 * consequence_scores.get(self.consequence_type, 0.2)
            weights_sum += 0.3

        # Prediction tools contribution (weight: 0.2)
        pred_score = 0.0
        pred_count = 0
        if self.sift_prediction:
            pred_score += 1.0 if self.sift_prediction == "deleterious" else 0.0
            pred_count += 1
        if self.polyphen_prediction:
            pred_scores = {"probably_damaging": 1.0, "possibly_damaging": 0.5, "benign": 0.0}
            pred_score += pred_scores.get(self.polyphen_prediction, 0.3)
            pred_count += 1

        if pred_count > 0:
            score += 0.2 * (pred_score / pred_count)
            weights_sum += 0.2

        # MAF contribution (weight: 0.1) - rarer variants score higher
        if self.maf is not None:
            maf_score = 1.0 - min(self.maf * 20, 1.0)  # MAF < 0.05 scores higher
            score += 0.1 * maf_score
            weights_sum += 0.1

        if weights_sum > 0:
            return score / weights_sum
        return 0.2  # Default


@dataclass
class GeneAnnotation:
    """Complete gene annotation from multiple sources"""
    gene_symbol: str
    gene_id: Optional[str] = None
    description: Optional[str] = None
    biotype: Optional[str] = None
    chromosome: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    length: Optional[int] = None

    # Disease associations (Open Targets)
    disease_associations: List[GeneDiseaseAssociation] = field(default_factory=list)
    top_disease: Optional[str] = None
    top_disease_score: float = 0.0

    # Network properties (STRING-DB)
    network_centrality: Optional[NetworkCentrality] = None
    interacting_genes: List[str] = field(default_factory=list)

    # Enrichment (Enrichr)
    pathway_enrichments: List[EnrichmentResult] = field(default_factory=list)
    top_pathway: Optional[str] = None

    # Causal evidence (EpiGraphDB)
    mr_evidence: List[MREvidence] = field(default_factory=list)
    causal_score: float = 0.0

    # Drug targets
    targeting_drugs: List[DrugTarget] = field(default_factory=list)

    def get_structure_score(self) -> float:
        """Get network structure score for ROCKET S_S"""
        if self.network_centrality:
            return self.network_centrality.compute_structure_score()
        return 0.0

    def get_enrichment_score(self) -> float:
        """Get enrichment score for ROCKET S_E"""
        if not self.pathway_enrichments:
            return 0.0
        scores = [e.get_enrichment_score() for e in self.pathway_enrichments[:5]]
        return sum(scores) / len(scores) if scores else 0.0


class UnifiedBiomedicalClient:
    """
    Unified client for all biomedical APIs

    Provides high-level methods for the evaluation pipeline
    that aggregate data from multiple sources.
    """

    def __init__(
        self,
        clinvar_api_key: Optional[str] = None,
        use_cache: bool = True,
        parallel_requests: bool = True,
        max_workers: int = 5
    ):
        """
        Initialize unified client

        Args:
            clinvar_api_key: Optional NCBI API key for faster ClinVar access
            use_cache: Enable caching of results
            parallel_requests: Enable parallel API calls
            max_workers: Number of parallel workers
        """
        # Initialize individual clients
        self.clinvar = ClinVarClient(api_key=clinvar_api_key)
        self.open_targets = OpenTargetsClient()
        self.ensembl = EnsemblClient()
        self.string_db = StringDBClient()
        self.enrichr = EnrichrClient()
        self.epigraphdb = EpiGraphDBClient()

        self.use_cache = use_cache
        self.parallel_requests = parallel_requests
        self.max_workers = max_workers

        # Caches
        self._variant_cache: Dict[str, VariantAnnotation] = {}
        self._gene_cache: Dict[str, GeneAnnotation] = {}

    def get_variant_annotation(
        self,
        rsid: str,
        include_phewas: bool = False
    ) -> VariantAnnotation:
        """
        Get complete variant annotation from multiple sources

        Args:
            rsid: Variant rsID
            include_phewas: Include PheWAS results

        Returns:
            VariantAnnotation object
        """
        # Normalize rsid
        if not rsid.startswith("rs"):
            rsid = f"rs{rsid}"

        # Check cache
        cache_key = f"{rsid}_{include_phewas}"
        if self.use_cache and cache_key in self._variant_cache:
            return self._variant_cache[cache_key]

        annotation = VariantAnnotation(rsid=rsid)

        if self.parallel_requests:
            self._fetch_variant_parallel(annotation, include_phewas)
        else:
            self._fetch_variant_sequential(annotation, include_phewas)

        # Cache result
        if self.use_cache:
            self._variant_cache[cache_key] = annotation

        return annotation

    def _fetch_variant_parallel(self, annotation: VariantAnnotation, include_phewas: bool):
        """Fetch variant data in parallel"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.clinvar.get_variant_by_rsid, annotation.rsid): "clinvar",
                executor.submit(self.ensembl.get_variant_by_rsid, annotation.rsid): "ensembl",
            }

            if include_phewas:
                futures[executor.submit(self.epigraphdb.get_phewas, annotation.rsid)] = "phewas"

            for future in as_completed(futures):
                source = futures[future]
                try:
                    result = future.result()
                    self._merge_variant_result(annotation, source, result)
                except Exception as e:
                    logger.warning(f"Failed to fetch {source} for {annotation.rsid}: {e}")

    def _fetch_variant_sequential(self, annotation: VariantAnnotation, include_phewas: bool):
        """Fetch variant data sequentially"""
        # ClinVar
        try:
            clinvar_result = self.clinvar.get_variant_by_rsid(annotation.rsid)
            self._merge_variant_result(annotation, "clinvar", clinvar_result)
        except Exception as e:
            logger.warning(f"ClinVar fetch failed: {e}")

        # Ensembl
        try:
            ensembl_result = self.ensembl.get_variant_by_rsid(annotation.rsid)
            self._merge_variant_result(annotation, "ensembl", ensembl_result)
        except Exception as e:
            logger.warning(f"Ensembl fetch failed: {e}")

        # PheWAS
        if include_phewas:
            try:
                phewas_result = self.epigraphdb.get_phewas(annotation.rsid)
                self._merge_variant_result(annotation, "phewas", phewas_result)
            except Exception as e:
                logger.warning(f"PheWAS fetch failed: {e}")

    def _merge_variant_result(self, annotation: VariantAnnotation, source: str, result: Any):
        """Merge API result into annotation"""
        if result is None:
            return

        if source == "clinvar" and isinstance(result, ClinVarVariant):
            annotation.clinical_significance = result.clinical_significance
            annotation.clinvar_review_status = result.review_status
            annotation.clinvar_condition = result.condition
            if result.gene_symbol:
                annotation.gene_symbol = result.gene_symbol

        elif source == "ensembl" and isinstance(result, VariantInfo):
            annotation.chromosome = result.chromosome
            annotation.position = result.position
            annotation.alleles = result.alleles
            annotation.maf = result.maf
            annotation.consequence_type = result.consequence_type
            if result.gene_symbol:
                annotation.gene_symbol = result.gene_symbol

        elif source == "phewas" and isinstance(result, list):
            annotation.associated_traits = [r.trait for r in result[:10]]
            if result:
                annotation.best_gwas_pvalue = result[0].p_value

    def get_gene_annotation(
        self,
        gene_symbol: str,
        include_network: bool = True,
        include_enrichment: bool = True,
        include_mr: bool = True,
        disease_context: Optional[List[str]] = None
    ) -> GeneAnnotation:
        """
        Get complete gene annotation from multiple sources

        Args:
            gene_symbol: Gene symbol (e.g., "TYK2")
            include_network: Include STRING-DB network analysis
            include_enrichment: Include Enrichr pathway enrichment
            include_mr: Include MR evidence from EpiGraphDB
            disease_context: Disease keywords for context

        Returns:
            GeneAnnotation object
        """
        # Check cache
        cache_key = f"{gene_symbol}_{include_network}_{include_enrichment}_{include_mr}"
        if self.use_cache and cache_key in self._gene_cache:
            return self._gene_cache[cache_key]

        annotation = GeneAnnotation(gene_symbol=gene_symbol)

        # Basic gene info (Ensembl)
        gene_info = self.ensembl.get_gene_by_symbol(gene_symbol)
        if gene_info:
            annotation.gene_id = gene_info.gene_id
            annotation.description = gene_info.description
            annotation.biotype = gene_info.biotype
            annotation.chromosome = gene_info.chromosome
            annotation.start = gene_info.start
            annotation.end = gene_info.end
            annotation.length = gene_info.length

        # Disease associations (Open Targets)
        associations = self.open_targets.get_gene_disease_associations(gene_symbol, limit=10)
        annotation.disease_associations = associations
        if associations:
            annotation.top_disease = associations[0].disease_name
            annotation.top_disease_score = associations[0].overall_score

        # Drug targets
        drugs = self.open_targets.get_drug_targets(gene_symbol)
        annotation.targeting_drugs = drugs

        # Network analysis
        if include_network:
            try:
                centralities = self.string_db.compute_all_centralities([gene_symbol])
                if gene_symbol in centralities:
                    annotation.network_centrality = centralities[gene_symbol]

                neighbors = self.string_db.get_neighbors(gene_symbol, limit=10)
                annotation.interacting_genes = [n[0] for n in neighbors]
            except Exception as e:
                logger.warning(f"Network analysis failed for {gene_symbol}: {e}")

        # Pathway enrichment
        if include_enrichment:
            try:
                pathways = self.enrichr.get_pathway_enrichment([gene_symbol], top_n=5)
                annotation.pathway_enrichments = pathways
                if pathways:
                    annotation.top_pathway = pathways[0].term
            except Exception as e:
                logger.warning(f"Enrichment failed for {gene_symbol}: {e}")

        # MR evidence
        if include_mr:
            try:
                mr_results = self.epigraphdb.get_mr_eve_evidence(gene_symbol)
                annotation.mr_evidence = mr_results[:10]

                # Compute causal score
                if disease_context:
                    annotation.causal_score = self.epigraphdb.get_causal_evidence_score(
                        gene_symbol, disease_context
                    )
            except Exception as e:
                logger.warning(f"MR evidence fetch failed for {gene_symbol}: {e}")

        # Cache result
        if self.use_cache:
            self._gene_cache[cache_key] = annotation

        return annotation

    def get_gene_risk_score(
        self,
        gene_symbol: str,
        variants: List[str],
        disease_context: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive gene risk score for GRASS

        Args:
            gene_symbol: Gene symbol
            variants: List of variant rsIDs in this gene
            disease_context: Disease keywords

        Returns:
            Dictionary with component scores
        """
        # Get gene annotation
        gene_ann = self.get_gene_annotation(
            gene_symbol,
            disease_context=disease_context
        )

        # Get variant annotations
        variant_scores = []
        for rsid in variants:
            var_ann = self.get_variant_annotation(rsid)
            variant_scores.append(var_ann.get_risk_score())

        # Compute aggregate scores
        gene_length = gene_ann.length or 27000  # Default to average human gene length

        # SNP-level score
        snp_score = sum(variant_scores) / len(variant_scores) if variant_scores else 0.0

        # GWAS score from disease associations
        gwas_score = gene_ann.top_disease_score

        # Structure score
        structure_score = gene_ann.get_structure_score()

        # Enrichment score
        enrichment_score = gene_ann.get_enrichment_score()

        # Causal score
        causal_score = gene_ann.causal_score

        # GRASS formula: (SNP_score / L) * (1 + λ * GWAS)
        lambda_weight = 10.0
        grass_raw = (snp_score * len(variants) / (gene_length / 1000)) * (1 + lambda_weight * gwas_score)
        grass_normalized = min(grass_raw, 1.0)  # Normalize to 0-1

        return {
            "gene_symbol": gene_symbol,
            "gene_length": gene_length,
            "variant_count": len(variants),
            "snp_score": snp_score,
            "gwas_score": gwas_score,
            "structure_score": structure_score,
            "enrichment_score": enrichment_score,
            "causal_score": causal_score,
            "grass_score": grass_normalized
        }

    def get_rocket_scores(
        self,
        genes: List[str],
        disease_context: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute ROCKET component scores for multiple genes

        Args:
            genes: List of gene symbols
            disease_context: Disease keywords for context

        Returns:
            Dictionary mapping gene to component scores
        """
        results = {}

        for gene in genes:
            try:
                gene_ann = self.get_gene_annotation(
                    gene,
                    include_network=True,
                    include_enrichment=True,
                    include_mr=True,
                    disease_context=disease_context
                )

                results[gene] = {
                    "S_R": gene_ann.top_disease_score,  # Risk score
                    "S_S": gene_ann.get_structure_score(),  # Structure score
                    "S_E": gene_ann.get_enrichment_score(),  # Enrichment score
                    "S_M": 0.0,  # Semantic score (requires LLM)
                    "S_C": gene_ann.causal_score,  # Causal score
                }
            except Exception as e:
                logger.error(f"Failed to get ROCKET scores for {gene}: {e}")
                results[gene] = {"S_R": 0, "S_S": 0, "S_E": 0, "S_M": 0, "S_C": 0}

        return results

    def validate_causal_relationship(
        self,
        gene: str,
        disease: str
    ) -> Tuple[float, List[MREvidence]]:
        """
        Validate a causal gene-disease relationship using MR evidence

        Args:
            gene: Gene symbol
            disease: Disease name

        Returns:
            Tuple of (causal_score, supporting_evidence)
        """
        # Get MR evidence
        mr_results = self.epigraphdb.get_mr_eve_evidence(gene)

        # Filter for relevant disease
        disease_lower = disease.lower()
        relevant = [
            mr for mr in mr_results
            if disease_lower in mr.outcome.lower()
        ]

        if not relevant:
            # Try broader search
            mr_results_broad = self.epigraphdb.get_mr_evidence(gene)
            relevant = [
                mr for mr in mr_results_broad
                if disease_lower in mr.outcome.lower()
            ]

        if not relevant:
            return 0.0, []

        # Compute aggregate score
        scores = [mr.get_causal_score() for mr in relevant]
        causal_score = max(scores) if scores else 0.0

        return causal_score, relevant

    def clear_cache(self):
        """Clear all caches"""
        self._variant_cache.clear()
        self._gene_cache.clear()
        self.ensembl._cache.clear()
        self.clinvar._cache.clear()


# Convenience functions
def get_variant_risk_score(rsid: str) -> float:
    """Quick variant risk score lookup"""
    client = UnifiedBiomedicalClient()
    annotation = client.get_variant_annotation(rsid)
    return annotation.get_risk_score()


def get_gene_disease_score(gene: str, disease: str) -> float:
    """Quick gene-disease score lookup"""
    client = UnifiedBiomedicalClient()
    causal_score, _ = client.validate_causal_relationship(gene, disease)
    return causal_score


# Example usage
if __name__ == "__main__":
    print("Unified Biomedical API Client")
    print("=" * 60)

    client = UnifiedBiomedicalClient()

    # Test variant annotation
    print("\nVariant Annotation for rs34536443:")
    var_ann = client.get_variant_annotation("rs34536443")
    print(f"  Gene: {var_ann.gene_symbol}")
    print(f"  Clinical Significance: {var_ann.clinical_significance}")
    print(f"  Consequence: {var_ann.consequence_type}")
    print(f"  Risk Score: {var_ann.get_risk_score():.3f}")

    # Test gene annotation
    print("\n" + "=" * 60)
    print("\nGene Annotation for TYK2:")
    gene_ann = client.get_gene_annotation("TYK2", disease_context=["rheumatoid", "arthritis"])
    print(f"  Ensembl ID: {gene_ann.gene_id}")
    print(f"  Length: {gene_ann.length:,} bp")
    print(f"  Top Disease: {gene_ann.top_disease}")
    print(f"  Disease Score: {gene_ann.top_disease_score:.3f}")
    print(f"  Structure Score: {gene_ann.get_structure_score():.3f}")
    print(f"  Causal Score: {gene_ann.causal_score:.3f}")
    if gene_ann.interacting_genes:
        print(f"  Top Interactors: {', '.join(gene_ann.interacting_genes[:5])}")

    # Test ROCKET scores
    print("\n" + "=" * 60)
    print("\nROCKET Scores:")
    rocket = client.get_rocket_scores(["TYK2", "JAK1", "STAT3"], disease_context=["autoimmune"])
    for gene, scores in rocket.items():
        print(f"\n{gene}:")
        for component, score in scores.items():
            print(f"  {component}: {score:.3f}")

    # Test causal validation
    print("\n" + "=" * 60)
    print("\nCausal Validation for TYK2 -> Rheumatoid Arthritis:")
    score, evidence = client.validate_causal_relationship("TYK2", "rheumatoid arthritis")
    print(f"  Causal Score: {score:.3f}")
    print(f"  Evidence Count: {len(evidence)}")
    for mr in evidence[:3]:
        print(f"    - {mr.outcome}: beta={mr.beta:.3f}, p={mr.p_value:.2e}")
