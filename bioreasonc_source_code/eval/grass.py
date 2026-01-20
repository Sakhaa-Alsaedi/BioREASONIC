"""
GRASS: Genetic Risk Aggregate Score (WGRS)

Implements the Weighted Gene Risk Score for SNP-to-gene risk aggregation
with gene length normalization and GWAS disease association weighting.

Reference: Chapter 3, Section 3.4.1

Now includes API integration for fetching real clinical annotations:
- ClinVar: Clinical significance
- Open Targets: GWAS disease associations
- Ensembl: Gene length and variant consequences
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import math
import logging

logger = logging.getLogger(__name__)


class ClinicalSignificance(Enum):
    """ClinVar clinical significance categories"""
    PATHOGENIC = "pathogenic"
    LIKELY_PATHOGENIC = "likely_pathogenic"
    VUS = "uncertain_significance"
    LIKELY_BENIGN = "likely_benign"
    BENIGN = "benign"


class FunctionalImpact(Enum):
    """Variant functional impact levels"""
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    MODIFIER = "MODIFIER"


# Score mappings from PDF Table 3.7
CLINVAR_SCORES = {
    ClinicalSignificance.PATHOGENIC: 1.0,
    ClinicalSignificance.LIKELY_PATHOGENIC: 0.8,
    ClinicalSignificance.VUS: 0.3,
    ClinicalSignificance.LIKELY_BENIGN: 0.1,
    ClinicalSignificance.BENIGN: 0.0
}

IMPACT_SCORES = {
    FunctionalImpact.HIGH: 1.0,
    FunctionalImpact.MODERATE: 0.6,
    FunctionalImpact.LOW: 0.3,
    FunctionalImpact.MODIFIER: 0.1
}


@dataclass
class SNPRisk:
    """
    SNP-level risk contribution using Mendelian Randomization

    Formula (MR-based with CAUSALdb2 real scores):
    SNP_Risk = w_causal × causal_score + w_beta × β_norm + w_gwas × S_GWAS

    Components (REAL DATA from CAUSALdb2):
    - causal_score: MR causal evidence [-0.86, 0.99]
    - evidence_score: GWAS significance/replication [0, 0.37]
    - risk_score: Combined risk assessment [0, 0.58]
    - beta: Effect size (positive = risk, negative = protective)
    - pvalue: GWAS p-value

    Only RISK SNPs (beta > 0) contribute to gene risk score.
    """
    rsid: str
    gene: str
    chromosome: str
    position: int
    disease: Optional[str] = None  # Disease association (e.g., 'AD', 'T2D')

    # === CAUSALdb2 SNP-LEVEL SCORES (REAL DATA) ===
    causal_score: Optional[float] = None      # MR causal evidence [-0.86, 0.99]
    evidence_score: Optional[float] = None    # GWAS significance [0, 0.37]
    risk_score_raw: Optional[float] = None    # Combined risk [0, 0.58]

    # === GWAS Data (REAL) ===
    beta: Optional[float] = None              # Effect size (+ = risk, - = protective)
    p_value: Optional[float] = None           # GWAS p-value
    maf: float = 0.0                          # Minor allele frequency

    # === Optional annotations ===
    clinical_significance: Optional[ClinicalSignificance] = None
    functional_impact: Optional[FunctionalImpact] = None
    odds_ratio: Optional[float] = None

    # Quality metrics (optional)
    qual: float = 0.0
    gq: float = 0.0
    dp: float = 0.0

    # Fine-mapping PIPs (optional - if available)
    pip_abf: Optional[float] = None
    pip_finemap: Optional[float] = None
    pip_susie: Optional[float] = None
    pip_paintor: Optional[float] = None
    pip_caviarbf: Optional[float] = None
    pip_polyfun: Optional[float] = None

    # Pre-calculated gene-level causal score (legacy)
    gene_causal_score: Optional[float] = None

    def get_causal_score(self) -> float:
        """
        Get causal score from available sources (REAL DATA ONLY):
        1. Pre-calculated gene-level causal score (from max_combined_score)
        2. Maximum PIP from fine-mapping methods (CAUSALdb)
        3. Returns 0 if no real data available (NO DEFAULT)

        Returns:
            Causal score in [0, 1], 0 if no real data
        """
        # Priority 1: Pre-calculated gene-level causal score (REAL)
        if self.gene_causal_score is not None:
            return self.gene_causal_score

        # Priority 2: Fine-mapping PIPs from CAUSALdb (REAL)
        pips = [
            self.pip_abf,
            self.pip_finemap,
            self.pip_susie,
            self.pip_paintor,
            self.pip_caviarbf,
            self.pip_polyfun
        ]

        valid_pips = [p for p in pips if p is not None]

        if valid_pips:
            return max(valid_pips)

        # NO DEFAULT - return 0 if no real data
        return 0.0

    def get_mean_causal_score(self) -> float:
        """
        Get mean fine-mapping causal score across methods (REAL DATA ONLY)

        Returns:
            Mean causal score in [0, 1], 0 if no real data
        """
        # First check pre-calculated gene causal score
        if self.gene_causal_score is not None:
            return self.gene_causal_score

        pips = [
            self.pip_abf,
            self.pip_finemap,
            self.pip_susie,
            self.pip_paintor,
            self.pip_caviarbf,
            self.pip_polyfun
        ]

        valid_pips = [p for p in pips if p is not None]

        if not valid_pips:
            return 0.0  # NO DEFAULT - return 0 if no real data

        return sum(valid_pips) / len(valid_pips)

    def get_gwas_score(self, max_beta: float = 1.0) -> float:
        """
        Get GWAS score from p-value (REAL DATA ONLY)

        Formula:
        - S_GWAS = min(-log10(p-value) / 10, 1.0)
        - Returns 0 if no p-value available (NO DEFAULT)

        Args:
            max_beta: Maximum expected beta for normalization (default 1.0)

        Returns:
            GWAS score in [0, 1], 0 if no real data
        """
        has_effect = self.beta is not None or self.odds_ratio is not None
        has_pvalue = self.p_value is not None

        if not has_pvalue:
            return 0.0  # NO DEFAULT - return 0 if no real data

        # Calculate significance score from p-value
        # -log10(p) / 10, so p=1e-10 gives score of 1.0
        # This differentiates SNPs based on GWAS significance
        if self.p_value > 0:
            significance_score = min(-math.log10(self.p_value) / 10.0, 1.0)
        else:
            significance_score = 1.0  # p=0 treated as highly significant

        # If no effect size, use significance alone
        if not has_effect:
            return significance_score

        # Calculate effect score from beta or odds ratio
        if self.beta is not None:
            effect_score = min(abs(self.beta) / max_beta, 1.0)
        elif self.odds_ratio is not None and self.odds_ratio > 0:
            # Convert OR to log scale: |log(OR)|
            effect_score = min(abs(math.log(self.odds_ratio)), 1.0)
        else:
            effect_score = 0.5  # Unknown effect

        # Combined GWAS score (geometric mean)
        s_gwas = math.sqrt(effect_score * significance_score)

        return s_gwas

    def is_risk_snp(self) -> bool:
        """Check if this is a RISK SNP (positive beta = increases disease risk)."""
        return self.beta is not None and self.beta > 0

    def compute_risk_score(self) -> float:
        """
        Get SNP risk score from CAUSALdb2 pre-calculated scores (REAL DATA)

        CAUSALdb2 scores are ALREADY CALCULATED:
        - causal_score: Mean of fine-mapping PIPs (ABF, FINEMAP, PAINTOR,
                       CAVIARBF, SUSIE, POLYFUN_FINEMAP, POLYFUN_SUSIE)
        - evidence_score: Normalized -log10(pvalue) [0-1]
        - risk_score_raw: 0.6 × evidence_score + 0.4 × causal_score

        Returns:
            risk_score_raw from CAUSALdb2, or 0 if not a risk SNP
        """
        # Only RISK SNPs (beta > 0) contribute to gene risk
        if not self.is_risk_snp():
            return 0.0

        # Use pre-calculated risk_score from CAUSALdb2
        if self.risk_score_raw is not None:
            return self.risk_score_raw

        # Fallback: calculate from components if risk_score_raw not available
        # risk_score = 0.3 × evidence_score + 0.7 × causal_score
        # (Emphasize causal fine-mapping PIPs over GWAS p-value)
        evidence = self.evidence_score if self.evidence_score is not None else 0.0
        causal = max(self.causal_score, 0.0) if self.causal_score is not None else 0.0

        return 0.3 * evidence + 0.7 * causal

    def get_effect_weight(self) -> float:
        """Get effect size weight from GWAS statistics"""
        if self.odds_ratio is not None and self.odds_ratio > 0:
            # Convert OR to effect weight
            return abs(math.log(self.odds_ratio))
        elif self.beta is not None:
            return abs(self.beta)
        return 1.0


@dataclass
class GeneScore:
    """
    Gene-level risk score aggregated from SNPs

    Formula (Eq. 3.6):
    Gene_Score_g = (Σ SNP_Risk_i) / (L_g / 1000)

    Final WGRS (Eq. 3.7):
    WGRS_g = Gene_Score_g × (1 + λ × GWAS_Score_g)
    """
    gene_symbol: str
    gene_id: Optional[str] = None
    chromosome: Optional[str] = None

    # Gene length in base pairs
    gene_length: int = 1000  # Default 1kb

    # SNP contributions
    snp_risks: List[SNPRisk] = field(default_factory=list)

    # GWAS association
    gwas_score: float = 0.0  # Disease association from GWAS/Open Targets

    # Computed scores
    raw_score: float = 0.0
    normalized_score: float = 0.0
    wgrs: float = 0.0

    def add_snp(self, snp: SNPRisk):
        """Add SNP to gene"""
        self.snp_risks.append(snp)

    def compute_gene_score(self, gwas_weight: float = 0.3) -> float:
        """
        Compute gene-level risk score (WGRS) - NORMALIZED to [0, 1]

        Uses MAX of RISK SNP scores (best evidence for the gene).
        Non-risk SNPs (beta <= 0) are excluded.

        Formula:
            base_score = MAX(risk_SNP_scores)  # Best SNP evidence
            WGRS = base × (1 + gwas_weight × gwas_score)

        Args:
            gwas_weight: Weight for GWAS contribution (default 0.3)

        Returns:
            WGRS score in [0, 1] range
        """
        if not self.snp_risks:
            self.wgrs = 0.0
            return 0.0

        # Only count RISK SNPs (beta > 0) for gene score
        risk_snp_scores = []
        for snp in self.snp_risks:
            if snp.is_risk_snp():  # Only positive beta
                score = snp.compute_risk_score()
                if score > 0:  # Only non-zero scores
                    risk_snp_scores.append(score)

        if not risk_snp_scores:
            self.wgrs = 0.0
            return 0.0

        # MAX of RISK SNP scores (best evidence)
        self.raw_score = sum(risk_snp_scores)
        self.normalized_score = max(risk_snp_scores)  # MAX - best SNP evidence

        # Apply GWAS boost
        base = self.normalized_score
        self.wgrs = base * (1.0 + gwas_weight * self.gwas_score)

        # Ensure [0, 1] bounds
        self.wgrs = max(0.0, min(1.0, self.wgrs))

        return self.wgrs

    def get_snp_count(self) -> int:
        """Get number of SNPs in gene"""
        return len(self.snp_risks)

    def get_high_impact_snps(self) -> List[SNPRisk]:
        """Get SNPs with HIGH functional impact"""
        return [
            snp for snp in self.snp_risks
            if snp.functional_impact == FunctionalImpact.HIGH
        ]

    def get_snps_by_disease(self) -> Dict[str, List[SNPRisk]]:
        """Get SNPs grouped by disease"""
        disease_snps: Dict[str, List[SNPRisk]] = {}
        for snp in self.snp_risks:
            disease = snp.disease or 'unknown'
            if disease not in disease_snps:
                disease_snps[disease] = []
            disease_snps[disease].append(snp)
        return disease_snps

    def compute_disease_scores(self, gwas_weight: float = 0.3) -> Dict[str, float]:
        """
        Compute WGRS separately for each disease - NORMALIZED to [0, 1]

        Uses MAX of RISK SNP scores only (beta > 0).

        Returns:
            Dictionary of disease -> WGRS score (all in [0, 1])
        """
        disease_snps = self.get_snps_by_disease()
        disease_scores = {}

        for disease, snps in disease_snps.items():
            # Only count RISK SNPs (beta > 0)
            risk_snp_scores = []
            for snp in snps:
                if snp.is_risk_snp():
                    score = snp.compute_risk_score()
                    if score > 0:
                        risk_snp_scores.append(score)

            # MAX of RISK SNP scores (best evidence)
            base = max(risk_snp_scores) if risk_snp_scores else 0.0

            # Apply GWAS boost
            wgrs = base * (1.0 + gwas_weight * self.gwas_score)

            # Ensure [0, 1] bounds
            disease_scores[disease] = max(0.0, min(1.0, wgrs))

        return disease_scores

    def get_diseases(self) -> List[str]:
        """Get list of diseases this gene is associated with"""
        diseases = set()
        for snp in self.snp_risks:
            if snp.disease:
                diseases.add(snp.disease)
        return list(diseases)

    def is_multi_disease(self) -> bool:
        """Check if gene is associated with multiple diseases"""
        return len(self.get_diseases()) > 1


@dataclass
class GeneRiskVector:
    """
    Multi-disease risk profile for a gene

    Stores per-disease WGRS scores and aggregation methods for genes
    that are associated with multiple diseases.

    All scores are NORMALIZED to [0, 1] range.

    Example for APOE (shared between AD and T2D):
    {
        'gene': 'APOE',
        'risk_vector': {'AD': 0.92, 'T2D': 0.38},
        'shared_score': 0.65,
        'is_multi_disease': True
    }
    """
    gene_symbol: str
    gene_id: Optional[str] = None
    chromosome: Optional[str] = None
    gene_length: int = 1000

    # Per-disease RAW WGRS scores (before normalization)
    disease_scores_raw: Dict[str, float] = field(default_factory=dict)

    # Per-disease NORMALIZED scores [0, 1]
    disease_scores: Dict[str, float] = field(default_factory=dict)

    # Per-disease SNP counts
    disease_snp_counts: Dict[str, int] = field(default_factory=dict)

    # Shared SNPs (present in multiple diseases)
    shared_snp_count: int = 0
    shared_snps: List[str] = field(default_factory=list)

    # Aggregated NORMALIZED scores [0, 1]
    max_score: float = 0.0
    mean_score: float = 0.0
    weighted_score: float = 0.0  # Weighted by SNP count

    def get_diseases(self) -> List[str]:
        """Get list of diseases"""
        return list(self.disease_scores.keys()) or list(self.disease_scores_raw.keys())

    def is_multi_disease(self) -> bool:
        """Check if gene has multiple disease associations"""
        return len(self.get_diseases()) > 1

    def compute_aggregations(self):
        """Compute all aggregation methods from normalized scores"""
        if not self.disease_scores:
            return

        scores = list(self.disease_scores.values())

        # Maximum score (conservative)
        self.max_score = max(scores)

        # Mean score (balanced)
        self.mean_score = sum(scores) / len(scores)

        # Weighted by SNP count
        total_snps = sum(self.disease_snp_counts.values())
        if total_snps > 0:
            self.weighted_score = sum(
                self.disease_scores.get(d, 0) * self.disease_snp_counts.get(d, 0)
                for d in self.disease_scores
            ) / total_snps
        else:
            self.weighted_score = self.mean_score

    def get_top_disease(self) -> Optional[str]:
        """Get disease with highest risk score"""
        if not self.disease_scores:
            return None
        return max(self.disease_scores, key=self.disease_scores.get)

    def to_dict(self) -> Dict:
        """Convert to dictionary for export"""
        return {
            'gene': self.gene_symbol,
            'gene_id': self.gene_id,
            'chromosome': self.chromosome,
            'gene_length': self.gene_length,
            'diseases': self.get_diseases(),
            'is_multi_disease': self.is_multi_disease(),
            'disease_scores': self.disease_scores,
            'disease_scores_raw': self.disease_scores_raw,
            'disease_snp_counts': self.disease_snp_counts,
            'shared_snp_count': self.shared_snp_count,
            'shared_snps': self.shared_snps,
            'max_score': self.max_score,
            'mean_score': self.mean_score,
            'weighted_score': self.weighted_score,
            'top_disease': self.get_top_disease()
        }


class GRASSCalculator:
    """
    GRASS/WGRS Calculator

    Computes Genetic Risk Aggregate Scores for genes based on
    SNP-level risk contributions. All scores are NORMALIZED to [0, 1].

    Key changes:
    - Uses MEAN of SNP risks (not sum) to preserve [0,1] range
    - GWAS boost applied as weighted average, not multiplier

    UPDATED: Now includes fine-mapping causal scores from CAUSALdb.
    """

    def __init__(
        self,
        gwas_weight: float = 0.3,  # Weight for GWAS contribution [0, 1]
        qual_threshold: float = 45.0,
        gq_threshold: float = 60.0,
        dp_threshold: float = 40.0,
        causaldb_path: Optional[str] = None  # Path to CAUSALdb credible_set.txt
    ):
        """
        Initialize GRASS calculator

        Args:
            gwas_weight: Weight for GWAS contribution (default 0.3, range [0, 1])
            qual_threshold: Minimum variant quality
            gq_threshold: Minimum genotype quality
            dp_threshold: Minimum read depth
            causaldb_path: Optional path to CAUSALdb credible_set.txt for PIP lookup
        """
        self.gwas_weight = gwas_weight
        self.qual_threshold = qual_threshold
        self.gq_threshold = gq_threshold
        self.dp_threshold = dp_threshold

        # Gene scores cache
        self.gene_scores: Dict[str, GeneScore] = {}

        # CAUSALdb lookup tables
        # Simple lookup: rsid -> PIPs (for backward compatibility)
        self.causaldb_pips: Dict[str, Dict[str, float]] = {}
        # Disease-linked lookup: (rsid, meta_id) -> full data including disease
        self.causaldb_snp_disease: Dict[Tuple[str, str], Dict] = {}
        # Meta lookup: meta_id -> trait info
        self.causaldb_meta: Dict[str, Dict[str, str]] = {}

        # Pre-calculated gene causal scores (from previous GRASS runs)
        # Maps (gene, disease) -> causal_score
        self.gene_causal_scores: Dict[Tuple[str, str], float] = {}

        if causaldb_path:
            self.load_causaldb(causaldb_path)

    def load_causaldb_meta(self, meta_path: str) -> int:
        """
        Load CAUSALdb trait/disease metadata from meta.txt

        Args:
            meta_path: Path to CAUSALdb meta.txt

        Returns:
            Number of traits loaded
        """
        import csv

        count = 0
        try:
            with open(meta_path, 'r') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    meta_id = row.get('meta_id', '')
                    if not meta_id:
                        continue

                    self.causaldb_meta[meta_id] = {
                        'trait': row.get('trait', ''),
                        'mesh_term': row.get('mesh_term', ''),
                        'mesh_id': row.get('mesh_id', ''),
                        'sample_size': row.get('sample_size', ''),
                        'n_case': row.get('n_case', ''),
                        'n_control': row.get('n_control', ''),
                        'population': row.get('popu', ''),
                        'author': row.get('author', ''),
                        'pmid': row.get('pmid', ''),
                        'year': row.get('year', ''),
                    }
                    count += 1

            logger.info(f"Loaded {count} traits from CAUSALdb meta")
        except Exception as e:
            logger.warning(f"Failed to load CAUSALdb meta: {e}")

        return count

    def load_causaldb(
        self,
        filepath: str,
        meta_path: Optional[str] = None,
        filter_diseases: Optional[List[str]] = None,
        filter_meta_ids: Optional[List[str]] = None,
        max_variants: Optional[int] = None
    ) -> int:
        """
        Load CAUSALdb fine-mapping PIPs from credible_set.txt with disease linkage

        Args:
            filepath: Path to CAUSALdb credible_set.txt
            meta_path: Optional path to meta.txt for disease info
            filter_diseases: Optional list of disease/trait names to filter (substring match)
            filter_meta_ids: Optional list of meta_ids to include
            max_variants: Optional limit on number of variants to load

        Returns:
            Number of variants loaded
        """
        import csv

        # Load meta first if provided
        if meta_path:
            self.load_causaldb_meta(meta_path)

        # Build set of allowed meta_ids for filtering
        allowed_meta_ids = None
        if filter_diseases and self.causaldb_meta:
            allowed_meta_ids = set()
            for meta_id, info in self.causaldb_meta.items():
                trait = info.get('trait', '').lower()
                mesh_term = info.get('mesh_term', '').lower()
                for disease in filter_diseases:
                    if disease.lower() in trait or disease.lower() in mesh_term:
                        allowed_meta_ids.add(meta_id)
            logger.info(f"Filtering to {len(allowed_meta_ids)} meta_ids matching diseases: {filter_diseases}")
        elif filter_meta_ids:
            allowed_meta_ids = set(filter_meta_ids)

        def safe_float(val):
            try:
                return float(val) if val and val.strip() else None
            except:
                return None

        count = 0
        try:
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    if max_variants and count >= max_variants:
                        break

                    rsid = row.get('rsid', '')
                    meta_id = row.get('meta_id', '')

                    if not rsid:
                        continue

                    # Apply meta_id filter if specified
                    if allowed_meta_ids and meta_id not in allowed_meta_ids:
                        continue

                    # Build PIP data
                    pip_data = {
                        'abf': safe_float(row.get('abf')),
                        'finemap': safe_float(row.get('finemap')),
                        'susie': safe_float(row.get('susie')),
                        'paintor': safe_float(row.get('paintor')),
                        'caviarbf': safe_float(row.get('caviarbf')),
                        'polyfun': safe_float(row.get('polyfun_susie')),
                    }

                    # Build full SNP data with GWAS stats
                    snp_data = {
                        'rsid': rsid,
                        'chr': row.get('chr'),
                        'bp': safe_float(row.get('bp')),
                        'maf': safe_float(row.get('maf')),
                        'ea': row.get('ea'),
                        'nea': row.get('nea'),
                        'beta': safe_float(row.get('beta')),
                        'se': safe_float(row.get('se')),
                        'p': safe_float(row.get('p')),
                        'zscore': safe_float(row.get('zscore')),
                        'meta_id': meta_id,
                        'lead_snp': row.get('lead_snp'),
                        **pip_data
                    }

                    # Add disease info if available
                    if meta_id in self.causaldb_meta:
                        meta_info = self.causaldb_meta[meta_id]
                        snp_data['trait'] = meta_info.get('trait')
                        snp_data['mesh_term'] = meta_info.get('mesh_term')
                        snp_data['mesh_id'] = meta_info.get('mesh_id')

                    # Store in both lookup tables
                    # Simple lookup (may overwrite if SNP appears for multiple diseases)
                    self.causaldb_pips[rsid] = pip_data

                    # Disease-linked lookup (preserves all associations)
                    self.causaldb_snp_disease[(rsid, meta_id)] = snp_data

                    count += 1

            logger.info(f"Loaded {count} SNP-disease associations from CAUSALdb")
            if allowed_meta_ids:
                logger.info(f"Filtered to {len(allowed_meta_ids)} diseases/traits")

        except Exception as e:
            logger.warning(f"Failed to load CAUSALdb: {e}")

        return count

    def get_snps_for_disease(self, disease_name: str) -> List[Dict]:
        """
        Get all SNPs associated with a specific disease/trait

        Args:
            disease_name: Disease or trait name (substring match)

        Returns:
            List of SNP data dictionaries
        """
        results = []
        disease_lower = disease_name.lower()

        for (rsid, meta_id), snp_data in self.causaldb_snp_disease.items():
            trait = snp_data.get('trait', '').lower()
            mesh_term = snp_data.get('mesh_term', '').lower()

            if disease_lower in trait or disease_lower in mesh_term:
                results.append(snp_data)

        return results

    def get_diseases_for_snp(self, rsid: str) -> List[Dict]:
        """
        Get all diseases/traits associated with a specific SNP

        Args:
            rsid: SNP identifier

        Returns:
            List of disease/trait info dictionaries
        """
        results = []

        for (snp_rsid, meta_id), snp_data in self.causaldb_snp_disease.items():
            if snp_rsid == rsid:
                results.append({
                    'meta_id': meta_id,
                    'trait': snp_data.get('trait'),
                    'mesh_term': snp_data.get('mesh_term'),
                    'mesh_id': snp_data.get('mesh_id'),
                    'maf': snp_data.get('maf'),
                    'beta': snp_data.get('beta'),
                    'p': snp_data.get('p'),
                })

        return results

    def load_gene_causal_scores(
        self,
        ad_scores_path: Optional[str] = None,
        t2d_scores_path: Optional[str] = None,
        score_column: str = 'max_combined_score'
    ) -> int:
        """
        Load pre-calculated gene causal scores from gene_scores files.

        These scores are used as S_Causal in the GRASS formula instead of
        the default 0.5 when no fine-mapping PIPs are available.

        Args:
            ad_scores_path: Path to AD gene scores CSV
            t2d_scores_path: Path to T2D gene scores CSV
            score_column: Column to use as causal score (default: 'max_combined_score')

        Returns:
            Number of gene scores loaded
        """
        import pandas as pd

        count = 0

        if ad_scores_path:
            try:
                df = pd.read_csv(ad_scores_path)
                for _, row in df.iterrows():
                    gene = row['gene_name']
                    score = row[score_column]
                    if pd.notna(score) and score > 0:
                        self.gene_causal_scores[(gene, 'AD')] = float(score)
                        count += 1
                logger.info(f"Loaded {count} AD gene causal scores from {ad_scores_path}")
            except Exception as e:
                logger.warning(f"Failed to load AD gene scores: {e}")

        if t2d_scores_path:
            t2d_count = 0
            try:
                df = pd.read_csv(t2d_scores_path)
                for _, row in df.iterrows():
                    gene = row['gene_name']
                    score = row[score_column]
                    if pd.notna(score) and score > 0:
                        self.gene_causal_scores[(gene, 'T2D')] = float(score)
                        t2d_count += 1
                logger.info(f"Loaded {t2d_count} T2D gene causal scores from {t2d_scores_path}")
                count += t2d_count
            except Exception as e:
                logger.warning(f"Failed to load T2D gene scores: {e}")

        return count

    def get_gene_causal_score(self, gene: str, disease: str) -> Optional[float]:
        """
        Get pre-calculated causal score for a gene-disease pair.

        Args:
            gene: Gene symbol
            disease: Disease name (e.g., 'AD', 'T2D')

        Returns:
            Causal score if available, None otherwise
        """
        return self.gene_causal_scores.get((gene, disease))

    def get_pips_for_snp(self, rsid: str) -> Dict[str, Optional[float]]:
        """
        Get PIPs for a SNP from loaded CAUSALdb data

        Args:
            rsid: SNP identifier

        Returns:
            Dictionary with PIP values for each method
        """
        return self.causaldb_pips.get(rsid, {})

    def add_snp(
        self,
        gene_symbol: str,
        rsid: str,
        chromosome: str,
        position: int,
        maf: float = 0.0,
        odds_ratio: Optional[float] = None,
        beta: Optional[float] = None,
        p_value: Optional[float] = None,
        clinical_significance: Optional[str] = None,
        functional_impact: Optional[str] = None,
        qual: float = 100.0,
        gq: float = 100.0,
        dp: float = 100.0,
        gene_length: int = 10000,
        # Fine-mapping PIPs from CAUSALdb (NEW)
        pip_abf: Optional[float] = None,
        pip_finemap: Optional[float] = None,
        pip_susie: Optional[float] = None,
        pip_paintor: Optional[float] = None,
        pip_caviarbf: Optional[float] = None,
        pip_polyfun: Optional[float] = None,
        auto_lookup_pips: bool = True  # Auto-lookup from loaded CAUSALdb
    ):
        """
        Add SNP to gene for WGRS calculation

        Args:
            gene_symbol: Gene symbol
            rsid: SNP identifier
            chromosome: Chromosome
            position: Genomic position
            maf: Minor allele frequency
            odds_ratio: GWAS odds ratio
            beta: GWAS effect size
            p_value: GWAS p-value
            clinical_significance: ClinVar significance
            functional_impact: Functional impact level
            qual: Variant quality
            gq: Genotype quality
            dp: Read depth
            gene_length: Gene length in bp
            pip_abf: ABF posterior inclusion probability (NEW)
            pip_finemap: FINEMAP PIP (NEW)
            pip_susie: SuSiE PIP (NEW)
            pip_paintor: PAINTOR PIP (NEW)
            pip_caviarbf: CAVIARBF PIP (NEW)
            pip_polyfun: PolyFun PIP (NEW)
            auto_lookup_pips: Auto-lookup PIPs from loaded CAUSALdb if not provided
        """
        # Parse clinical significance
        clin_sig = None
        if clinical_significance:
            clin_map = {
                'pathogenic': ClinicalSignificance.PATHOGENIC,
                'likely_pathogenic': ClinicalSignificance.LIKELY_PATHOGENIC,
                'uncertain_significance': ClinicalSignificance.VUS,
                'vus': ClinicalSignificance.VUS,
                'likely_benign': ClinicalSignificance.LIKELY_BENIGN,
                'benign': ClinicalSignificance.BENIGN
            }
            clin_sig = clin_map.get(clinical_significance.lower())

        # Parse functional impact
        func_impact = None
        if functional_impact:
            impact_map = {
                'high': FunctionalImpact.HIGH,
                'moderate': FunctionalImpact.MODERATE,
                'low': FunctionalImpact.LOW,
                'modifier': FunctionalImpact.MODIFIER
            }
            func_impact = impact_map.get(functional_impact.lower())

        # Auto-lookup PIPs from CAUSALdb if enabled and not provided
        if auto_lookup_pips and rsid in self.causaldb_pips:
            pips = self.causaldb_pips[rsid]
            pip_abf = pip_abf if pip_abf is not None else pips.get('abf')
            pip_finemap = pip_finemap if pip_finemap is not None else pips.get('finemap')
            pip_susie = pip_susie if pip_susie is not None else pips.get('susie')
            pip_paintor = pip_paintor if pip_paintor is not None else pips.get('paintor')
            pip_caviarbf = pip_caviarbf if pip_caviarbf is not None else pips.get('caviarbf')
            pip_polyfun = pip_polyfun if pip_polyfun is not None else pips.get('polyfun')

        # Create SNP risk with causal scores
        snp = SNPRisk(
            rsid=rsid,
            gene=gene_symbol,
            chromosome=chromosome,
            position=position,
            maf=maf,
            odds_ratio=odds_ratio,
            beta=beta,
            p_value=p_value,
            clinical_significance=clin_sig,
            functional_impact=func_impact,
            qual=qual,
            gq=gq,
            dp=dp,
            # Fine-mapping PIPs (NEW)
            pip_abf=pip_abf,
            pip_finemap=pip_finemap,
            pip_susie=pip_susie,
            pip_paintor=pip_paintor,
            pip_caviarbf=pip_caviarbf,
            pip_polyfun=pip_polyfun
        )

        # Add to gene
        if gene_symbol not in self.gene_scores:
            self.gene_scores[gene_symbol] = GeneScore(
                gene_symbol=gene_symbol,
                chromosome=chromosome,
                gene_length=gene_length
            )

        self.gene_scores[gene_symbol].add_snp(snp)

    def set_gwas_score(self, gene_symbol: str, gwas_score: float):
        """Set GWAS disease association score for a gene"""
        if gene_symbol in self.gene_scores:
            self.gene_scores[gene_symbol].gwas_score = gwas_score

    def compute_all_scores(self) -> Dict[str, float]:
        """
        Compute WGRS for all genes

        Returns:
            Dictionary of gene symbol to WGRS score
        """
        results = {}
        for gene_symbol, gene_score in self.gene_scores.items():
            wgrs = gene_score.compute_gene_score(self.gwas_weight)
            results[gene_symbol] = wgrs
        return results

    def get_ranked_genes(self, top_n: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Get genes ranked by WGRS score

        Args:
            top_n: Return only top N genes (None for all)

        Returns:
            List of (gene_symbol, wgrs) tuples sorted by score
        """
        scores = self.compute_all_scores()
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if top_n:
            return ranked[:top_n]
        return ranked

    def get_gene_details(self, gene_symbol: str) -> Optional[Dict]:
        """
        Get detailed scoring breakdown for a gene

        Args:
            gene_symbol: Gene symbol

        Returns:
            Dictionary with scoring details
        """
        if gene_symbol not in self.gene_scores:
            return None

        gene = self.gene_scores[gene_symbol]
        gene.compute_gene_score(self.gwas_weight)

        snp_details = []
        for snp in gene.snp_risks:
            snp_details.append({
                'rsid': snp.rsid,
                'risk_score': snp.compute_risk_score(),
                'maf': snp.maf,
                'odds_ratio': snp.odds_ratio,
                'clinical_significance': snp.clinical_significance.value if snp.clinical_significance else None,
                'functional_impact': snp.functional_impact.value if snp.functional_impact else None
            })

        return {
            'gene_symbol': gene_symbol,
            'gene_length': gene.gene_length,
            'snp_count': gene.get_snp_count(),
            'raw_score': gene.raw_score,
            'normalized_score': gene.normalized_score,
            'gwas_score': gene.gwas_score,
            'wgrs': gene.wgrs,
            'snps': snp_details
        }

    def normalize_scores(self) -> Dict[str, float]:
        """
        Normalize WGRS scores to [0, 1] range

        Returns:
            Dictionary of gene symbol to normalized score
        """
        scores = self.compute_all_scores()
        if not scores:
            return {}

        min_score = min(scores.values())
        max_score = max(scores.values())

        if max_score == min_score:
            return {gene: 0.5 for gene in scores}

        return {
            gene: (score - min_score) / (max_score - min_score)
            for gene, score in scores.items()
        }

    # ==================== MULTI-DISEASE METHODS ====================

    def add_snp_for_disease(
        self,
        disease: str,
        gene_symbol: str,
        rsid: str,
        chromosome: str,
        position: int,
        # CAUSALdb2 SNP-level scores (REAL DATA)
        causal_score: Optional[float] = None,      # MR causal evidence
        evidence_score: Optional[float] = None,    # GWAS significance
        risk_score_raw: Optional[float] = None,    # Combined risk
        # GWAS data
        beta: Optional[float] = None,              # Effect size
        p_value: Optional[float] = None,           # P-value
        maf: float = 0.0,
        odds_ratio: Optional[float] = None,
        # Optional annotations
        clinical_significance: Optional[str] = None,
        functional_impact: Optional[str] = None,
        gene_length: int = 10000,
        **kwargs
    ):
        """
        Add SNP with disease association using CAUSALdb2 real scores

        Args:
            disease: Disease name (e.g., 'AD', 'T2D')
            gene_symbol: Gene symbol
            rsid: SNP identifier
            chromosome: Chromosome
            position: Genomic position
            causal_score: MR causal evidence from CAUSALdb2 [-0.86, 0.99]
            evidence_score: GWAS significance from CAUSALdb2 [0, 0.37]
            risk_score_raw: Combined risk from CAUSALdb2 [0, 0.58]
            beta: Effect size (positive = risk SNP)
            p_value: GWAS p-value
        """
        # Parse clinical significance
        clin_sig = None
        if clinical_significance:
            clin_map = {
                'pathogenic': ClinicalSignificance.PATHOGENIC,
                'likely_pathogenic': ClinicalSignificance.LIKELY_PATHOGENIC,
                'uncertain_significance': ClinicalSignificance.VUS,
                'vus': ClinicalSignificance.VUS,
                'likely_benign': ClinicalSignificance.LIKELY_BENIGN,
                'benign': ClinicalSignificance.BENIGN
            }
            clin_sig = clin_map.get(clinical_significance.lower())

        # Parse functional impact
        func_impact = None
        if functional_impact:
            impact_map = {
                'high': FunctionalImpact.HIGH,
                'moderate': FunctionalImpact.MODERATE,
                'low': FunctionalImpact.LOW,
                'modifier': FunctionalImpact.MODIFIER
            }
            func_impact = impact_map.get(functional_impact.lower())

        # Auto-lookup PIPs from CAUSALdb
        pip_abf = pip_finemap = pip_susie = pip_paintor = pip_caviarbf = pip_polyfun = None
        if rsid in self.causaldb_pips:
            pips = self.causaldb_pips[rsid]
            pip_abf = pips.get('abf')
            pip_finemap = pips.get('finemap')
            pip_susie = pips.get('susie')
            pip_paintor = pips.get('paintor')
            pip_caviarbf = pips.get('caviarbf')
            pip_polyfun = pips.get('polyfun')

        # Lookup pre-calculated gene causal score
        gene_causal_score = self.get_gene_causal_score(gene_symbol, disease)

        # Create SNP with disease association and CAUSALdb2 scores
        snp = SNPRisk(
            rsid=rsid,
            gene=gene_symbol,
            chromosome=chromosome,
            position=position,
            disease=disease,
            # CAUSALdb2 SNP-level scores (REAL DATA)
            causal_score=causal_score,
            evidence_score=evidence_score,
            risk_score_raw=risk_score_raw,
            # GWAS data
            beta=beta,
            p_value=p_value,
            maf=maf,
            odds_ratio=odds_ratio,
            # Annotations
            clinical_significance=clin_sig,
            functional_impact=func_impact,
            # PIPs (optional)
            pip_abf=pip_abf,
            pip_finemap=pip_finemap,
            pip_susie=pip_susie,
            pip_paintor=pip_paintor,
            pip_caviarbf=pip_caviarbf,
            pip_polyfun=pip_polyfun,
            gene_causal_score=gene_causal_score
        )

        # Add to gene
        if gene_symbol not in self.gene_scores:
            self.gene_scores[gene_symbol] = GeneScore(
                gene_symbol=gene_symbol,
                chromosome=chromosome,
                gene_length=gene_length
            )

        self.gene_scores[gene_symbol].add_snp(snp)

    def get_multi_disease_genes(self) -> List[str]:
        """Get genes associated with multiple diseases"""
        return [
            gene for gene, score in self.gene_scores.items()
            if score.is_multi_disease()
        ]

    def get_shared_snps(self, gene_symbol: str) -> List[str]:
        """
        Get SNPs that appear in multiple diseases for a gene

        Returns:
            List of rsIDs that are shared across diseases
        """
        if gene_symbol not in self.gene_scores:
            return []

        gene = self.gene_scores[gene_symbol]
        disease_snps = gene.get_snps_by_disease()

        # Find SNPs that appear in multiple diseases
        snp_diseases: Dict[str, set] = {}
        for disease, snps in disease_snps.items():
            for snp in snps:
                if snp.rsid not in snp_diseases:
                    snp_diseases[snp.rsid] = set()
                snp_diseases[snp.rsid].add(disease)

        # Return SNPs in more than one disease
        return [rsid for rsid, diseases in snp_diseases.items() if len(diseases) > 1]

    def compute_risk_vector(self, gene_symbol: str, normalize: bool = False) -> Optional[GeneRiskVector]:
        """
        Compute risk vector for a gene

        Args:
            gene_symbol: Gene symbol
            normalize: If True, self-normalize (not recommended, use compute_all_risk_vectors)

        Returns:
            GeneRiskVector with per-disease scores (raw if normalize=False)
        """
        if gene_symbol not in self.gene_scores:
            return None

        gene = self.gene_scores[gene_symbol]
        disease_scores_raw = gene.compute_disease_scores(self.gwas_weight)
        disease_snps = gene.get_snps_by_disease()

        # Count SNPs per disease
        disease_snp_counts = {d: len(snps) for d, snps in disease_snps.items()}

        # Find shared SNPs
        shared_snps = self.get_shared_snps(gene_symbol)

        # Create risk vector with RAW scores
        risk_vector = GeneRiskVector(
            gene_symbol=gene_symbol,
            gene_id=gene.gene_id,
            chromosome=gene.chromosome,
            gene_length=gene.gene_length,
            disease_scores_raw=disease_scores_raw,
            disease_scores=disease_scores_raw.copy() if normalize else {},
            disease_snp_counts=disease_snp_counts,
            shared_snp_count=len(shared_snps),
            shared_snps=shared_snps
        )

        # Compute aggregations only if self-normalizing
        if normalize:
            risk_vector.compute_aggregations()

        return risk_vector

    def normalize_risk_vectors(
        self,
        vectors: Dict[str, GeneRiskVector],
        method: str = 'minmax'
    ) -> Dict[str, GeneRiskVector]:
        """
        Normalize all risk vectors to [0, 1] range

        Args:
            vectors: Dictionary of gene -> GeneRiskVector with raw scores
            method: Normalization method ('minmax', 'zscore', 'rank')

        Returns:
            Same vectors with normalized disease_scores
        """
        if not vectors:
            return vectors

        # Collect all diseases
        all_diseases = set()
        for v in vectors.values():
            all_diseases.update(v.disease_scores_raw.keys())

        # For each disease, compute normalization parameters
        disease_stats: Dict[str, Dict[str, float]] = {}

        for disease in all_diseases:
            scores = [
                v.disease_scores_raw.get(disease, 0)
                for v in vectors.values()
                if disease in v.disease_scores_raw
            ]

            if not scores:
                continue

            if method == 'minmax':
                min_val = min(scores)
                max_val = max(scores)
                disease_stats[disease] = {'min': min_val, 'max': max_val}
            elif method == 'zscore':
                mean_val = sum(scores) / len(scores)
                std_val = (sum((s - mean_val) ** 2 for s in scores) / len(scores)) ** 0.5
                disease_stats[disease] = {'mean': mean_val, 'std': std_val if std_val > 0 else 1}
            elif method == 'rank':
                # Sort scores and assign ranks
                sorted_scores = sorted(set(scores))
                rank_map = {s: i / (len(sorted_scores) - 1) if len(sorted_scores) > 1 else 0.5
                           for i, s in enumerate(sorted_scores)}
                disease_stats[disease] = {'rank_map': rank_map}

        # Apply normalization to each vector
        for gene_symbol, vector in vectors.items():
            normalized_scores = {}

            for disease, raw_score in vector.disease_scores_raw.items():
                if disease not in disease_stats:
                    normalized_scores[disease] = 0.0
                    continue

                stats = disease_stats[disease]

                if method == 'minmax':
                    min_val, max_val = stats['min'], stats['max']
                    if max_val > min_val:
                        normalized_scores[disease] = (raw_score - min_val) / (max_val - min_val)
                    else:
                        normalized_scores[disease] = 0.5  # All same value
                elif method == 'zscore':
                    # Z-score then sigmoid to [0, 1]
                    z = (raw_score - stats['mean']) / stats['std']
                    # Sigmoid: 1 / (1 + exp(-z))
                    normalized_scores[disease] = 1 / (1 + math.exp(-z))
                elif method == 'rank':
                    normalized_scores[disease] = stats['rank_map'].get(raw_score, 0.5)

            vector.disease_scores = normalized_scores
            vector.compute_aggregations()

        return vectors

    def compute_all_risk_vectors(self, normalize: bool = True, method: str = 'minmax') -> Dict[str, GeneRiskVector]:
        """
        Compute risk vectors for all genes with normalization

        Args:
            normalize: Whether to normalize scores to [0, 1] (default True)
            method: Normalization method ('minmax', 'zscore', 'rank')

        Returns:
            Dictionary of gene symbol to GeneRiskVector with NORMALIZED scores
        """
        # First pass: compute raw vectors
        vectors = {}
        for gene_symbol in self.gene_scores:
            vector = self.compute_risk_vector(gene_symbol, normalize=False)
            if vector:
                vectors[gene_symbol] = vector

        # Second pass: normalize all vectors together
        if normalize and vectors:
            vectors = self.normalize_risk_vectors(vectors, method=method)
        else:
            # If not normalizing, just copy raw to normalized and compute aggregations
            for vector in vectors.values():
                vector.disease_scores = vector.disease_scores_raw.copy()
                vector.compute_aggregations()

        return vectors

    def get_risk_vectors_summary(self) -> Dict:
        """
        Get summary of all risk vectors

        Returns:
            Dictionary with summary statistics
        """
        vectors = self.compute_all_risk_vectors()

        multi_disease = [v for v in vectors.values() if v.is_multi_disease()]
        single_disease = [v for v in vectors.values() if not v.is_multi_disease()]

        # Get all diseases
        all_diseases = set()
        for v in vectors.values():
            all_diseases.update(v.get_diseases())

        return {
            'total_genes': len(vectors),
            'multi_disease_genes': len(multi_disease),
            'single_disease_genes': len(single_disease),
            'diseases': list(all_diseases),
            'top_multi_disease_genes': [
                {
                    'gene': v.gene_symbol,
                    'diseases': v.get_diseases(),
                    'max_score': v.max_score,
                    'mean_score': v.mean_score
                }
                for v in sorted(multi_disease, key=lambda x: x.max_score, reverse=True)[:10]
            ]
        }

    def export_risk_vectors_to_csv(self, filepath: str):
        """
        Export risk vectors to CSV file.

        All scores are inherently normalized to [0, 1] - no post-hoc normalization needed.

        Args:
            filepath: Output file path
        """
        import csv

        vectors = self.compute_all_risk_vectors(normalize=False)  # Scores are inherently [0,1]

        # Get all diseases
        all_diseases = set()
        for v in vectors.values():
            all_diseases.update(v.get_diseases())
        all_diseases = sorted(all_diseases)

        # Build rows
        rows = []
        for gene_symbol, vector in sorted(vectors.items()):
            row = {
                'gene': gene_symbol,
                'gene_id': vector.gene_id or '',
                'chromosome': vector.chromosome or '',
                'gene_length': vector.gene_length,
                'is_multi_disease': vector.is_multi_disease(),
                'num_diseases': len(vector.get_diseases()),
                'diseases': ','.join(vector.get_diseases()),
                'max_score': round(vector.max_score, 6),
                'mean_score': round(vector.mean_score, 6),
                'weighted_score': round(vector.weighted_score, 6),
                'top_disease': vector.get_top_disease() or '',
                'shared_snp_count': vector.shared_snp_count,
                'total_snp_count': sum(vector.disease_snp_counts.values())
            }

            # Add per-disease columns (NORMALIZED scores)
            for disease in all_diseases:
                row[f'{disease}_score'] = round(vector.disease_scores.get(disease, 0), 6)
                row[f'{disease}_score_raw'] = round(vector.disease_scores_raw.get(disease, 0), 6)
                row[f'{disease}_snps'] = vector.disease_snp_counts.get(disease, 0)

            rows.append(row)

        # Write CSV
        if rows:
            fieldnames = list(rows[0].keys())
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    @staticmethod
    def compute_snp_risk_from_gwas(
        odds_ratio: float,
        maf: float,
        p_value: float = 5e-8
    ) -> float:
        """
        Simplified SNP risk calculation from GWAS data only

        Args:
            odds_ratio: GWAS odds ratio
            maf: Minor allele frequency
            p_value: GWAS p-value

        Returns:
            SNP risk contribution
        """
        # Effect size from OR
        effect = abs(math.log(odds_ratio)) if odds_ratio > 0 else 0

        # MAF penalty (rare variants weighted higher)
        maf_weight = 1.0 - min(maf, 0.5)

        # P-value significance
        p_weight = min(-math.log10(p_value) / 10.0, 1.0) if p_value > 0 else 1.0

        return effect * maf_weight * p_weight

    def compute_wgrs_from_gwas_data(
        self,
        gene_symbol: str,
        snp_data: List[Dict],
        gene_length: int = 10000,
        gwas_score: float = 0.0
    ) -> float:
        """
        Compute WGRS directly from GWAS data

        Args:
            gene_symbol: Gene symbol
            snp_data: List of dicts with 'rsid', 'maf', 'odds_ratio', 'p_value'
            gene_length: Gene length in bp
            gwas_score: GWAS disease association score

        Returns:
            WGRS score
        """
        # Compute SNP risks
        snp_risks = []
        for snp in snp_data:
            risk = self.compute_snp_risk_from_gwas(
                odds_ratio=snp.get('odds_ratio', 1.0),
                maf=snp.get('maf', 0.01),
                p_value=snp.get('p_value', 5e-8)
            )
            snp_risks.append(risk)

        if not snp_risks:
            return 0.0

        # Use MEAN (preserves [0, 1])
        base = sum(snp_risks) / len(snp_risks)

        # Apply GWAS boost (preserves [0, 1])
        gwas_boost = base * gwas_score
        wgrs = (1 - self.gwas_weight) * base + self.gwas_weight * (base + gwas_boost) / 2

        return max(0.0, min(1.0, wgrs))


class APIEnabledGRASSCalculator(GRASSCalculator):
    """
    GRASS Calculator with API integration

    Fetches real clinical annotations from:
    - ClinVar: Clinical significance
    - Open Targets: GWAS disease associations
    - Ensembl: Gene length and variant consequences
    """

    def __init__(
        self,
        gwas_weight: float = 0.3,
        qual_threshold: float = 45.0,
        gq_threshold: float = 60.0,
        dp_threshold: float = 40.0,
        use_apis: bool = True,
        clinvar_api_key: Optional[str] = None
    ):
        """
        Initialize API-enabled GRASS calculator

        Args:
            gwas_weight: GWAS weight contribution [0, 1]
            qual_threshold: Minimum variant quality
            gq_threshold: Minimum genotype quality
            dp_threshold: Minimum read depth
            use_apis: Enable API calls for data fetching
            clinvar_api_key: Optional NCBI API key
        """
        super().__init__(gwas_weight, qual_threshold, gq_threshold, dp_threshold)

        self.use_apis = use_apis
        self._api_client = None
        self._clinvar_api_key = clinvar_api_key

    @property
    def api_client(self):
        """Lazy-load API client"""
        if self._api_client is None and self.use_apis:
            try:
                from .apis import UnifiedBiomedicalClient
                self._api_client = UnifiedBiomedicalClient(
                    clinvar_api_key=self._clinvar_api_key
                )
            except ImportError:
                logger.warning("API client not available, using offline mode")
                self.use_apis = False
        return self._api_client

    def add_snp_with_api_lookup(
        self,
        gene_symbol: str,
        rsid: str,
        chromosome: str = "",
        position: int = 0,
        qual: float = 100.0,
        gq: float = 100.0,
        dp: float = 100.0
    ):
        """
        Add SNP with automatic API lookup for annotations

        Args:
            gene_symbol: Gene symbol
            rsid: SNP identifier
            chromosome: Chromosome (optional, fetched from API)
            position: Position (optional, fetched from API)
            qual: Variant quality
            gq: Genotype quality
            dp: Read depth
        """
        if not self.use_apis or self.api_client is None:
            # Fallback to basic addition
            self.add_snp(
                gene_symbol=gene_symbol,
                rsid=rsid,
                chromosome=chromosome,
                position=position,
                qual=qual,
                gq=gq,
                dp=dp
            )
            return

        try:
            # Fetch variant annotation from APIs
            var_ann = self.api_client.get_variant_annotation(rsid)

            # Map API clinical significance to local enum
            clin_sig = None
            if var_ann.clinical_significance:
                from .apis import ClinicalSignificance as APIClinSig
                clin_map = {
                    APIClinSig.PATHOGENIC: 'pathogenic',
                    APIClinSig.LIKELY_PATHOGENIC: 'likely_pathogenic',
                    APIClinSig.UNCERTAIN_SIGNIFICANCE: 'uncertain_significance',
                    APIClinSig.LIKELY_BENIGN: 'likely_benign',
                    APIClinSig.BENIGN: 'benign',
                }
                clin_sig = clin_map.get(var_ann.clinical_significance)

            # Map consequence to impact
            func_impact = None
            if var_ann.consequence_type:
                high_impact = ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant',
                              'splice_donor_variant', 'start_lost', 'stop_lost']
                moderate_impact = ['missense_variant', 'inframe_insertion', 'inframe_deletion']
                low_impact = ['splice_region_variant', 'synonymous_variant']

                if var_ann.consequence_type in high_impact:
                    func_impact = 'high'
                elif var_ann.consequence_type in moderate_impact:
                    func_impact = 'moderate'
                elif var_ann.consequence_type in low_impact:
                    func_impact = 'low'
                else:
                    func_impact = 'modifier'

            # Add SNP with fetched data
            self.add_snp(
                gene_symbol=gene_symbol,
                rsid=rsid,
                chromosome=var_ann.chromosome or chromosome,
                position=var_ann.position or position,
                maf=var_ann.maf or 0.0,
                clinical_significance=clin_sig,
                functional_impact=func_impact,
                qual=qual,
                gq=gq,
                dp=dp
            )

        except Exception as e:
            logger.warning(f"API lookup failed for {rsid}: {e}")
            self.add_snp(
                gene_symbol=gene_symbol,
                rsid=rsid,
                chromosome=chromosome,
                position=position,
                qual=qual,
                gq=gq,
                dp=dp
            )

    def fetch_gene_data(self, gene_symbol: str):
        """
        Fetch gene data from APIs and update scores

        Args:
            gene_symbol: Gene symbol
        """
        if not self.use_apis or self.api_client is None:
            return

        if gene_symbol not in self.gene_scores:
            return

        try:
            gene_ann = self.api_client.get_gene_annotation(
                gene_symbol,
                include_network=False,
                include_enrichment=False,
                include_mr=False
            )

            # Update gene length
            if gene_ann.length:
                self.gene_scores[gene_symbol].gene_length = gene_ann.length

            # Update GWAS score
            if gene_ann.top_disease_score > 0:
                self.gene_scores[gene_symbol].gwas_score = gene_ann.top_disease_score

        except Exception as e:
            logger.warning(f"Failed to fetch gene data for {gene_symbol}: {e}")

    def compute_wgrs_with_api(
        self,
        gene_symbol: str,
        variants: List[str],
        disease_context: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute WGRS with full API data fetching

        Args:
            gene_symbol: Gene symbol
            variants: List of variant rsIDs
            disease_context: Disease keywords for context

        Returns:
            Dictionary with component scores
        """
        if not self.use_apis or self.api_client is None:
            # Fallback to basic calculation
            for rsid in variants:
                self.add_snp(gene_symbol=gene_symbol, rsid=rsid, chromosome="", position=0)
            return {'wgrs': self.compute_all_scores().get(gene_symbol, 0)}

        return self.api_client.get_gene_risk_score(
            gene_symbol,
            variants,
            disease_context
        )


def example_usage():
    """Example usage of GRASS calculator"""
    # Initialize calculator (all scores normalized to [0, 1])
    calc = GRASSCalculator(gwas_weight=0.3)

    # Add SNPs for TYK2 gene (shared between RA and COVID-19)
    calc.add_snp(
        gene_symbol='TYK2',
        rsid='rs34536443',
        chromosome='19',
        position=10463118,
        maf=0.04,
        odds_ratio=0.63,  # Protective
        p_value=1e-20,
        functional_impact='high',
        gene_length=30000
    )

    calc.add_snp(
        gene_symbol='TYK2',
        rsid='rs12720356',
        chromosome='19',
        position=10469975,
        maf=0.08,
        odds_ratio=0.85,
        p_value=5e-10,
        functional_impact='moderate',
        gene_length=30000
    )

    # Set GWAS association score
    calc.set_gwas_score('TYK2', 0.8)

    # Compute scores
    scores = calc.compute_all_scores()
    print(f"TYK2 WGRS: {scores['TYK2']:.4f}")

    # Get detailed breakdown
    details = calc.get_gene_details('TYK2')
    print(f"Gene details: {details}")


def example_api_usage():
    """Example usage with API integration"""
    print("\nAPI-Enabled GRASS Calculator")
    print("=" * 50)

    # Initialize API-enabled calculator (all scores normalized to [0, 1])
    calc = APIEnabledGRASSCalculator(gwas_weight=0.3, use_apis=True)

    # Add SNPs with automatic API lookup
    calc.add_snp_with_api_lookup('TYK2', 'rs34536443')
    calc.add_snp_with_api_lookup('TYK2', 'rs12720356')

    # Fetch gene-level data
    calc.fetch_gene_data('TYK2')

    # Compute scores
    scores = calc.compute_all_scores()
    print(f"TYK2 WGRS: {scores.get('TYK2', 0):.4f}")

    # Or use full API computation
    result = calc.compute_wgrs_with_api(
        'TYK2',
        ['rs34536443', 'rs12720356'],
        disease_context=['rheumatoid', 'arthritis']
    )
    print(f"Full API result: {result}")


if __name__ == '__main__':
    example_usage()

    # Uncomment to test API integration
    # example_api_usage()
