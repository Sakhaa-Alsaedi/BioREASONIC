# GRASS: Genetic Risk Aggregation Scoring System

A systematic pipeline for prioritizing disease-risk genes by integrating fine-mapping posterior inclusion probabilities (PIPs) with GWAS evidence and functional annotations.

## Overview

GRASS calculates gene-level risk scores using a mathematically principled formula that combines:
- **Causal evidence** from fine-mapping (7 methods from CAUSALdb2)
- **GWAS significance** (-log10 p-value)
- **Effect size** (|beta|)
- **Functional annotations** (VEP impact scores)

## GRASS Formula

```
GRASS_score = MAX + (1 - MAX) × β × support
```

Where:
| Component | Description | Range |
|-----------|-------------|-------|
| MAX | Maximum extended_score among SNPs mapped to gene | [0, 1] |
| β | Support weight (default: 0.3) | constant |
| support | log(1 + n_sig) / log(1 + N_max) | [0, 1] |
| n_sig | Count of genome-wide significant SNPs (p < 5×10⁻⁸) | integer |

### Extended SNP Score

```
extended_score = 0.65 × causal_score    (fine-mapping PIPs)
               + 0.05 × vep_score       (functional impact)
               + 0.10 × evidence_score  (GWAS significance)
               + 0.20 × effect_score    (effect size |beta|)
```

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/GRASS.git
cd GRASS

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run the complete pipeline
python run_pipeline.py

# Or run steps individually:
python pipeline/step0_load_data.py          # Load and QC filter data
python pipeline/step3_annotate_snps.py      # Annotate with ClinVar/VEP
python pipeline/step3b_extended_scores.py   # Calculate extended scores
python pipeline/step4_map_to_genes.py       # Map SNPs to genes
python pipeline/step5_grass_scores.py       # Calculate GRASS scores
python pipeline/step6_prepare_enrichment.py # Prepare for enrichment
python enrichment/comprehensive_enrichment.py # Run enrichment analysis
```

## Directory Structure

```
GRASS/
├── README.md                    # This file
├── run_pipeline.py              # Main pipeline runner
├── requirements.txt             # Python dependencies
├── pipeline/                    # Core pipeline scripts
│   ├── config.py               # Central configuration
│   ├── step0_load_data.py      # Data loading and QC
│   ├── step3_annotate_snps.py  # ClinVar + VEP annotation
│   ├── step3b_extended_scores.py # Extended score calculation
│   ├── step4_map_to_genes.py   # SNP-gene mapping
│   ├── step5_grass_scores.py   # GRASS gene scoring
│   └── step6_prepare_enrichment.py # Enrichment prep
├── data/                        # Input data
│   ├── AD_snps.csv             # Alzheimer's Disease SNPs
│   └── t2d_snps.csv            # Type 2 Diabetes SNPs
├── annotations/                 # Gene annotations
│   ├── grch37_genes.csv        # Ensembl GRCh37 genes
│   └── cache/                  # API response cache
├── seedExp/                     # Pre-computed SNP-gene mappings
├── enrichment/                  # Enrichment analysis scripts
│   └── comprehensive_enrichment.py
├── output/                      # Generated results
│   ├── GRASS_AD_gene_scores.csv
│   ├── GRASS_T2D_gene_scores.csv
│   └── GRASS_master_scores.csv
└── docs/                        # Documentation
```

## Pipeline Steps

### Step 0: Load Pre-processed Data
Loads SNP data with quality control filtering:
- P-value: p < 5×10⁻⁸ (genome-wide significance)
- MAF: 0.001 to 0.5
- Risk direction: beta > 0

### Step 3: Annotate SNPs
Fetches annotations from:
- **ClinVar**: Clinical significance
- **VEP (Ensembl)**: Variant consequences

### Step 3b: Extended Scores
Calculates 4-component weighted SNP scores.

### Step 4: Map to Genes
Maps SNPs to genes within 500kb window using GRCh37 coordinates.

### Step 5: GRASS Scores
Aggregates SNP scores to gene-level using the GRASS formula.

### Step 6 + Enrichment
Prepares normalized rankings and runs pathway/disease enrichment via Enrichr and STRING PPI analysis.

## Expected Results

### Alzheimer's Disease (AD)
| Rank | Gene | GRASS Score |
|------|------|-------------|
| 1 | APOE | 0.8089 |
| 2 | PVRL2 | 0.7195 |
| 3 | APOC1 | 0.7156 |
| 4 | BIN1 | 0.6250 |
| 5 | CLPTM1 | 0.6090 |

### Type 2 Diabetes (T2D)
| Rank | Gene | GRASS Score |
|------|------|-------------|
| 1 | COBLL1 | 0.7064 |
| 2 | PPARG | 0.7055 |
| 3 | TCF7L2 | 0.6944 |
| 4 | CDKN1C | 0.6911 |
| 5 | KCNQ1 | 0.6911 |

## Configuration

Edit `pipeline/config.py` to customize:

```python
# GRASS formula
GRASS_BETA = 0.3      # Support weight

# Extended score weights
W_CAUSAL = 0.65       # Fine-mapping PIPs
W_VEP = 0.05          # VEP functional impact
W_EVIDENCE = 0.10     # GWAS significance
W_EFFECT = 0.20       # Effect size |beta|

# QC thresholds
GWAS_SIGNIFICANCE = 5e-8
MAF_MIN = 0.001
MAF_MAX = 0.5

# Gene mapping
WINDOW_SIZE = 500000  # 500kb
```

## Input Data Format

### SNP Data (AD_snps.csv, t2d_snps.csv)

Required columns:
| Column | Description |
|--------|-------------|
| rsid | SNP identifier |
| chr | Chromosome |
| bp | Base pair position |
| beta | Effect size |
| p | P-value |
| maf | Minor allele frequency |
| causal_score | Mean fine-mapping PIP |

## Output Files

### GRASS_*_gene_scores.csv

| Column | Description |
|--------|-------------|
| gene_name | Gene symbol |
| gene_id | Ensembl gene ID |
| max_score | MAX(extended_score) |
| snp_count | Number of mapped SNPs |
| n_sig | Significant SNP count |
| support | Normalized support term |
| GRASS_score | Final gene score |
| rank | GRASS rank |

## Data Sources

- **Fine-mapping PIPs**: CAUSALdb2 v2.1
  - Methods: ABF, FINEMAP, PAINTOR, CAVIARBF, SuSiE, PolyFun+FINEMAP, PolyFun+SuSiE
- **Gene annotations**: Ensembl GRCh37
- **Functional annotations**: VEP (Ensembl), ClinVar
- **Enrichment**: Enrichr, STRING PPI

## Citation

If you use GRASS in your research, please cite:

```
GRASS: Genetic Risk Aggregation Scoring System
A gene prioritization framework integrating fine-mapping and GWAS evidence.
```

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue.
