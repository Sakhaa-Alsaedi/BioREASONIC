#!/usr/bin/env python3
"""
================================================================================
GRASS Comprehensive Enrichment Analysis with PPI Network
================================================================================
All 41 risk genes analyzed together with category tracking:
- AD genes
- T2D genes
- Shared genes (AD & T2D)

Expanded Enrichr Libraries:
- Pathway: KEGG, Reactome, WikiPathways
- GO: Biological Process, Molecular Function, Cellular Component
- Disease: OMIM, DisGeNET, GWAS Catalog, Jensen DISEASES, ClinVar
- Tissue: GTEx, ARCHS4, Human Gene Atlas, Allen Brain Atlas
- Transcription: ENCODE/ChEA TFs
- Drug: DSigDB, DrugMatrix, Drug Perturbations

STRING PPI network with gene category visualization.
================================================================================
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# COLORS (from project config)
# =============================================================================
COLORS = {
    'shared': '#FF7E79',
    'ad_only': '#B18CC7',
    't2d_only': '#80B4E1',
    'extra1': '#B4D8E9',
    'extra2': '#F3B0AA',
    'extra3': '#A9D7B0',
    'extra4': '#F6CF9F',
    'extra5': '#50D8CE',
    'extra6': '#FEF6A8',
    'extra7': '#CDA7A7',
    'extra8': '#917EAA',
}

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
ENRICH_DIR = BASE_DIR / "enrichment"

print("="*70)
print("GRASS COMPREHENSIVE ENRICHMENT - ALL 41 GENES")
print("="*70)

# =============================================================================
# ENRICHR API FUNCTIONS
# =============================================================================
ENRICHR_URL = "https://maayanlab.cloud/Enrichr"

def enrichr_add_list(gene_list, description=""):
    """Add gene list to Enrichr and get list ID."""
    genes_str = "\n".join(gene_list)
    payload = {
        'list': (None, genes_str),
        'description': (None, description)
    }
    response = requests.post(f"{ENRICHR_URL}/addList", files=payload, timeout=30)
    if not response.ok:
        raise Exception(f"Error adding gene list: {response.text}")
    return response.json()['userListId']

def enrichr_get_results(user_list_id, gene_set_library):
    """Get enrichment results for a gene set library."""
    url = f"{ENRICHR_URL}/enrich"
    params = {'userListId': user_list_id, 'backgroundType': gene_set_library}
    response = requests.get(url, params=params, timeout=60)
    if not response.ok:
        return []

    data = response.json()
    results = data.get(gene_set_library, [])

    parsed = []
    for r in results:
        parsed.append({
            'Rank': r[0],
            'Term': r[1],
            'P-value': r[2],
            'Z-score': r[3],
            'Combined Score': r[4],
            'Genes': ';'.join(r[5]) if r[5] else '',
            'Adjusted P-value': r[6],
            'Gene_set': gene_set_library,
        })
    return parsed

# =============================================================================
# STRING PPI FUNCTIONS
# =============================================================================
STRING_API = "https://string-db.org/api"
SPECIES = 9606  # Human

def string_map_genes(genes):
    """Map gene symbols to STRING IDs."""
    print(f"    Mapping {len(genes)} genes to STRING IDs...")

    url = f"{STRING_API}/json/get_string_ids"
    params = {
        'identifiers': '\r'.join(genes),
        'species': SPECIES,
        'limit': 1,
        'caller_identity': 'grass_enrichment'
    }

    try:
        response = requests.post(url, data=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        mapping = {}
        for item in data:
            query = item.get('queryItem', '')
            string_id = item.get('stringId', '')
            preferred = item.get('preferredName', '')
            if query and string_id:
                mapping[query] = {
                    'string_id': string_id,
                    'preferred_name': preferred
                }

        print(f"    Mapped {len(mapping)} of {len(genes)} genes")
        return mapping
    except Exception as e:
        print(f"    Error mapping genes: {e}")
        return {}

def string_get_interactions(string_ids, score_threshold=700):
    """Get PPI interactions from STRING for each gene individually."""
    print(f"    Fetching PPI interactions for each gene (score >= {score_threshold})...")

    all_edges = []

    for i, sid in enumerate(string_ids):
        print(f"      [{i+1}/{len(string_ids)}] Getting partners for {sid.split('.')[-1]}...", end=" ")

        url = f"{STRING_API}/json/interaction_partners"
        params = {
            'identifiers': sid,
            'species': SPECIES,
            'required_score': score_threshold,
            'limit': 50,
            'caller_identity': 'grass_enrichment'
        }

        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            for item in data:
                all_edges.append({
                    'protein1': item.get('preferredName_A', ''),
                    'protein2': item.get('preferredName_B', ''),
                    'string_id1': item.get('stringId_A', ''),
                    'string_id2': item.get('stringId_B', ''),
                    'score': item.get('score', 0),
                })

            print(f"{len(data)} partners")
            time.sleep(0.2)

        except Exception as e:
            print(f"Error: {e}")

    # Remove duplicates
    df = pd.DataFrame(all_edges)
    if len(df) > 0:
        df = df.drop_duplicates(subset=['string_id1', 'string_id2'])

    print(f"    Total unique interactions: {len(df)}")
    return df

def string_get_enrichment(string_ids):
    """Get functional enrichment from STRING."""
    print(f"    Fetching STRING functional enrichment...")

    url = f"{STRING_API}/json/enrichment"
    params = {
        'identifiers': '\r'.join(string_ids),
        'species': SPECIES,
        'caller_identity': 'grass_enrichment'
    }

    try:
        response = requests.post(url, data=params, timeout=120)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data:
            results.append({
                'category': item.get('category', ''),
                'term': item.get('term', ''),
                'description': item.get('description', ''),
                'p_value': item.get('p_value', 1.0),
                'fdr': item.get('fdr', 1.0),
                'genes': ','.join(item.get('preferredNames', [])),
                'gene_count': item.get('number_of_genes', 0),
            })

        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.sort_values('fdr')
        return df
    except Exception as e:
        print(f"    Error: {e}")
        return pd.DataFrame()

# =============================================================================
# LOAD TOP GENES
# =============================================================================
print("\n[1] Loading top genes...")

ad_genes = pd.read_csv(OUTPUT_DIR / "ad_genes_normalized_ranking.csv")
t2d_genes = pd.read_csv(OUTPUT_DIR / "t2d_genes_normalized_ranking.csv")
shared_genes = pd.read_csv(OUTPUT_DIR / "shared_genes_normalized_ranking.csv")

top15_ad = ad_genes.head(15)['gene_name'].tolist()
top15_t2d = t2d_genes.head(15)['gene_name'].tolist()
top15_shared = shared_genes.head(15)['gene_name'].tolist()

# Clean gene names
def clean_genes(genes):
    cleaned = []
    for g in genes:
        if 'APOC4-APOC2' in g:
            continue
        if g.startswith('CTB-') or g.startswith('C19orf') or g.startswith('C6orf'):
            continue
        cleaned.append(g)
    return list(set(cleaned))

top15_ad_clean = clean_genes(top15_ad)
top15_t2d_clean = clean_genes(top15_t2d)
top15_shared_clean = clean_genes(top15_shared)

# All genes combined - 41 unique genes
all_genes = list(set(top15_ad_clean + top15_t2d_clean + top15_shared_clean))

# Create gene category mapping
gene_categories = {}
for g in top15_shared_clean:
    gene_categories[g] = 'shared'
for g in top15_ad_clean:
    if g not in gene_categories:
        gene_categories[g] = 'ad_only'
for g in top15_t2d_clean:
    if g not in gene_categories:
        gene_categories[g] = 't2d_only'

print(f"  AD genes ({len(top15_ad_clean)}): {top15_ad_clean}")
print(f"  T2D genes ({len(top15_t2d_clean)}): {top15_t2d_clean}")
print(f"  Shared genes ({len(top15_shared_clean)}): {top15_shared_clean}")
print(f"  All unique genes: {len(all_genes)}")

# Save gene category mapping
gene_cat_df = pd.DataFrame([
    {'gene': g, 'category': c,
     'ad': 1 if g in top15_ad_clean or g in top15_shared_clean else 0,
     't2d': 1 if g in top15_t2d_clean or g in top15_shared_clean else 0}
    for g, c in gene_categories.items()
])
gene_cat_df.to_csv(ENRICH_DIR / "gene_categories.csv", index=False)
print(f"  Saved: gene_categories.csv")

# =============================================================================
# EXPANDED ENRICHR LIBRARIES
# =============================================================================
# Pathway libraries
PATHWAY_LIBS = [
    'KEGG_2021_Human',
    'Reactome_2022',
    'WikiPathways_2023_Human',
    'BioPlanet_2019',
]

# GO libraries
GO_LIBS = [
    'GO_Biological_Process_2023',
    'GO_Molecular_Function_2023',
    'GO_Cellular_Component_2023',
]

# Disease libraries
DISEASE_LIBS = [
    'OMIM_Disease',
    'DisGeNET',
    'GWAS_Catalog_2023',
    'Jensen_DISEASES',
    'ClinVar_2019',
]

# Tissue expression libraries
TISSUE_LIBS = [
    'GTEx_Tissue_Expression_Up',
    'GTEx_Tissue_Expression_Down',
    'ARCHS4_Tissues',
    'Human_Gene_Atlas',
    'Allen_Brain_Atlas_up',
    'Allen_Brain_Atlas_down',
]

# Transcription factor libraries
TF_LIBS = [
    'ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X',
    'ChEA_2022',
    'ENCODE_TF_ChIP-seq_2015',
]

# Drug/perturbation libraries
DRUG_LIBS = [
    'DSigDB',
    'DrugMatrix',
    'Drug_Perturbations_from_GEO_up',
    'Drug_Perturbations_from_GEO_down',
]

# All libraries combined
ALL_GENE_SETS = PATHWAY_LIBS + GO_LIBS + DISEASE_LIBS + TISSUE_LIBS + TF_LIBS + DRUG_LIBS

# Library colors
LIBRARY_COLORS = {
    # Pathway
    'KEGG_2021_Human': '#3498DB',
    'Reactome_2022': '#9B59B6',
    'WikiPathways_2023_Human': '#1ABC9C',
    'BioPlanet_2019': '#16A085',
    # GO
    'GO_Biological_Process_2023': '#2ECC71',
    'GO_Molecular_Function_2023': '#27AE60',
    'GO_Cellular_Component_2023': '#F1C40F',
    # Disease
    'OMIM_Disease': '#E74C3C',
    'DisGeNET': '#C0392B',
    'GWAS_Catalog_2023': '#E67E22',
    'Jensen_DISEASES': '#D35400',
    'ClinVar_2019': '#E91E63',
    # Tissue
    'GTEx_Tissue_Expression_Up': COLORS['ad_only'],
    'GTEx_Tissue_Expression_Down': COLORS['extra8'],
    'ARCHS4_Tissues': COLORS['t2d_only'],
    'Human_Gene_Atlas': COLORS['extra1'],
    'Allen_Brain_Atlas_up': '#8E44AD',
    'Allen_Brain_Atlas_down': '#6C3483',
    # TF
    'ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X': '#34495E',
    'ChEA_2022': '#2C3E50',
    'ENCODE_TF_ChIP-seq_2015': '#7F8C8D',
    # Drug
    'DSigDB': '#27AE60',
    'DrugMatrix': '#16A085',
    'Drug_Perturbations_from_GEO_up': COLORS['shared'],
    'Drug_Perturbations_from_GEO_down': COLORS['extra2'],
}

# =============================================================================
# RUN ENRICHMENT ON ALL 41 GENES
# =============================================================================
print("\n[2] Running Enrichr on ALL 41 genes (expanded libraries)...")

def run_enrichr_all(gene_list, name, libraries):
    """Run Enrichr analysis with expanded libraries."""
    print(f"\n  Enrichr for {name} ({len(gene_list)} genes)...")

    try:
        list_id = enrichr_add_list(gene_list, name)
        print(f"    List ID: {list_id}")

        all_results = []
        for lib in libraries:
            print(f"    {lib}...", end=" ")
            try:
                results = enrichr_get_results(list_id, lib)
                sig_results = [r for r in results if r['Adjusted P-value'] < 0.05]
                print(f"{len(sig_results)} sig")

                # Add gene category info to each result
                for r in sig_results:
                    genes_in_term = r['Genes'].split(';') if r['Genes'] else []
                    ad_genes_in = [g for g in genes_in_term if g in top15_ad_clean or g in top15_shared_clean]
                    t2d_genes_in = [g for g in genes_in_term if g in top15_t2d_clean or g in top15_shared_clean]
                    shared_genes_in = [g for g in genes_in_term if g in top15_shared_clean]

                    r['AD_genes'] = ';'.join(ad_genes_in)
                    r['T2D_genes'] = ';'.join(t2d_genes_in)
                    r['Shared_genes'] = ';'.join(shared_genes_in)
                    r['N_AD'] = len(ad_genes_in)
                    r['N_T2D'] = len(t2d_genes_in)
                    r['N_Shared'] = len(shared_genes_in)

                all_results.extend(sig_results)
            except Exception as e:
                print(f"Error: {e}")
            time.sleep(0.3)

        df = pd.DataFrame(all_results)
        if len(df) > 0:
            df = df.sort_values('Adjusted P-value')
        return df
    except Exception as e:
        print(f"    Error: {e}")
        return pd.DataFrame()

# Run enrichment on all 41 genes
enrichr_all = run_enrichr_all(all_genes, 'All 41 Risk Genes', ALL_GENE_SETS)

# Save full results
if len(enrichr_all) > 0:
    enrichr_all.to_csv(ENRICH_DIR / "enrichr_all_41_genes.csv", index=False)
    print(f"  Saved: enrichr_all_41_genes.csv ({len(enrichr_all)} terms)")

# =============================================================================
# SEPARATE RESULTS BY LIBRARY TYPE
# =============================================================================
print("\n[3] Separating results by library type...")

# Pathway enrichment
pathway_df = enrichr_all[enrichr_all['Gene_set'].isin(PATHWAY_LIBS)] if len(enrichr_all) > 0 else pd.DataFrame()
if len(pathway_df) > 0:
    pathway_df.to_csv(ENRICH_DIR / "enrichr_pathways.csv", index=False)
    print(f"  Saved: enrichr_pathways.csv ({len(pathway_df)} terms)")

# GO enrichment
go_df = enrichr_all[enrichr_all['Gene_set'].isin(GO_LIBS)] if len(enrichr_all) > 0 else pd.DataFrame()
if len(go_df) > 0:
    go_df.to_csv(ENRICH_DIR / "enrichr_go_terms.csv", index=False)
    print(f"  Saved: enrichr_go_terms.csv ({len(go_df)} terms)")

# Disease enrichment
disease_df = enrichr_all[enrichr_all['Gene_set'].isin(DISEASE_LIBS)] if len(enrichr_all) > 0 else pd.DataFrame()
if len(disease_df) > 0:
    disease_df.to_csv(ENRICH_DIR / "enrichr_diseases.csv", index=False)
    print(f"  Saved: enrichr_diseases.csv ({len(disease_df)} terms)")

# Tissue enrichment
tissue_df = enrichr_all[enrichr_all['Gene_set'].isin(TISSUE_LIBS)] if len(enrichr_all) > 0 else pd.DataFrame()
if len(tissue_df) > 0:
    tissue_df.to_csv(ENRICH_DIR / "enrichr_tissues.csv", index=False)
    print(f"  Saved: enrichr_tissues.csv ({len(tissue_df)} terms)")

# TF enrichment
tf_df = enrichr_all[enrichr_all['Gene_set'].isin(TF_LIBS)] if len(enrichr_all) > 0 else pd.DataFrame()
if len(tf_df) > 0:
    tf_df.to_csv(ENRICH_DIR / "enrichr_transcription_factors.csv", index=False)
    print(f"  Saved: enrichr_transcription_factors.csv ({len(tf_df)} terms)")

# Drug enrichment
drug_df = enrichr_all[enrichr_all['Gene_set'].isin(DRUG_LIBS)] if len(enrichr_all) > 0 else pd.DataFrame()
if len(drug_df) > 0:
    drug_df.to_csv(ENRICH_DIR / "enrichr_drugs.csv", index=False)
    print(f"  Saved: enrichr_drugs.csv ({len(drug_df)} terms)")

# =============================================================================
# STRING PPI NETWORK
# =============================================================================
print("\n[4] Building STRING PPI Network...")

# Map all genes to STRING IDs
gene_mapping = string_map_genes(all_genes)
string_ids = [v['string_id'] for v in gene_mapping.values()]

# Get PPI interactions (score >= 700 for high confidence)
ppi_df = string_get_interactions(string_ids, score_threshold=700)
ppi_df.to_csv(ENRICH_DIR / "ppi_interactions.csv", index=False)
print(f"  Saved: ppi_interactions.csv ({len(ppi_df)} edges)")

# Get STRING functional enrichment
string_enrichment = string_get_enrichment(string_ids)
if len(string_enrichment) > 0:
    string_enrichment.to_csv(ENRICH_DIR / "string_enrichment.csv", index=False)
    print(f"  Saved: string_enrichment.csv ({len(string_enrichment)} terms)")

# =============================================================================
# BUILD NETWORK GRAPH
# =============================================================================
print("\n[5] Building network graph...")

G = nx.Graph()

# Create mapping from input gene to STRING preferred name
gene_to_preferred = {}
preferred_to_gene = {}
seed_proteins = set()

for gene, info in gene_mapping.items():
    preferred = info.get('preferred_name', gene)
    gene_to_preferred[gene] = preferred
    preferred_to_gene[preferred] = gene
    seed_proteins.add(preferred)

# Determine node type for seed genes
def get_seed_type(gene):
    return gene_categories.get(gene, 'seed')

# Add seed gene nodes
for gene in all_genes:
    preferred = gene_to_preferred.get(gene, gene)
    node_type = get_seed_type(gene)
    G.add_node(preferred, node_type=node_type, original_gene=gene, is_seed=True)

# Add PPI edges and partner nodes
if len(ppi_df) > 0:
    for _, row in ppi_df.iterrows():
        p1, p2 = row['protein1'], row['protein2']
        score = row['score']

        # Add partner nodes if not present
        if p1 not in G.nodes():
            G.add_node(p1, node_type='partner', original_gene=p1, is_seed=False)
        if p2 not in G.nodes():
            G.add_node(p2, node_type='partner', original_gene=p2, is_seed=False)

        G.add_edge(p1, p2, weight=score, edge_type='ppi')

# Count node types
seed_count = sum(1 for n in G.nodes() if G.nodes[n].get('is_seed', False))
partner_count = G.number_of_nodes() - seed_count
print(f"  Network: {G.number_of_nodes()} nodes ({seed_count} seed, {partner_count} partners), {G.number_of_edges()} edges")

# Save network
nx.write_graphml(G, ENRICH_DIR / "ppi_network.graphml")
print(f"  Saved: ppi_network.graphml")

# =============================================================================
# VISUALIZATION 1: TOP ENRICHED TERMS (ALL 41 GENES)
# =============================================================================
print("\n[6] Creating visualizations...")

if len(enrichr_all) > 0:
    fig, ax = plt.subplots(figsize=(14, 12))

    top25 = enrichr_all.nsmallest(25, 'Adjusted P-value').copy()
    top25['neg_log_p'] = -np.log10(top25['Adjusted P-value'].clip(lower=1e-50))
    top25 = top25.sort_values('neg_log_p', ascending=True)

    bar_colors = [LIBRARY_COLORS.get(gs, '#95A5A6') for gs in top25['Gene_set']]

    y_pos = range(len(top25))
    bars = ax.barh(y_pos, top25['neg_log_p'], color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{t[:45]}..." if len(t) > 45 else t for t in top25['Term']], fontsize=9)
    ax.set_xlabel('-log10(Adjusted P-value)', fontsize=12)
    ax.set_title('Top 25 Enriched Terms\nAll 41 Risk Genes Combined', fontsize=14, fontweight='bold')
    ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p=0.05')
    ax.grid(axis='x', alpha=0.3)

    # Add gene counts on bars
    for i, (idx, row) in enumerate(top25.iterrows()):
        ax.text(row['neg_log_p'] + 0.2, i, f"AD:{row['N_AD']} T2D:{row['N_T2D']}",
                va='center', fontsize=7, color='black')

    plt.tight_layout()
    plt.savefig(ENRICH_DIR / "enrichr_all_genes_top25.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: enrichr_all_genes_top25.png")

# =============================================================================
# VISUALIZATION 2: TISSUE EXPRESSION
# =============================================================================
if len(tissue_df) > 0:
    print("\n[7] Creating tissue expression visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    # GTEx Up
    gtex_up = tissue_df[tissue_df['Gene_set'] == 'GTEx_Tissue_Expression_Up']
    if len(gtex_up) > 0:
        ax = axes[0]
        top15 = gtex_up.nsmallest(15, 'Adjusted P-value').copy()
        top15['neg_log_p'] = -np.log10(top15['Adjusted P-value'].clip(lower=1e-50))
        top15 = top15.sort_values('neg_log_p', ascending=True)

        y_pos = range(len(top15))
        ax.barh(y_pos, top15['neg_log_p'], color=COLORS['ad_only'], alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([t[:40] for t in top15['Term']], fontsize=9)
        ax.set_xlabel('-log10(Adjusted P-value)', fontsize=11)
        ax.set_title('GTEx Tissue Expression (Upregulated)\nAll 41 Risk Genes', fontsize=12, fontweight='bold')
        ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.5)
        ax.grid(axis='x', alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'No significant GTEx Up terms', ha='center', va='center', fontsize=12)

    # ARCHS4 Tissues
    archs4 = tissue_df[tissue_df['Gene_set'] == 'ARCHS4_Tissues']
    if len(archs4) > 0:
        ax = axes[1]
        top15 = archs4.nsmallest(15, 'Adjusted P-value').copy()
        top15['neg_log_p'] = -np.log10(top15['Adjusted P-value'].clip(lower=1e-50))
        top15 = top15.sort_values('neg_log_p', ascending=True)

        y_pos = range(len(top15))
        ax.barh(y_pos, top15['neg_log_p'], color=COLORS['t2d_only'], alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([t[:40] for t in top15['Term']], fontsize=9)
        ax.set_xlabel('-log10(Adjusted P-value)', fontsize=11)
        ax.set_title('ARCHS4 Tissue Expression\nAll 41 Risk Genes', fontsize=12, fontweight='bold')
        ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.5)
        ax.grid(axis='x', alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No significant ARCHS4 terms', ha='center', va='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(ENRICH_DIR / "tissue_expression.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: tissue_expression.png")

# =============================================================================
# VISUALIZATION 3: DISEASE ENRICHMENT
# =============================================================================
if len(disease_df) > 0:
    print("\n[8] Creating disease enrichment visualization...")

    fig, ax = plt.subplots(figsize=(14, 10))

    top20 = disease_df.nsmallest(20, 'Adjusted P-value').copy()
    top20['neg_log_p'] = -np.log10(top20['Adjusted P-value'].clip(lower=1e-50))
    top20 = top20.sort_values('neg_log_p', ascending=True)

    bar_colors = [LIBRARY_COLORS.get(gs, '#95A5A6') for gs in top20['Gene_set']]

    y_pos = range(len(top20))
    ax.barh(y_pos, top20['neg_log_p'], color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([t[:50] for t in top20['Term']], fontsize=9)
    ax.set_xlabel('-log10(Adjusted P-value)', fontsize=11)
    ax.set_title('Disease Enrichment\nAll 41 Risk Genes', fontsize=13, fontweight='bold')
    ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.5)
    ax.grid(axis='x', alpha=0.3)

    # Legend
    legend_elements = [Patch(facecolor=LIBRARY_COLORS.get(l, '#95A5A6'), label=l.replace('_', ' ')[:25])
                      for l in DISEASE_LIBS if l in top20['Gene_set'].values]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(ENRICH_DIR / "disease_enrichment.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: disease_enrichment.png")

# =============================================================================
# VISUALIZATION 4: PATHWAY ENRICHMENT
# =============================================================================
if len(pathway_df) > 0:
    print("\n[9] Creating pathway enrichment visualization...")

    fig, ax = plt.subplots(figsize=(14, 10))

    top20 = pathway_df.nsmallest(20, 'Adjusted P-value').copy()
    top20['neg_log_p'] = -np.log10(top20['Adjusted P-value'].clip(lower=1e-50))
    top20 = top20.sort_values('neg_log_p', ascending=True)

    bar_colors = [LIBRARY_COLORS.get(gs, '#95A5A6') for gs in top20['Gene_set']]

    y_pos = range(len(top20))
    ax.barh(y_pos, top20['neg_log_p'], color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([t[:50] for t in top20['Term']], fontsize=9)
    ax.set_xlabel('-log10(Adjusted P-value)', fontsize=11)
    ax.set_title('Pathway Enrichment\nAll 41 Risk Genes', fontsize=13, fontweight='bold')
    ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.5)
    ax.grid(axis='x', alpha=0.3)

    # Legend
    legend_elements = [Patch(facecolor=LIBRARY_COLORS.get(l, '#95A5A6'), label=l.replace('_', ' ')[:25])
                      for l in PATHWAY_LIBS if l in top20['Gene_set'].values]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(ENRICH_DIR / "pathway_enrichment.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: pathway_enrichment.png")

# =============================================================================
# VISUALIZATION 5: LIBRARY HEATMAP
# =============================================================================
print("\n[10] Creating library heatmap...")

fig, ax = plt.subplots(figsize=(12, 14))

# Count significant terms per library
lib_counts = {}
for lib in ALL_GENE_SETS:
    lib_df = enrichr_all[enrichr_all['Gene_set'] == lib] if len(enrichr_all) > 0 else pd.DataFrame()
    if len(lib_df) > 0:
        lib_counts[lib] = {
            'N_terms': len(lib_df),
            'Best_P': -np.log10(lib_df['Adjusted P-value'].min()),
            'AD_genes': lib_df['N_AD'].max() if 'N_AD' in lib_df.columns else 0,
            'T2D_genes': lib_df['N_T2D'].max() if 'N_T2D' in lib_df.columns else 0,
        }
    else:
        lib_counts[lib] = {'N_terms': 0, 'Best_P': 0, 'AD_genes': 0, 'T2D_genes': 0}

lib_summary_df = pd.DataFrame(lib_counts).T
lib_summary_df = lib_summary_df.sort_values('Best_P', ascending=True)

# Plot
y_pos = range(len(lib_summary_df))
bar_colors = [LIBRARY_COLORS.get(l, '#95A5A6') for l in lib_summary_df.index]

ax.barh(y_pos, lib_summary_df['Best_P'], color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels([l.replace('_', ' ')[:35] for l in lib_summary_df.index], fontsize=9)
ax.set_xlabel('Best -log10(Adjusted P-value)', fontsize=11)
ax.set_title('Enrichment by Library\nAll 41 Risk Genes', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add term counts
for i, (lib, row) in enumerate(lib_summary_df.iterrows()):
    if row['N_terms'] > 0:
        ax.text(row['Best_P'] + 0.2, i, f"n={int(row['N_terms'])}", va='center', fontsize=8)

plt.tight_layout()
plt.savefig(ENRICH_DIR / "library_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: library_summary.png")

# =============================================================================
# VISUALIZATION 6: PPI NETWORK
# =============================================================================
print("\n[11] Creating PPI network visualization...")

fig, ax = plt.subplots(figsize=(24, 20))

if G.number_of_edges() > 0:
    # Layout
    pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42)

    # Separate seed and partner nodes
    seed_nodes = [n for n in G.nodes() if G.nodes[n].get('is_seed', False)]
    partner_nodes = [n for n in G.nodes() if not G.nodes[n].get('is_seed', False)]

    # Node colors and sizes by type
    seed_colors = []
    seed_sizes = []
    for node in seed_nodes:
        node_type = G.nodes[node].get('node_type', 'seed')
        if node_type == 'shared':
            seed_colors.append(COLORS['shared'])
            seed_sizes.append(1600)
        elif node_type == 'ad_only':
            seed_colors.append(COLORS['ad_only'])
            seed_sizes.append(1400)
        elif node_type == 't2d_only':
            seed_colors.append(COLORS['t2d_only'])
            seed_sizes.append(1400)
        else:
            seed_colors.append(COLORS['extra5'])
            seed_sizes.append(1200)

    # Draw edges
    edges = G.edges(data=True)
    edge_weights = [max(d.get('weight', 0.7) * 2, 0.5) for _, _, d in edges]
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=edge_weights, edge_color='#888888', ax=ax)

    # Draw partner nodes first
    if partner_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=partner_nodes, node_color='#DDDDDD',
                               node_size=350, alpha=0.6, edgecolors='#999999', linewidths=0.5, ax=ax)

    # Draw seed nodes on top
    nx.draw_networkx_nodes(G, pos, nodelist=seed_nodes, node_color=seed_colors,
                           node_size=seed_sizes, alpha=0.95, edgecolors='black', linewidths=2, ax=ax)

    # Labels for seed nodes
    seed_labels = {n: n for n in seed_nodes}
    nx.draw_networkx_labels(G, pos, labels=seed_labels, font_size=10, font_weight='bold', ax=ax)

    # Legend
    legend_elements = [
        Patch(facecolor=COLORS['shared'], edgecolor='black', linewidth=2, label=f'Shared ({len([n for n in seed_nodes if G.nodes[n]["node_type"]=="shared"])} genes)'),
        Patch(facecolor=COLORS['ad_only'], edgecolor='black', linewidth=2, label=f'AD Only ({len([n for n in seed_nodes if G.nodes[n]["node_type"]=="ad_only"])} genes)'),
        Patch(facecolor=COLORS['t2d_only'], edgecolor='black', linewidth=2, label=f'T2D Only ({len([n for n in seed_nodes if G.nodes[n]["node_type"]=="t2d_only"])} genes)'),
        Patch(facecolor='#DDDDDD', edgecolor='#999999', label=f'PPI Partners ({partner_count} proteins)'),
        Line2D([0], [0], color='#888888', linewidth=3, alpha=0.4, label='Interaction (score≥700)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12, framealpha=0.95)

    ax.set_title(f'Protein-Protein Interaction Network (STRING score ≥ 700)\n{seed_count} seed genes, {partner_count} partners, {G.number_of_edges()} interactions',
                 fontsize=18, fontweight='bold')
else:
    ax.text(0.5, 0.5, 'No PPI interactions found', ha='center', va='center', fontsize=14, transform=ax.transAxes)

ax.axis('off')
plt.tight_layout()
plt.savefig(ENRICH_DIR / "ppi_network.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: ppi_network.png")

# =============================================================================
# VISUALIZATION 7: GENE-CATEGORY BREAKDOWN
# =============================================================================
print("\n[12] Creating gene category breakdown...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 1. Pie chart of gene categories
ax = axes[0, 0]
cat_counts = gene_cat_df['category'].value_counts()
colors_pie = [COLORS.get(c, '#95A5A6') for c in cat_counts.index]
ax.pie(cat_counts.values, labels=[f"{c.replace('_', ' ').title()}\n({v} genes)" for c, v in cat_counts.items()],
       colors=colors_pie, autopct='%1.1f%%', startangle=90)
ax.set_title('Gene Distribution by Category\n(41 Risk Genes)', fontsize=12, fontweight='bold')

# 2. Top pathways by category
ax = axes[0, 1]
if len(pathway_df) > 0:
    top10 = pathway_df.nsmallest(10, 'Adjusted P-value').copy()

    # Color bars by which category dominates
    bar_colors = []
    for _, row in top10.iterrows():
        if row.get('N_Shared', 0) > 0:
            bar_colors.append(COLORS['shared'])
        elif row.get('N_AD', 0) > row.get('N_T2D', 0):
            bar_colors.append(COLORS['ad_only'])
        else:
            bar_colors.append(COLORS['t2d_only'])

    top10 = top10.sort_values('Adjusted P-value', ascending=False)
    y_pos = range(len(top10))
    ax.barh(y_pos, -np.log10(top10['Adjusted P-value'].clip(lower=1e-50)), color=bar_colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([t[:35] for t in top10['Term']], fontsize=8)
    ax.set_xlabel('-log10(Adj. P-value)', fontsize=10)
    ax.set_title('Top 10 Pathways\n(colored by dominant gene category)', fontsize=11, fontweight='bold')
else:
    ax.text(0.5, 0.5, 'No pathway data', ha='center', va='center')

# 3. Top diseases by category
ax = axes[1, 0]
if len(disease_df) > 0:
    top10 = disease_df.nsmallest(10, 'Adjusted P-value').copy()

    bar_colors = []
    for _, row in top10.iterrows():
        if row.get('N_Shared', 0) > 0:
            bar_colors.append(COLORS['shared'])
        elif row.get('N_AD', 0) > row.get('N_T2D', 0):
            bar_colors.append(COLORS['ad_only'])
        else:
            bar_colors.append(COLORS['t2d_only'])

    top10 = top10.sort_values('Adjusted P-value', ascending=False)
    y_pos = range(len(top10))
    ax.barh(y_pos, -np.log10(top10['Adjusted P-value'].clip(lower=1e-50)), color=bar_colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([t[:35] for t in top10['Term']], fontsize=8)
    ax.set_xlabel('-log10(Adj. P-value)', fontsize=10)
    ax.set_title('Top 10 Diseases\n(colored by dominant gene category)', fontsize=11, fontweight='bold')
else:
    ax.text(0.5, 0.5, 'No disease data', ha='center', va='center')

# 4. Top tissues by category
ax = axes[1, 1]
if len(tissue_df) > 0:
    top10 = tissue_df.nsmallest(10, 'Adjusted P-value').copy()

    bar_colors = []
    for _, row in top10.iterrows():
        if row.get('N_Shared', 0) > 0:
            bar_colors.append(COLORS['shared'])
        elif row.get('N_AD', 0) > row.get('N_T2D', 0):
            bar_colors.append(COLORS['ad_only'])
        else:
            bar_colors.append(COLORS['t2d_only'])

    top10 = top10.sort_values('Adjusted P-value', ascending=False)
    y_pos = range(len(top10))
    ax.barh(y_pos, -np.log10(top10['Adjusted P-value'].clip(lower=1e-50)), color=bar_colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([t[:35] for t in top10['Term']], fontsize=8)
    ax.set_xlabel('-log10(Adj. P-value)', fontsize=10)
    ax.set_title('Top 10 Tissues\n(colored by dominant gene category)', fontsize=11, fontweight='bold')
else:
    ax.text(0.5, 0.5, 'No tissue data', ha='center', va='center')

plt.tight_layout()
plt.savefig(ENRICH_DIR / "gene_category_breakdown.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: gene_category_breakdown.png")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n[13] Creating summary...")

summary_rows = []
for lib_type, libs in [('Pathway', PATHWAY_LIBS), ('GO', GO_LIBS), ('Disease', DISEASE_LIBS),
                        ('Tissue', TISSUE_LIBS), ('TF', TF_LIBS), ('Drug', DRUG_LIBS)]:
    for lib in libs:
        lib_df = enrichr_all[enrichr_all['Gene_set'] == lib] if len(enrichr_all) > 0 else pd.DataFrame()
        if len(lib_df) > 0:
            best_row = lib_df.iloc[0]
            summary_rows.append({
                'Type': lib_type,
                'Library': lib,
                'N_Terms': len(lib_df),
                'Top_Term': best_row['Term'][:60],
                'Best_Adj_P': best_row['Adjusted P-value'],
                'Genes_in_Term': best_row['Genes'][:50],
                'N_AD': best_row.get('N_AD', 0),
                'N_T2D': best_row.get('N_T2D', 0),
                'N_Shared': best_row.get('N_Shared', 0),
            })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(ENRICH_DIR / "enrichment_summary.csv", index=False)
print(f"  Saved: enrichment_summary.csv")

# =============================================================================
# FINAL OUTPUT
# =============================================================================
print("\n" + "="*70)
print("OUTPUT FILES")
print("="*70)
print(f"""
Gene Categories:
  gene_categories.csv          - Gene to category mapping

Enrichr Results (All 41 Genes):
  enrichr_all_41_genes.csv     - Full enrichment results
  enrichr_pathways.csv         - Pathway enrichment
  enrichr_go_terms.csv         - GO enrichment
  enrichr_diseases.csv         - Disease enrichment
  enrichr_tissues.csv          - Tissue expression
  enrichr_transcription_factors.csv - TF enrichment
  enrichr_drugs.csv            - Drug/perturbation enrichment

STRING PPI:
  ppi_interactions.csv         - PPI edges ({len(ppi_df)} interactions)
  ppi_network.graphml          - Network file
  string_enrichment.csv        - STRING enrichment

Visualizations:
  enrichr_all_genes_top25.png  - Top 25 enriched terms
  tissue_expression.png        - GTEx & ARCHS4 tissue expression
  disease_enrichment.png       - Disease enrichment
  pathway_enrichment.png       - Pathway enrichment
  library_summary.png          - Enrichment by library
  ppi_network.png              - PPI network graph
  gene_category_breakdown.png  - Category breakdown
""")

print("="*70)
print("ANALYSIS COMPLETE")
print("="*70)
