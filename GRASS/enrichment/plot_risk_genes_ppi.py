#!/usr/bin/env python3
"""
Plot PPI network for 41 risk genes only with category colors.
Colors from config.py:
- shared: #FF7E79 (coral red)
- ad_only: #B18CC7 (purple)
- t2d_only: #80B4E1 (blue)
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import requests
import time

# Colors from config.py
COLORS = {
    'shared': '#FF7E79',    # coral red
    'ad_only': '#B18CC7',   # purple
    't2d_only': '#80B4E1'   # blue
}

def get_string_interactions(genes, species=9606, score_threshold=400):
    """Get PPI interactions from STRING database for given genes."""
    string_api_url = "https://string-db.org/api/json/network"

    params = {
        "identifiers": "%0d".join(genes),
        "species": species,
        "required_score": score_threshold,
        "network_type": "physical",
        "caller_identity": "GRASS_analysis"
    }

    try:
        response = requests.post(string_api_url, data=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"STRING API error: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching STRING data: {e}")
        return []

def main():
    # Load gene categories
    gene_df = pd.read_csv('/ibex/user/alsaedsb/Causal_Benchmark/bioreasonc-bench/GRASS/enrichment/gene_categories.csv')

    # Create gene to category mapping
    gene_category = dict(zip(gene_df['gene'], gene_df['category']))
    all_genes = list(gene_df['gene'])

    print(f"Total risk genes: {len(all_genes)}")
    print(f"AD-only: {sum(1 for g in gene_category.values() if g == 'ad_only')}")
    print(f"T2D-only: {sum(1 for g in gene_category.values() if g == 't2d_only')}")
    print(f"Shared: {sum(1 for g in gene_category.values() if g == 'shared')}")

    # Get STRING interactions
    print("\nFetching PPI interactions from STRING...")
    interactions = get_string_interactions(all_genes, score_threshold=400)

    # Build network
    G = nx.Graph()

    # Add all genes as nodes
    for gene in all_genes:
        G.add_node(gene, category=gene_category[gene])

    # Add edges from STRING interactions
    edges_added = 0
    for interaction in interactions:
        gene1 = interaction.get('preferredName_A', interaction.get('stringId_A', '').split('.')[1] if '.' in interaction.get('stringId_A', '') else '')
        gene2 = interaction.get('preferredName_B', interaction.get('stringId_B', '').split('.')[1] if '.' in interaction.get('stringId_B', '') else '')
        score = interaction.get('score', 0)

        # Only add edges between our risk genes
        if gene1 in gene_category and gene2 in gene_category:
            G.add_edge(gene1, gene2, weight=score)
            edges_added += 1

    print(f"Edges between risk genes: {edges_added}")
    print(f"Network nodes: {G.number_of_nodes()}")
    print(f"Network edges: {G.number_of_edges()}")

    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))

    # Node colors based on category
    node_colors = [COLORS[gene_category[node]] for node in G.nodes()]

    # Node sizes - slightly larger for nodes with more connections
    degrees = dict(G.degree())
    node_sizes = [800 + degrees[node] * 150 for node in G.nodes()]

    # Layout - use spring layout with adjusted parameters
    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)

    # Draw edges
    edge_weights = [G[u][v].get('weight', 0.5) for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [0.5 + 2 * (w / max_weight) for w in edge_weights]

    nx.draw_networkx_edges(G, pos,
                           width=edge_widths,
                           alpha=0.4,
                           edge_color='gray',
                           ax=ax)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=0.9,
                           ax=ax)

    # Draw labels
    nx.draw_networkx_labels(G, pos,
                            font_size=8,
                            font_weight='bold',
                            ax=ax)

    # Create legend
    legend_elements = [
        plt.scatter([], [], c=COLORS['ad_only'], s=200, label=f'AD Only (n=13)', marker='o'),
        plt.scatter([], [], c=COLORS['t2d_only'], s=200, label=f'T2D Only (n=15)', marker='o'),
        plt.scatter([], [], c=COLORS['shared'], s=200, label=f'Shared (n=13)', marker='o')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12, framealpha=0.9)

    ax.set_title('PPI Network of 41 GRASS Risk Genes\n(AD, T2D, and Shared)',
                 fontsize=16, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('/ibex/user/alsaedsb/Causal_Benchmark/bioreasonc-bench/GRASS/enrichment/risk_genes_41_ppi_network.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('/ibex/user/alsaedsb/Causal_Benchmark/bioreasonc-bench/GRASS/enrichment/risk_genes_41_ppi_network.pdf',
                bbox_inches='tight', facecolor='white')
    print("\nSaved: risk_genes_41_ppi_network.png")
    print("Saved: risk_genes_41_ppi_network.pdf")

    # Also create a version with better separation by category
    fig2, ax2 = plt.subplots(1, 1, figsize=(18, 14))

    # Group genes by category for layout
    ad_genes = [g for g in G.nodes() if gene_category[g] == 'ad_only']
    t2d_genes = [g for g in G.nodes() if gene_category[g] == 't2d_only']
    shared_genes = [g for g in G.nodes() if gene_category[g] == 'shared']

    # Create custom positions - arrange by category
    pos2 = {}

    # AD genes on the left
    for i, gene in enumerate(ad_genes):
        angle = np.pi/2 + (i / len(ad_genes)) * np.pi  # left semicircle
        pos2[gene] = (np.cos(angle) * 2 - 1.5, np.sin(angle) * 2)

    # T2D genes on the right
    for i, gene in enumerate(t2d_genes):
        angle = -np.pi/2 + (i / len(t2d_genes)) * np.pi  # right semicircle
        pos2[gene] = (np.cos(angle) * 2 + 1.5, np.sin(angle) * 2)

    # Shared genes in the middle
    for i, gene in enumerate(shared_genes):
        angle = (i / len(shared_genes)) * 2 * np.pi
        pos2[gene] = (np.cos(angle) * 0.8, np.sin(angle) * 0.8)

    # Apply spring layout refinement while keeping general structure
    pos2 = nx.spring_layout(G, pos=pos2, k=1.5, iterations=50, seed=42)

    # Draw edges with color based on connection type
    for u, v in G.edges():
        cat_u = gene_category[u]
        cat_v = gene_category[v]

        if cat_u == cat_v:
            # Same category - use that category's color
            edge_color = COLORS[cat_u]
            alpha = 0.5
        else:
            # Cross-category - use gray
            edge_color = 'gray'
            alpha = 0.3

        weight = G[u][v].get('weight', 0.5)
        width = 0.5 + 2 * (weight / max_weight) if max_weight > 0 else 1

        ax2.plot([pos2[u][0], pos2[v][0]], [pos2[u][1], pos2[v][1]],
                 color=edge_color, alpha=alpha, linewidth=width, zorder=1)

    # Draw nodes
    for category, color in COLORS.items():
        genes_in_cat = [g for g in G.nodes() if gene_category[g] == category]
        if genes_in_cat:
            x = [pos2[g][0] for g in genes_in_cat]
            y = [pos2[g][1] for g in genes_in_cat]
            sizes = [800 + degrees[g] * 150 for g in genes_in_cat]
            ax2.scatter(x, y, c=color, s=sizes, alpha=0.9, zorder=2, edgecolors='white', linewidths=1.5)

    # Draw labels
    for node in G.nodes():
        ax2.annotate(node, pos2[node], fontsize=8, fontweight='bold',
                     ha='center', va='center', zorder=3)

    # Add category labels
    ax2.text(-2.5, 2.5, 'AD Only', fontsize=14, fontweight='bold', color=COLORS['ad_only'])
    ax2.text(1.8, 2.5, 'T2D Only', fontsize=14, fontweight='bold', color=COLORS['t2d_only'])
    ax2.text(-0.3, -2.5, 'Shared', fontsize=14, fontweight='bold', color=COLORS['shared'])

    # Legend
    legend_elements2 = [
        plt.scatter([], [], c=COLORS['ad_only'], s=200, label=f'AD Only (n={len(ad_genes)})', marker='o', edgecolors='white'),
        plt.scatter([], [], c=COLORS['t2d_only'], s=200, label=f'T2D Only (n={len(t2d_genes)})', marker='o', edgecolors='white'),
        plt.scatter([], [], c=COLORS['shared'], s=200, label=f'Shared (n={len(shared_genes)})', marker='o', edgecolors='white')
    ]
    ax2.legend(handles=legend_elements2, loc='upper right', fontsize=12, framealpha=0.9)

    ax2.set_title('PPI Network of 41 GRASS Risk Genes\n(Organized by Disease Category)',
                  fontsize=16, fontweight='bold')
    ax2.axis('off')
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-3.5, 3.5)

    plt.tight_layout()
    plt.savefig('/ibex/user/alsaedsb/Causal_Benchmark/bioreasonc-bench/GRASS/enrichment/risk_genes_41_ppi_organized.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('/ibex/user/alsaedsb/Causal_Benchmark/bioreasonc-bench/GRASS/enrichment/risk_genes_41_ppi_organized.pdf',
                bbox_inches='tight', facecolor='white')
    print("Saved: risk_genes_41_ppi_organized.png")
    print("Saved: risk_genes_41_ppi_organized.pdf")

    # Print network statistics
    print("\n" + "="*50)
    print("Network Statistics")
    print("="*50)

    # Count edges by category
    ad_ad = sum(1 for u, v in G.edges() if gene_category[u] == 'ad_only' and gene_category[v] == 'ad_only')
    t2d_t2d = sum(1 for u, v in G.edges() if gene_category[u] == 't2d_only' and gene_category[v] == 't2d_only')
    shared_shared = sum(1 for u, v in G.edges() if gene_category[u] == 'shared' and gene_category[v] == 'shared')
    ad_t2d = sum(1 for u, v in G.edges() if (gene_category[u] == 'ad_only' and gene_category[v] == 't2d_only') or
                                             (gene_category[u] == 't2d_only' and gene_category[v] == 'ad_only'))
    ad_shared = sum(1 for u, v in G.edges() if (gene_category[u] == 'ad_only' and gene_category[v] == 'shared') or
                                               (gene_category[u] == 'shared' and gene_category[v] == 'ad_only'))
    t2d_shared = sum(1 for u, v in G.edges() if (gene_category[u] == 't2d_only' and gene_category[v] == 'shared') or
                                                (gene_category[u] == 'shared' and gene_category[v] == 't2d_only'))

    print(f"\nEdge distribution:")
    print(f"  AD-AD edges: {ad_ad}")
    print(f"  T2D-T2D edges: {t2d_t2d}")
    print(f"  Shared-Shared edges: {shared_shared}")
    print(f"  AD-T2D edges: {ad_t2d}")
    print(f"  AD-Shared edges: {ad_shared}")
    print(f"  T2D-Shared edges: {t2d_shared}")

    # Degree statistics by category
    print(f"\nDegree statistics:")
    for category in ['ad_only', 't2d_only', 'shared']:
        genes_in_cat = [g for g in G.nodes() if gene_category[g] == category]
        degrees_cat = [degrees[g] for g in genes_in_cat]
        if degrees_cat:
            print(f"  {category}: mean={np.mean(degrees_cat):.2f}, max={max(degrees_cat)}, min={min(degrees_cat)}")

    # Top connected genes
    print(f"\nTop 10 most connected risk genes:")
    sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    for gene, deg in sorted_degrees:
        print(f"  {gene} ({gene_category[gene]}): {deg} connections")

    # Save network as GraphML
    nx.write_graphml(G, '/ibex/user/alsaedsb/Causal_Benchmark/bioreasonc-bench/GRASS/enrichment/risk_genes_41_ppi.graphml')
    print("\nSaved: risk_genes_41_ppi.graphml")

    plt.close('all')

if __name__ == "__main__":
    main()
