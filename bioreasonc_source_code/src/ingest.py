"""
Data Ingestion Module for BioREASONC-Bench

DEPRECATED: This module has moved to src/Causal_KG/data_loader.py
This file is kept for backwards compatibility.

New code should use:
    from src.Causal_KG import DataIngestor, load_data
"""

# Re-export from new location for backwards compatibility
from .Causal_KG.data_loader import DataIngestor, load_data

__all__ = ['DataIngestor', 'load_data']
