"""
TASEP Models - Python library for TASEP (Totally Asymmetric Simple Exclusion Process) simulations.

This package provides tools for modeling ribosome traffic and translation dynamics
on mRNA transcripts using both stochastic (SSA) and deterministic (ODE) approaches.
"""

from .models import (
    # Core simulation functions
    TASEP_SSA,
    simulate_TASEP_SSA,
    TASEP_ODE,
    simulate_TASEP_ODE,
    
    # Sequence analysis
    read_sequence,
    read_gene_sequence,
    read_gene_sequence_return_probes,
    create_probe_vector,
    calculate_codon_elongation_rates,
    
    # CAI and codon optimization
    compute_CAI,
    sliding_window_cai,
    plot_sliding_window_cai,
    calculate_codon_usage,
    plot_codon_usage_grouped,
    optimize_sequence,
    deoptimize_sequence,
    
    # Visualization
    plot_trajectories,
    plot_dual_signal_trajectories,
    plot_RibosomeMovement,
    plot_RibosomeMovement_and_Microscope,
    plot_plasmid,
    
    # Utility functions
    simulate_missing_data,
    simulate_photobleaching_in_trajectories,
    correct_photobleaching_in_trajectories,
    delay_signal,
    find_TAG_location,
    
    # Constants and data
    GFP_TAG,
    HA_TAG,
    U_TAG,
    SUN_TAG,
    ALFA_TAG,
    MCHERRY_TAG,
    XBP1,
    tag_dict,
    synonymous_codons,
    codon_frequency_dict,
    download_human_genome_cds,
    HUMAN_GENOME_PATH,
)

from .codon_weights import HUMAN_CODON_WEIGHTS

__version__ = "0.1.0"
__author__ = "Luis U. Aguilera"
__all__ = [
    # Core simulation functions
    "TASEP_SSA",
    "simulate_TASEP_SSA",
    "TASEP_ODE",
    "simulate_TASEP_ODE",
    
    # Sequence analysis
    "read_sequence",
    "read_gene_sequence",
    "read_gene_sequence_return_probes",
    "create_probe_vector",
    "calculate_codon_elongation_rates",
    
    # CAI and codon optimization
    "compute_CAI",
    "sliding_window_cai",
    "plot_sliding_window_cai",
    "calculate_codon_usage",
    "plot_codon_usage_grouped",
    "optimize_sequence",
    "deoptimize_sequence",
    
    # Visualization
    "plot_trajectories",
    "plot_dual_signal_trajectories",
    "plot_RibosomeMovement",
    "plot_RibosomeMovement_and_Microscope",
    "plot_plasmid",
    
    # Utility functions
    "simulate_missing_data",
    "simulate_photobleaching_in_trajectories",
    "correct_photobleaching_in_trajectories",
    "delay_signal",
    "find_TAG_location",
    
    # Constants and data
    "GFP_TAG",
    "HA_TAG",
    "U_TAG",
    "SUN_TAG",
    "ALFA_TAG",
    "MCHERRY_TAG",
    "XBP1",
    "tag_dict",
    "synonymous_codons",
    "codon_frequency_dict",
    "HUMAN_CODON_WEIGHTS",
    "download_human_genome_cds",
    "HUMAN_GENOME_PATH",
]
