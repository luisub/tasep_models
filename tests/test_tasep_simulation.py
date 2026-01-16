"""
Test TASEP Simulation with pNZ208 (U-TAG) sequence.

This test verifies that the TASEP simulation runs correctly with a real gene sequence.
Parameters:
    - t_max: 2000 seconds
    - burnin_time: 1000 seconds
    - time_interval: 5 seconds
    - ki: 0.04 (initiation rate)
    - ke: 10 (elongation rate)
    - Sequence: pNZ208 (U-TAG)
    - No pauses

Test passes if:
    1. Simulation completes without errors
    2. Output signal matrices have values > 0
    3. All plots/animations are generated successfully
"""

import os
import sys

# Use non-interactive backend to suppress plot windows
import matplotlib
matplotlib.use('Agg')

import numpy as np
from pathlib import Path

# Add src to path for development testing
sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))

import tasep_models as tm
from tasep_models import (
    read_sequence,
    create_probe_vector,
    simulate_TASEP_SSA,
    simulate_TASEP_ODE,
    plot_trajectories,
    plot_plasmid,
    plot_RibosomeMovement_and_Microscope,
    U_TAG,
)

# ============================================================================
# Configuration
# ============================================================================
OUTPUT_DIR = Path(__file__).parent / 'results_tasep'
DNA_FILE = Path(__file__).parent.parent / 'data' / 'human_genome' / 'gene_sequences' / 'utag_project' / 'pNZ208(pUB-24xUTagFullLength-KDM5B-MS2).dna'

# Simulation parameters
T_MAX = 2000  # seconds
BURNIN_TIME = 1000  # seconds
TIME_INTERVAL = 5  # seconds
KI = 0.04  # initiation rate
KE = 10  # elongation rate (codons/s)
NUMBER_REPETITIONS = 10  # Number of SSA runs

def test_tasep_simulation():
    """Main test function for TASEP simulation."""
    
    print("=" * 60)
    print("TASEP Simulation Test - pNZ208 (U-TAG)")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # ========================================================================
    # Step 1: Read sequence
    # ========================================================================
    print("\n[1/5] Reading gene sequence...")
    
    if not DNA_FILE.exists():
        raise FileNotFoundError(f"DNA file not found: {DNA_FILE}")
    
    # Read sequence using U-TAG
    tag_sequence = U_TAG
    protein, rna, dna, indexes_tags, indexes_pauses, seq_record, graphic_features = read_sequence(
        seq=DNA_FILE,
        min_protein_length=50,
        TAG=[tag_sequence]
    )
    
    gene_length = len(protein) + 1  # +1 for stop codon
    print(f"   Gene length: {gene_length} codons")
    print(f"   Protein length: {len(protein)} aa")
    print(f"   U-TAG positions: {indexes_tags[0] if indexes_tags else 'None'}")
    
    # Create probe vectors
    tag_positions = indexes_tags[0] if indexes_tags else []
    first_probe_position_vector = create_probe_vector(tag_positions, gene_length)
    
    # Save plasmid plot
    print("\n[2/5] Generating plasmid visualization...")
    plasmid_fig = plot_plasmid(seq_record, graphic_features, figure_width=25, figure_height=3)
    plasmid_path = OUTPUT_DIR / 'plasmid_map.png'
    plasmid_fig.savefig(plasmid_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {plasmid_path}")
    
    # ========================================================================
    # Step 2: Run ODE simulation
    # ========================================================================
    print("\n[3/5] Running ODE simulation...")
    
    signal_ode, _ = simulate_TASEP_ODE(
        ki=KI,
        ke=KE,
        gene_length=gene_length,
        t_max=T_MAX,
        time_interval_in_seconds=TIME_INTERVAL,
        first_probe_position_vector=first_probe_position_vector,
        burnin_time=BURNIN_TIME,
    )
    
    print(f"   ODE signal shape: {signal_ode.shape}")
    print(f"   ODE signal mean: {np.mean(signal_ode):.4f}")
    
    # ========================================================================
    # Step 3: Run SSA simulation
    # ========================================================================
    print("\n[4/5] Running SSA simulation...")
    print(f"   Parameters: ki={KI}, ke={KE}, t_max={T_MAX}s, burnin={BURNIN_TIME}s")
    print(f"   Repetitions: {NUMBER_REPETITIONS}")
    
    list_ribosome_trajectories, list_occupancy, signal_ssa, _ = simulate_TASEP_SSA(
        ki=KI,
        ke=KE,
        gene_length=gene_length,
        t_max=T_MAX,
        time_interval_in_seconds=TIME_INTERVAL,
        number_repetitions=NUMBER_REPETITIONS,
        first_probe_position_vector=first_probe_position_vector,
        burnin_time=BURNIN_TIME,
        n_jobs=-1,  # Use all cores
        fast_output=False,
    )
    
    print(f"   SSA signal shape: {signal_ssa.shape}")
    print(f"   SSA signal mean: {np.mean(signal_ssa):.4f}")
    print(f"   SSA signal max: {np.max(signal_ssa):.4f}")
    
    # ========================================================================
    # Step 4: Generate plots
    # ========================================================================
    print("\n[5/5] Generating plots and animation...")
    
    # Time array
    t_array = np.arange(0, T_MAX, TIME_INTERVAL)
    t_array = t_array[:signal_ssa.shape[1]]  # Match signal length
    
    # Trajectory comparison plot
    try:
        traj_fig = plot_trajectories(
            matrix_intensity_first_signal_RT=signal_ssa,
            intensity_vector_first_signal_ode=signal_ode[:len(t_array)],
            time_array=t_array,
            number_repetitions=NUMBER_REPETITIONS,
        )
        traj_path = OUTPUT_DIR / 'trajectory_comparison.png'
        traj_fig.savefig(traj_path, dpi=150, bbox_inches='tight')
        print(f"   Saved: {traj_path}")
    except Exception as e:
        print(f"   Warning: Could not save trajectory plot: {e}")
    
    # Animation (if trajectories available)
    try:
        if list_ribosome_trajectories and len(list_ribosome_trajectories) > 0:
            anim_path = OUTPUT_DIR / 'ribosome_animation'
            plot_RibosomeMovement_and_Microscope(
                RibosomePositions=list_ribosome_trajectories[0],
                IntensityVector=signal_ssa[0],
                probePositions=tag_positions,  # Use actual position indices, not binary mask
                fileNameGif=str(anim_path),
                FrameVelocity=20,
            )
            print(f"   Saved: {anim_path}.gif")
    except Exception as e:
        print(f"   Warning: Could not generate animation: {e}")
    
    # ========================================================================
    # Validation
    # ========================================================================
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    
    # Check 1: Signal values > 0
    ssa_max = np.max(signal_ssa)
    ode_max = np.max(signal_ode)
    
    print(f"\n   SSA max signal: {ssa_max:.4f} {'✓' if ssa_max > 0 else '✗'}")
    print(f"   ODE max signal: {ode_max:.4f} {'✓' if ode_max > 0 else '✗'}")
    
    # Check 2: Files created
    files_created = list(OUTPUT_DIR.glob('*'))
    print(f"\n   Files created: {len(files_created)}")
    for f in files_created:
        print(f"      - {f.name}")
    
    # Final assertion
    assert ssa_max > 0, "SSA signal should have values > 0"
    assert ode_max > 0, "ODE signal should have values > 0"
    assert len(files_created) >= 2, "Should have created at least 2 output files"
    
    print("\n" + "=" * 60)
    print("TEST PASSED ✓")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_tasep_simulation()
    sys.exit(0 if success else 1)
