# TASEP Models: Cotranslational Folding Simulation Library

**Current Version:** 0.1.0  
**Author:** Luis U. Aguilera  
**License:** BSD-3-Clause

**TASEP Models** is a robust Python library for simulating ribosome traffic and translation dynamics. It provides high-performance Stochastic Simulation Algorithms (SSA) via Numba and deterministic Ordinary Differential Equation (ODE) solvers to model the Totally Asymmetric Simple Exclusion Process (TASEP).

## Features

- **High-Performance Simulation**: Numba-accelerated Gillespie SSA for accurate stochastic modeling of ribosome movement.
- **Deterministic Solvers**: Fast ODE approximations for mean-field behavior.
- **Sequence Analysis**: Tools to calculate Codon Adaptation Index (CAI) and map codon usage to elongation rates using human genome data.
- **Optimization Tools**: Functions to `optimize_sequence()` or `deoptimize_sequence()` coding sequences for translation efficiency.
- **Visualization**: Built-in plotting for trajectories, kymographs, and dual-channel fluorescence signals (e.g., SunTag, MoonTag).
- **Self-Healing Data**: Automatically downloads required human genome CDS data from Ensembl when needed.

## Installation

### Prerequisites

- Python 3.10+
- [Conda](https://docs.conda.io/en/latest/) (recommended)

### Setup

1. **Clone or Download** the repository to your local machine.

2. **Create the Environment**:
    Use the provided `environment.yml` to set up a clean environment with all dependencies.

    ```bash
    conda env create -f environment.yml
    conda activate tasep_models
    ```

3. **Install the Package**:
    Install in editable mode (`-e`) to allow for development and easy updates.

    ```bash
    pip install -e .
    ```

### Dependencies

The following packages are required (automatically installed):

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 1.26.4 | Numerical computing |
| pandas | 2.2.2 | Data manipulation |
| scipy | 1.13.1 | Scientific computing |
| matplotlib | 3.9.2 | Visualization |
| numba | 0.60.0 | JIT compilation |
| llvmlite | 0.43.0 | LLVM bindings for numba |
| biopython | ≥1.81 | Sequence parsing |
| joblib | ≥1.3 | Parallel processing |
| dna-features-viewer | ≥3.1 | Plasmid visualization |
| snapgene-reader | ≥0.1 | Reading .dna files |

## Quick Start

Here is a complete example of running a comparative simulation (SSA vs ODE) for a hypothetical gene.

```python
import numpy as np
import tasep_models as tm

# 1. Define Parameters
gene_length = 300       # Length in codons
ki = 0.05               # Initiation rate (1/s)
ke = 10.0               # Elongation rate (codons/s)
t_max = 600             # Simulation time (s)

# 2. Define Probe Positions (e.g., for SunTag signal simulation)
first_probe_vec = np.zeros(gene_length)
first_probe_vec[50:] = 1 

# 3. Run Stochastic Simulation (SSA)
trajectories, occupancy, signal1_ssa, signal2_ssa = tm.simulate_TASEP_SSA(
    ki=ki, 
    ke=ke, 
    gene_length=gene_length, 
    t_max=t_max,
    first_probe_position_vector=first_probe_vec,
    number_repetitions=20,
    n_jobs=-1
)

# 4. Run Deterministic Simulation (ODE)
signal1_ode, _ = tm.simulate_TASEP_ODE(
    ki=ki, 
    ke=ke, 
    gene_length=gene_length, 
    t_max=t_max,
    first_probe_position_vector=first_probe_vec
)

# 5. Visualize Comparison
time_array = np.arange(0, t_max + 1, 1)
time_array = time_array[:signal1_ssa.shape[1]] 

tm.plot_trajectories(
    matrix_intensity_first_signal_RT=signal1_ssa, 
    intensity_vector_first_signal_ode=signal1_ode, 
    time_array=time_array, 
    number_repetitions=20
)
```

## Data Management

The library relies on `Homo_sapiens.GRCh38.cds.all.fa` for calculating accurate codon weights and CAI.

- **Automatic Download**: You do *not* need to manually download this file. The first time you run a function like `optimize_sequence()` or `compute_CAI()`, the library will detect if the file is missing and automatically download it (~177MB) from Ensembl to the `data/human_genome/` directory.
- **Location**: `tasep_models/data/human_genome/`

## Project Structure

```text
tasep_models/
├── src/
│   └── tasep_models/       # Source code
│       ├── __init__.py     # Exports public API
│       ├── models.py       # Core logic (SSA, ODE, Sequence Utils)
│       └── codon_weights.py # Human codon weights data
├── notebooks/              # Interactive examples
│   ├── TASEP_Simulation.ipynb      # Deep dive into simulation & plotting
│   ├── Codon_Adaptation_Index.ipynb # CAI calculation tutorial
│   └── Codon_Optimization.ipynb    # Sequence optimization tutorial
├── tests/                  # Unit tests
│   └── test_tasep_simulation.py
├── data/                   # Data storage (gitignored)
│   └── human_genome/       # Auto-downloaded genome data
├── environment.yml         # Conda environment definition
├── pyproject.toml          # Package configuration (PEP 517)
├── LICENSE                 # BSD-3-Clause license
└── README.md
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `TASEP_Simulation.ipynb` | Full SSA/ODE simulation workflow |
| `Codon_Adaptation_Index.ipynb` | Calculate and visualize CAI |
| `Codon_Optimization.ipynb` | Optimize/deoptimize sequences |

## API Reference

### Simulation Functions

- `simulate_TASEP_SSA()` - Stochastic simulation
- `simulate_TASEP_ODE()` - Deterministic simulation

### Sequence Analysis

- `read_sequence()` - Parse DNA/plasmid files
- `compute_CAI()` - Calculate Codon Adaptation Index
- `sliding_window_cai()` - CAI along the gene
- `optimize_sequence()` - Maximize translation efficiency
- `deoptimize_sequence()` - Minimize translation efficiency

### Visualization

- `plot_trajectories()` - SSA vs ODE comparison
- `plot_RibosomeMovement_and_Microscope()` - Animated ribosome movement
- `plot_plasmid()` - Plasmid map visualization
- `plot_codon_usage_grouped()` - Codon frequency bar plots

## License

BSD 3-Clause License. Copyright (c) 2026 Luis U. Aguilera.

This project is a standalone component originally derived from the MicroLive project.
