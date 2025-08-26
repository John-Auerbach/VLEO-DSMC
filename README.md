# VLEO-DSMC: Satellite Atmospheric Simulation with SPARTA

## Table of Contents
1. [Simulation Overview](#simulation-overview)
2. [Installing SPARTA](#1-install-sparta)
3. [Setting Up Python Environment](#2-set-up-python-environment)
4. [Running Simulations](#3-run-simulations)
5. [Converting Dump Data](#4-convert-dump-data-for-memory-efficient-analysis)
6. [Visualization Scripts](#5-visualization-scripts)

## Simulation Overview

### What is DSMC?

**Direct Simulation Monte Carlo (DSMC)** is a computational method for simulating rarefied gas flows where the continuum assumption breaks down. Instead of solving fluid equations, DSMC tracks individual representative particles (each particle is really a cluster of many, many actual particles) and models molecular collisions probabilistically. This simulation toolkit uses SPARTA DSMC.

### How It Works

The following parameters are ideal for a high-fidelity ~30 minute simulation on my laptop. For lower fidelity, decrease grid dimensions, number of particles, and time steps, but make sure that constraints are followed for computational accuracy. You can test run the program and it will immediately output what the maximum cell size and timesteps can be:

(example)
CELL SIZE MUST BE < 7.70136701893303 m
TIMESTEP MUST BE < 0.0224217529927786 s

#### Physical Domain
- **3D Cartesian domain:** 2.2m × 2.2m × 2.2m cube (±1.1m in each direction)
- **Boundary conditions:** 
  - X-direction: Outflow (gas escapes at +X boundary, injected at -X)
  - Y,Z-directions: Periodic (wraparound)
- **Grid resolution:** 350 × 200 × 50 cells
- **Cell size:** λ/3 (one-third of mean free path) for accurate collision modeling

#### Atmospheric Modeling
The simulation integrates real atmospheric data using the **NRLMSIS-00 empirical model**:
- **Density (ρ):** Mass density at specified altitude
- **Number density (nrho):** Molecular concentration 
- **Temperature (T):** Atmospheric temperature
- **Bulk velocity (vx):** Atmospheric wind speed

**Altitude range:** 75-100 km (thermosphere/mesopause region below Kármán line; 'very-VLEO')

#### Particle Representation
- **Target particles:** 2,000,000 computational particles
- **Species:** 79% N₂, 21% O₂ (atmospheric composition)
- **Weighting factor:** Each computational particle represents ~10¹⁰-10¹² real molecules
- **Injection:** Continuous inflow of atmospheric gas at domain boundary

#### Collision Physics
**Variable Soft Sphere (VSS) Model:**
- **Probabilistic collisions:** Particles don't physically collide; instead collision probability is calculated based on:
  - Molecular cross-sections (σ ≈ 3.7×10⁻¹⁰ m diameter)
  - Relative velocities
  - Local density
- **Mean free path:** λ = kT/(√2πd²ρR) ≈ meters at high altitude
- **Collision frequency:** Determined by kinetic theory and local gas properties

#### Surface Interactions
**Diffuse Surface Model:**
- **Surface geometry:** 3D satellite model (STL → surface mesh)
- **Temperature coupling:** Stefan-Boltzmann radiation (ε = 0.9, T₀ = 300K)
- **Molecular accommodation:** Gas molecules thermalize with surface temperature
- **Energy transfer:** Tracks kinetic + internal energy flux to surface

#### Time
- **Timestep:** 1×10⁻⁷ seconds (much smaller than collision time)
- **Duration:** 10,000 timesteps (1.0 milliseconds physical time)
- **Diagnostics:** Data output every 100 timesteps (100 frames total)

#### Scope of Results
- **Rarefied gas dynamics** in the thermosphere
- **Molecular velocity distributions** (Maxwell-Boltzmann)
- **Surface heating** from molecular bombardment
- **Altitude-dependent** atmospheric properties
- **Non-equilibrium effects** not captured by continuum models

#### Computational Approach
- **SPARTA:** Open-source DSMC code (particle-based, not grid-based)
- **Memory efficiency:** Parquet format for large (6GB+) datasets
- **Parallelization:** MPI support for multi-core execution
- **Visualization:** Real-time particle tracking and temperature mapping

#### Scientific Applications
This simulation models **satellite atmospheric drag and heating** in the thermosphere, relevant for:
- Low Earth Orbit (LEO) satellite design
- Atmospheric re-entry analysis  
- Hypersonic vehicle aerothermodynamics
- Spacecraft surface temperature prediction

The DSMC method is essential at altitude ranges where the Knudsen number (Kn = λ/L > 0.1) indicates rarefied flow conditions that may violate continuum assumptions used in traditional CFD.

## 1. Installing SPARTA

```bash
sudo apt update
sudo apt install build-essential gfortran mpich
git clone https://github.com/sparta/sparta.git
cd sparta/src
make serial
echo 'export PATH=$PATH:$HOME/sparta/src' >> ~/.bashrc
source ~/.bashrc
sparta -h  # should print help
```

## 2. Setting Up Python Environment

```bash
cd ~/AMPT
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You must run `source .venv/bin/activate` **every time you open a new terminal**  
(unless you're using system-wide Python via `apt`, which doesn't require activation)
I have it set up like this to isolate dependencies and guarantee it will run on any machine with just requirements.txt

To auto-activate in **VS Code**:
- Press `Ctrl+Shift+P`
- Type `Python: Select Interpreter`
- Choose `.venv/bin/python` from the list

## 3. Running Simulations

### Single Altitude Simulation

```bash
# Generate atmospheric data for specific altitude (70-300 km)
python3 tools/load_atm_data.py 150

# Run SPARTA simulation (single core)
sparta < in.ampt

# OR run with multiple cores for faster execution
./run_sparta.sh    # Uses 8 cores by default. Modify this to work with your setup.
```

**Performance:** Multi-core execution is typically 4-6x faster for DSMC simulations. 

### Multi-Altitude Analysis

```bash
# Single core (default)
python3 multi_altitude.py

# Multi-core parallel execution (faster)
python3 multi_altitude.py --cores 8
python3 multi_altitude.py -c 4

# Edit altitude list in multi_altitude.py (default: 75, 80, 85, 90, 95, 100 km)
```

This will:
- Run SPARTA simulations at each altitude
- Save results to `dumps/alt_XXkm/` directories  
- **Automatically convert dumps to Parquet format** for memory-efficient analysis  

**Performance:** Using `--cores 8` is typically 4-6x faster than single core for DSMC simulations.

### Analyze Multi-Altitude Results

After running multi-altitude simulations:

```bash
python3 tools/analyze_multi_altitude.py
```

This will:
- Generate a plot of surface temperature vs altitude for each triangle
- Export data to `multi_altitude_results.csv` spreadsheet
- Save plot as `surface_temps_vs_altitude.png`
- All output files are saved to the `outputs/` folder

## 4. Convert Dump Data for Analysis

After running simulations, convert dump data to Parquet format for memory-efficient analysis:

```bash
# Convert dumps from default directory (dumps/)
python3 tools/load_dumps.py

# Convert dumps from specific directory
python3 tools/load_dumps.py dumps/alt_XXkm/

```

This will:
- Parse raw dump files (part.*.dat, grid.*.dat, surf.*.dat) 
- Save memory-efficient Parquet files (.parquet) in the same directory (uses significantly less RAM than pickle files)
- Can handle large (10GB+) datasets without crashes
- Python analysis scripts automatically use Parquet files for streaming data access

Large SPARTA simulations can generate GB of particle data. The old pickle format was simpler in that it generated a single usable file for each set of dumps, but would cause RAM crashes when loading entire datasets. Parquet enables streaming access (i.e. loading only one timestep at a time) which reduces memory usage.

## 5. Visualization Scripts

All visualization scripts default to `dumps/` folder but can analyze specific altitude data:

```bash
# Default (uses dumps/ folder)
python3 scripts/animate_particles.py
python3 scripts/surface_temp_heatmap.py
python3 scripts/grid_temp_heatmap.py
python3 scripts/velocity_heatmap.py

# Analyze specific altitude results
python3 scripts/animate_particles.py dumps/alt_80km
python3 scripts/surface_temp_heatmap.py dumps/alt_100km
python3 scripts/grid_temp_heatmap.py dumps/alt_75km
python3 scripts/velocity_heatmap.py dumps/alt_95km
```

**Note:** 
- Run `python3 tools/load_dumps.py <folder>` first to convert dump data to Parquet format
- Scripts now use streaming data access, preventing RAM crashes on large datasets. No more catastrophic failure! :D
- All output files (.mp4, .png, .csv) are saved to the `outputs/` folder