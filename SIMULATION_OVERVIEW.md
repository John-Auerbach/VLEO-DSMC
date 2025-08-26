# AMPT-DSMC Simulation Overview

## What is DSMC?

**Direct Simulation Monte Carlo (DSMC)** is a computational method for simulating rarefied gas flows where the continuum assumption breaks down. Instead of solving fluid equations, DSMC tracks individual representative particles and models molecular collisions probabilistically. This simulation toolkit uses SPARTA DSMC.

## How This Simulation Works

The following parameters are ideal for a high fidelity ~30 minute simulation on my laptop. For lower fidelity, decrease grid dimensions, number of particles, and time steps, but make sure that constraints are followed for computational accuracy. You can test run the program and it will immediately output what the maximum cell size and timesteps can be:

(example)
CELL SIZE MUST BE < 7.70136701893303 m
TIMESTEP MUST BE < 0.0224217529927786 s

### Physical Domain
- **3D Cartesian domain:** 2.2m × 2.2m × 2.2m cube (±1.1m in each direction)
- **Boundary conditions:** 
  - X-direction: Outflow (gas escapes at +X boundary, injected at -X)
  - Y,Z-directions: Periodic (wraparound)
- **Grid resolution:** 350 × 200 × 50 cells
- **Cell size:** λ/3 (one-third of mean free path) for accurate collision modeling

### Atmospheric Modeling
The simulation integrates real atmospheric data using the **NRLMSIS-00 empirical model**:
- **Density (ρ):** Mass density at specified altitude
- **Number density (nrho):** Molecular concentration 
- **Temperature (T):** Atmospheric temperature
- **Bulk velocity (vx):** Atmospheric wind speed

**Altitude range:** 75-100 km (thermosphere/mesopause region where satellites operate)

### Particle Representation
- **Target particles:** 100,000 computational particles
- **Species:** 79% N₂, 21% O₂ (atmospheric composition)
- **Weighting factor:** Each computational particle represents ~10¹⁰-10¹² real molecules
- **Injection:** Continuous inflow of atmospheric gas at domain boundary

### Collision Physics
**Variable Soft Sphere (VSS) Model:**
- **Probabilistic collisions:** Particles don't physically collide; instead collision probability is calculated based on:
  - Molecular cross-sections (σ ≈ 3.7×10⁻¹⁰ m diameter)
  - Relative velocities
  - Local density
- **Mean free path:** λ = kT/(√2πd²ρR) ≈ meters at high altitude
- **Collision frequency:** Determined by kinetic theory and local gas properties

### Surface Interactions
**Diffuse Surface Model:**
- **Surface geometry:** 3D satellite model (STL → surface mesh)
- **Temperature coupling:** Stefan-Boltzmann radiation (ε = 0.9, T₀ = 300K)
- **Molecular accommodation:** Gas molecules thermalize with surface temperature
- **Energy transfer:** Tracks kinetic + internal energy flux to surface

### Time
- **Timestep:** 1×10⁻⁷ seconds (much smaller than collision time)
- **Duration:** 10,000 timesteps (2.5 milliseconds physical time)
- **Diagnostics:** Data output every 100 timesteps (100 frames total)

### Scope of Results
- **Rarefied gas dynamics** in the thermosphere
- **Molecular velocity distributions** (Maxwell-Boltzmann)
- **Surface heating** from molecular bombardment
- **Altitude-dependent** atmospheric properties
- **Non-equilibrium effects** not captured by continuum models

### Computational Approach
- **SPARTA:** Open-source DSMC code (particle-based, not grid-based)
- **Memory efficiency:** Parquet format for 6GB+ datasets
- **Parallelization:** MPI support for multi-core execution
- **Visualization:** Real-time particle tracking and temperature mapping

### Scientific Applications
This simulation models **satellite atmospheric drag and heating** in the thermosphere, relevant for:
- Low Earth Orbit (LEO) satellite design
- Atmospheric re-entry analysis  
- Hypersonic vehicle aerothermodynamics
- Spacecraft surface temperature prediction

The DSMC method is essential at altitude ranges where the Knudsen number (Kn = λ/L > 0.1) indicates rarefied flow conditions that may violate continuum assumptions used in traditional CFD.
