#!/usr/bin/env python3
"""
calculate drag force from particle momentum influx and outflux across control surfaces.

python3 tools/calculate_drag.py <altitude_folder>
"""

import os
import sys
import glob
import numpy as np
import re
from typing import Tuple, List, Dict

def get_species_masses(species_file: str = "species/air.species") -> Dict[int, float]:
    with open(species_file, "r") as f:
        lines = f.readlines()
    
    masses = {}
    species_index = 1
    for line in lines:
        if line.strip() and not line.strip().startswith('#'):
            parts = line.strip().split()
            molecular_weight = float(parts[1])  # g/mol
            mass_kg = molecular_weight * 1.66054e-27  # kg per molecule
            masses[species_index] = mass_kg
            species_index += 1
    
    return masses 

def extract_timestep_from_input(input_file: str = "in.ampt") -> float:
    with open(input_file, "r") as f:
        for line in f:
            m = re.match(r"variable\s+tstep\s+equal\s+([eE\d.+-]+)", line)
            if m:
                return float(m.group(1))

def extract_domain_bounds(input_file: str = "in.ampt") -> Tuple[float, float]:
    xmin, xmax = None, None
    with open(input_file, "r") as f:
        for line in f:
            if "xmin equal" in line:
                parts = line.split()
                xmin = float(parts[3])  # variable xmin equal -1.1
            elif "xmax equal" in line:
                parts = line.split()
                xmax = float(parts[3])  # variable xmax equal 1.1
    
    return xmin, xmax

def load_particle_data(part_file: str) -> Tuple[int, np.ndarray]:
    with open(part_file) as f:
        lines = f.readlines()
    
    step = int(lines[1].strip())
    
    for i, line in enumerate(lines):
        if 'ITEM: ATOMS' in line:
            data_start = i + 1
            break
    
    data = np.loadtxt(lines[data_start:])
    if data.size == 0:
        return step, np.array([]).reshape(0, 8)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    return step, data

def calculate_momentum_flux(particles: np.ndarray, x_plane: float, 
                          species_masses: Dict[int, float], tolerance: float = 1e-10) -> Tuple[float, int]:
    if len(particles) == 0:
        return 0.0, 0
    
    x_pos = particles[:, 2]
    near_plane = np.abs(x_pos - x_plane) < tolerance
    
    if np.any(near_plane):
        crossing_particles = particles[near_plane]
        crossing_types = crossing_particles[:, 1].astype(int)
        crossing_vx = crossing_particles[:, 5]
        
        momentum_flux = 0.0
        for i, ptype in enumerate(crossing_types):
            mass = species_masses[ptype]
            momentum_flux += mass * crossing_vx[i]
        
        return momentum_flux, len(crossing_particles)
    
    return 0.0, 0

def calculate_drag_time_series(dump_folder: str, inlet_x: float, outlet_x: float,
                             plane_tolerance: float = 0.05) -> Dict[str, List]:
    
    dt = extract_timestep_from_input()
    species_masses = get_species_masses()
    
    part_files = glob.glob(os.path.join(dump_folder, 'part.*.dat'))
    part_files.sort(key=lambda x: int(os.path.basename(x).split('.')[1]))
    
    print(f"Found {len(part_files)} files, inlet={inlet_x:.2f}, outlet={outlet_x:.2f}")
    
    # init results
    results = {
        'timesteps': [],
        'times': [],
        'drag_force': [],
        'inlet_momentum': [],
        'outlet_momentum': [],
        'inlet_particles': [],
        'outlet_particles': []
    }
    
    for part_file in part_files:
        step, particles = load_particle_data(part_file)
        time = step * dt
        
        inlet_flux, inlet_count = calculate_momentum_flux(particles, inlet_x, species_masses, plane_tolerance)
        outlet_flux, outlet_count = calculate_momentum_flux(particles, outlet_x, species_masses, plane_tolerance)
        drag = inlet_flux - outlet_flux
        
        results['timesteps'].append(step)
        results['times'].append(time)
        results['drag_force'].append(drag)
        results['inlet_momentum'].append(inlet_flux)
        results['outlet_momentum'].append(outlet_flux)
        results['inlet_particles'].append(inlet_count)
        results['outlet_particles'].append(outlet_count)
        
        if step % 1000 == 0:
            print(f"Step {step}: Drag = {drag:.2e} N")
    
    for key in results:
        results[key] = np.array(results[key])
    
    return results

# ------------------------------------------------------------------------------------------------------------------------

def main():
    if len(sys.argv) >= 2:
        dump_folder = sys.argv[1]
    else:
        dump_folder = "dumps"
    
    print(f"Calculating drag for {dump_folder}")
    
    xmin, xmax = extract_domain_bounds()
    inlet_x = xmin + 0.25 * (xmax - xmin)
    outlet_x = xmin + 0.75 * (xmax - xmin)

    results = calculate_drag_time_series(dump_folder, inlet_x, outlet_x)
    
    mean_drag = np.mean(results['drag_force'])
    final_drag = np.mean(results['drag_force'][-10:])
    
    print(f"Mean drag: {mean_drag:.6e} N")
    print(f"Final drag: {final_drag:.6e} N")
        
    import matplotlib.pyplot as plt
    
    import matplotlib.pyplot as plt
    
    os.makedirs('outputs', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(results['times'], results['drag_force'])
    plt.xlabel('Time (s)')
    plt.ylabel('Drag Force (N)')
    plt.title(f'Drag - {os.path.basename(dump_folder)}')
    plt.grid(True)
    
    plot_file = os.path.join('outputs', f'drag_{os.path.basename(dump_folder)}.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Plot: {plot_file}")

if __name__ == "__main__":
    main()