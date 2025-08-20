#!/usr/bin/env python3
"""
Analyze multi-altitude SPARTA results: create time animation
Usage: python3 tools/analyze_multi_altitude.py
"""

import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# import pandas as pd  # Uncomment for CSV export

def extract_tstep_from_input(path):
    """Extract timestep from SPARTA input file"""
    with open(path, "r") as f:
        for line in f:
            m = re.match(r"variable\s+tstep\s+equal\s+([eE\d.+-]+)", line)
            if m:
                return float(m.group(1))
    raise ValueError("tstep not found in input")

def load_surface_data(surf_file):
    """Load surface temperature data from a surf file"""
    with open(surf_file) as f:
        lines = f.readlines()
    
    # Get timestep from header
    step = int(lines[1].strip())
    
    # Find data section
    for i, line in enumerate(lines):
        if 'ITEM: SURFS' in line:
            data_start = i + 1
            break
    
    # Load surface temperatures
    data = np.loadtxt(lines[data_start:])
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    # Last column is s_Tsurf
    temps = data[:, -1]
    return step, temps

def main():
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    # Get timestep from input file
    try:
        tstep = extract_tstep_from_input("in.ampt")
    except:
        print("Warning: Could not extract timestep from in.ampt, using default")
        tstep = 1e-6
    
    # Find all altitude directories
    alt_dirs = glob.glob('dumps/alt_*km')
    if not alt_dirs:
        print("No altitude directories found. Run multi_altitude.py first.")
        return
    
    # Collect all timestep data across altitudes
    all_data = {}  # {altitude: {timestep: temperatures}}
    all_timesteps = set()
    
    for alt_dir in alt_dirs:
        # Extract altitude from directory name
        alt = int(alt_dir.split('_')[1].replace('km', ''))
        
        # Find all surface files
        surf_files = glob.glob(f'{alt_dir}/surf.*.dat')
        if not surf_files:
            print(f"No surface files in {alt_dir}")
            continue
        
        # Sort by timestep number
        surf_files = sorted(surf_files, key=lambda x: int(x.split('.')[-2]))
        
        all_data[alt] = {}
        for surf_file in surf_files:
            try:
                step, temps = load_surface_data(surf_file)
                all_data[alt][step] = temps
                all_timesteps.add(step)
            except Exception as e:
                print(f"Error loading {surf_file}: {e}")
                continue
        
        print(f"{alt} km: {len(all_data[alt])} timesteps")
    
    if not all_data:
        print("No valid results found.")
        return
    
    # Get sorted lists
    alts = sorted(all_data.keys())
    timesteps = sorted(all_timesteps)
    
    # COMMENTED OUT: Static analysis (final timestep only)
    # Uncomment the section below to generate CSV and PNG for final timestep
    """
    print("Creating static analysis for final timestep...")
    final_results = {}
    for alt in alts:
        if timesteps and timesteps[-1] in all_data[alt]:
            final_results[alt] = all_data[alt][timesteps[-1]]
    
    if final_results:
        # Create DataFrame for spreadsheet
        max_triangles = max(len(final_results[alt]) for alt in alts)
        
        # Build spreadsheet data
        spreadsheet_data = {'Altitude_km': alts}
        for i in range(max_triangles):
            triangle_temps = []
            for alt in alts:
                if i < len(final_results[alt]):
                    triangle_temps.append(final_results[alt][i])
                else:
                    triangle_temps.append(np.nan)
            spreadsheet_data[f'Triangle_{i+1}_K'] = triangle_temps
        
        # Add mean temperature column
        spreadsheet_data['Mean_Temperature_K'] = [np.mean(final_results[alt]) for alt in alts]
        
        df = pd.DataFrame(spreadsheet_data)
        df.to_csv('outputs/multi_altitude_results.csv', index=False)
        print(f"Exported final timestep data to outputs/multi_altitude_results.csv")
        
        # Plot static results
        plt.figure(figsize=(10, 6))
        for i in range(max_triangles):
            triangle_temps = [final_results[alt][i] if i < len(final_results[alt]) else np.nan for alt in alts]
            plt.plot(alts, triangle_temps, 'o-', label=f'Triangle {i+1}')
        
        plt.xlabel('Altitude (km)')
        plt.ylabel('Surface Temperature (K)')
        plt.title('Surface Temperature vs Altitude (Final Timestep)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('outputs/surface_temps_vs_altitude.png', dpi=150)
        print("Static plot saved as outputs/surface_temps_vs_altitude.png")
    """
    
    # Create animated plot showing temperature evolution over time
    if len(timesteps) > 1:
        print(f"Creating animation..")
        
        # Determine number of triangles (use first available data)
        num_triangles = 0
        for alt in alts:
            for step in timesteps:
                if step in all_data[alt]:
                    num_triangles = len(all_data[alt][step])
                    break
            if num_triangles > 0:
                break
        
        # Set up the animated plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Initialize empty lines for each triangle
        lines = []
        for i in range(num_triangles):
            line, = ax.plot([], [], 'o-', label=f'Triangle {i+1}')
            lines.append(line)
        
        ax.set_xlabel('Altitude (km)')
        ax.set_ylabel('Surface Temperature (K)')
        ax.legend()
        ax.grid(True)
        
        # Calculate global temperature range for consistent y-axis
        all_temps = []
        for alt in alts:
            for step_data in all_data[alt].values():
                all_temps.extend(step_data)
        
        if all_temps:
            temp_min, temp_max = np.min(all_temps), np.max(all_temps)
            temp_range = temp_max - temp_min
            ax.set_ylim(temp_min - 0.1 * temp_range, temp_max + 0.1 * temp_range)
        
        ax.set_xlim(min(alts) - 1, max(alts) + 1)
        
        title = ax.set_title("")
        
        def init():
            for line in lines:
                line.set_data([], [])
            return lines + [title]
        
        def update(frame_idx):
            step = timesteps[frame_idx]
            
            # Update each triangle line
            for i, line in enumerate(lines):
                alt_list = []
                temp_list = []
                
                for alt in alts:
                    if step in all_data[alt] and i < len(all_data[alt][step]):
                        alt_list.append(alt)
                        temp_list.append(all_data[alt][step][i])
                
                line.set_data(alt_list, temp_list)
            
            # Update title with current time
            title.set_text(f"Surface Temperature vs Altitude | time = {step * tstep:.2e} s")
            
            return lines + [title]
        
        # Create animation
        ani = FuncAnimation(fig, update, frames=len(timesteps), init_func=init, 
                          blit=False, interval=200, repeat=True)
        
        # Save animation at 30fps
        ani.save("outputs/multi_altitude_temp_evolution.mp4", fps=30, dpi=150)
        print("Animation saved as outputs/multi_altitude_temp_evolution.mp4")
        
    else:
        print("Only one timestep found, skipping animation.")

if __name__ == "__main__":
    main()
