#!/usr/bin/env python3
"""
Analyze multi-altitude SPARTA results: plot and export spreadsheet
Usage: python3 tools/analyze_multi_altitude.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    # Find all altitude directories
    alt_dirs = glob.glob('dumps/alt_*km')
    if not alt_dirs:
        print("No altitude directories found. Run multi_altitude.py first.")
        return
    
    results = {}
    
    for alt_dir in alt_dirs:
        # Extract altitude from directory name
        alt = int(alt_dir.split('_')[1].replace('km', ''))
        
        # Find final surface file
        surf_files = glob.glob(f'{alt_dir}/surf.*.dat')
        if not surf_files:
            print(f"No surface files in {alt_dir}")
            continue
            
        final_surf = sorted(surf_files, key=lambda x: int(x.split('.')[-2]))[-1]
        
        # Load surface data
        with open(final_surf) as f:
            lines = f.readlines()
        
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
        results[alt] = temps
        print(f"{alt} km: {len(temps)} triangles, mean = {np.mean(temps):.1f} K")
    
    if not results:
        print("No valid results found.")
        return
    
    # Create DataFrame for spreadsheet
    alts = sorted(results.keys())
    max_triangles = max(len(results[alt]) for alt in alts)
    
    # Build spreadsheet data
    spreadsheet_data = {'Altitude_km': alts}
    for i in range(max_triangles):
        triangle_temps = []
        for alt in alts:
            if i < len(results[alt]):
                triangle_temps.append(results[alt][i])
            else:
                triangle_temps.append(np.nan)
        spreadsheet_data[f'Triangle_{i+1}_K'] = triangle_temps
    
    # Add mean temperature column
    spreadsheet_data['Mean_Temperature_K'] = [np.mean(results[alt]) for alt in alts]
    
    df = pd.DataFrame(spreadsheet_data)
    df.to_csv('outputs/multi_altitude_results.csv', index=False)
    print(f"Exported data to outputs/multi_altitude_results.csv")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for i in range(max_triangles):
        triangle_temps = [results[alt][i] if i < len(results[alt]) else np.nan for alt in alts]
        plt.plot(alts, triangle_temps, 'o-', label=f'Triangle {i+1}')
    
    plt.xlabel('Altitude (km)')
    plt.ylabel('Surface Temperature (K)')
    plt.title('Surface Temperature vs Altitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/surface_temps_vs_altitude.png', dpi=150)
    plt.show()
    
    print("Plot saved as outputs/surface_temps_vs_altitude.png")

if __name__ == "__main__":
    main()
