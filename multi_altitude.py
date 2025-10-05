#!/usr/bin/env python3
"""
Simple multi-altitude SPARTA runner
Usage: 
  python3 multi_altitude.py                    # Single core (default)
  python3 multi_altitude.py --cores 8          # 8 cores with mpirun
  python3 multi_altitude.py -c 4               # 4 cores with mpirun
"""

import os
import subprocess
import shutil
import numpy as np
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run SPARTA simulations at multiple altitudes')
parser.add_argument('--cores', '-c', type=int, default=1, 
                   help='Number of cores to use (default: 1, use mpirun if > 1)')
args = parser.parse_args()

# EDIT THESE ALTITUDES
altitudes = [75, 80, 85, 90, 95, 100]

results = {}

print(f"Running with {args.cores} core(s)")
if args.cores > 1:
    print(f"Using MPI parallel execution")

for alt in altitudes:
    print(f"\n=== Running {alt} km ===")
    
    # Generate atmospheric data
    os.system(f'python3 tools/load_atm_data.py {alt}')
    
    # Run SPARTA (single core or parallel)
    if args.cores == 1:
        os.system('/home/scien/sparta/src/sparta < in.ampt')
    else:
        os.system(f'mpirun -np {args.cores} /home/scien/sparta/src/sparta -in in.ampt')
    
    # Move dumps to altitude folder
    os.makedirs(f'dumps/alt_{alt}km', exist_ok=True)
    for file in os.listdir('dumps'):
        if file.endswith('.dat'):
            shutil.move(f'dumps/{file}', f'dumps/alt_{alt}km/{file}')
    
    # Convert to Parquet for this altitude
    print(f"Converting {alt}km dumps to Parquet...")
    os.system(f'python3 tools/load_dumps.py dumps/alt_{alt}km')
    
    # Calculate drag forces
    print(f"Calculating drag forces for {alt}km...")
    os.system(f'python3 tools/calculate_drag.py dumps/alt_{alt}km')
    
    # Get final surface temps
    surf_files = [f for f in os.listdir(f'dumps/alt_{alt}km') if f.startswith('surf.')]
    final_surf = sorted(surf_files, key=lambda x: int(x.split('.')[1]))[-1]
    
    with open(f'dumps/alt_{alt}km/{final_surf}') as f:
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
    
    # Assuming s_Tsurf is the last column (adjust if needed)
    temps = data[:, -1]  # Last column is usually s_Tsurf
    results[alt] = temps
    
    print(f"Got {len(temps)} surface temps, mean = {np.mean(temps):.1f} K")

print(f"\nDone! Results saved for altitudes: {sorted(results.keys())}")
print("Run 'python3 tools/analyze_multi_altitude.py' to plot and export data.")
