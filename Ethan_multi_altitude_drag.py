#!/usr/bin/env python3
"""
Multi-altitude SPARTA runner for drag analysis
Skips parquet conversion since plot_drag.py only needs .dat files
Usage: 
  python3 Ethan_multi_altitude_drag.py                    # Single core (default)
  python3 Ethan_multi_altitude_drag.py --cores 8          # 8 cores with mpirun
  python3 Ethan_multi_altitude_drag.py -c 4               # 4 cores with mpirun
"""

import os
import subprocess
import shutil
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run SPARTA simulations at multiple altitudes')
parser.add_argument('--cores', '-c', type=int, default=1, 
                   help='Number of cores to use (default: 1, use mpirun if > 1)')
args = parser.parse_args()

# EDIT THESE ALTITUDES
altitudes = list(range(285, 305, 5))  # 70 to 300 km in 5 km intervals

print(f"Running with {args.cores} core(s)")
if args.cores > 1:
    print(f"Using MPI parallel execution")

for alt in altitudes:
    print(f"\n=== Running {alt} km ===")
    
    # Generate atmospheric data
    os.system(f'python3 tools/load_Ethan_NRLMSISE00.py {alt}')
    
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
    
    print(f"Completed {alt}km simulation")

print(f"\nDone! Simulations completed for altitudes: {altitudes}")
print("Run 'python3 scripts/plot_drag.py dumps/alt_<altitude>km/' to plot drag for each altitude")
