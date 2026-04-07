#!/usr/bin/env python3
"""
Linearity test: SPARTA runner for drag analysis vs panel length
Tests drag scaling by varying flat panel length from 1m to 10m
Usage: 
  python3 Ethan_linearity_test.py                    # Single core (default)
  python3 Ethan_linearity_test.py --cores 8          # 8 cores with mpirun
  python3 Ethan_linearity_test.py -c 4               # 4 cores with mpirun
"""

import os
import subprocess
import shutil
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run SPARTA simulations at multiple panel lengths')
parser.add_argument('--cores', '-c', type=int, default=1, 
                   help='Number of cores to use (default: 1, use mpirun if > 1)')
args = parser.parse_args()

# Panel lengths 1-10m
lengths = list(range(1, 11))  # 1, 2, 3, ..., 10 meters

print(f"Running with {args.cores} core(s)")
if args.cores > 1:
    print(f"Using MPI parallel execution")

def modify_surf_file(length):
    """Modify linearity_test.surf to have the specified x-length centered at 0"""
    # read surf file
    with open('surf/linearity_test.surf', 'r') as f:
        lines = f.readlines()
    
    # modify the x-coordinates to center panel at x=0
    # points 1-4 should be at x=-length/2
    # points 5-8 should be at x=+length/2
    half_length = length / 2.0
    modified_lines = []
    in_points_section = False
    
    for line in lines:
        stripped = line.strip()
        
        if stripped == 'Points':
            in_points_section = True
            modified_lines.append(line)
        elif stripped == 'Triangles':
            in_points_section = False
            modified_lines.append(line)
        elif in_points_section and len(line.split()) == 4:
            parts = line.split()
            point_id = int(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            if point_id <= 4:
                x = -half_length
            else:
                x = half_length
            modified_lines.append(f"{point_id} {x} {y} {z}\n")
        else:
            modified_lines.append(line)
    
    # write modified surf file
    with open('surf/linearity_test.surf', 'w') as f:
        f.writelines(modified_lines)
    
    print(f"Modified surf file for length = {length}m (centered at x=0: [{-half_length}, {half_length}])")

for length in lengths:
    print(f"\n=== Running {length}m panel ===")
    
    # modify the surf file for this length
    modify_surf_file(length)
    
    # Run SPARTA (single core or parallel)
    if args.cores == 1:
        os.system('/home/scien/sparta/src/sparta < in.ampt')
    else:
        os.system(f'mpirun -np {args.cores} /home/scien/sparta/src/sparta -in in.ampt')
    
    # Move dumps to length folder
    os.makedirs(f'dumps/length_{length}m', exist_ok=True)
    for file in os.listdir('dumps'):
        if file.endswith('.dat'):
            shutil.move(f'dumps/{file}', f'dumps/length_{length}m/{file}')
    
    print(f"Completed {length}m simulation")

print(f"\nDone! Simulations completed for lengths: {lengths}")
print("Run 'python3 scripts/analyze_linearity_drag.py' to analyze results")
