#!/usr/bin/env python3
"""
Analyze multi-altitude SPARTA results: animate density and pressure at origin vs altitude
Usage:
    python3 scripts/analyze_multi_altitude_origin.py
    python3 scripts/analyze_multi_altitude_origin.py --csv   # CSV only to outputs/, no plot
"""

import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse


def load_grid_origin_data(grid_files):
    """
    Load all grid dump files and extract density/pressure at cell closest to origin (0,0,0).
    Returns: timesteps, density_at_origin, pressure_at_origin
    """
    timesteps = []
    nrho_vals = []
    press_vals = []
    
    for gfile in sorted(grid_files):
        # extract timestep from filename like grid_origin.100.dat
        match = re.search(r'grid_origin\.(\d+)\.dat', os.path.basename(gfile))
        if not match:
            continue
        timestep = int(match.group(1))
        
        # load grid data: skip SPARTA dump file header (lines with ITEM:, numbers, box bounds)
        # format is:
        # ITEM: TIMESTEP
        # <timestep>
        # ITEM: NUMBER OF CELLS
        # <ncells>
        # ITEM: BOX BOUNDS ...
        # <xlo> <xhi>
        # <ylo> <yhi>
        # <zlo> <zhi>
        # ITEM: CELLS id xc yc zc f_avg_nrho f_avg_press
        # <data rows>
        with open(gfile, 'r') as f:
            lines = f.readlines()
        
        # find where data starts (after "ITEM: CELLS" line)
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith('ITEM: CELLS'):
                data_start = i + 1
                break
        
        # parse data rows: id xc yc zc f_avg_nrho f_avg_press
        data = []
        for line in lines[data_start:]:
            parts = line.strip().split()
            if len(parts) >= 6:
                data.append([float(x) for x in parts[:6]])
        
        data = np.array(data)
        if len(data) == 0:
            continue
        
        # columns: id(0), xc(1), yc(2), zc(3), f_avg_nrho(4), f_avg_press(5)
        xc = data[:, 1]
        yc = data[:, 2]
        zc = data[:, 3]
        nrho = data[:, 4]
        press = data[:, 5]
        
        # find cell closest to origin
        dist = np.sqrt(xc**2 + yc**2 + zc**2)
        idx = np.argmin(dist)
        
        timesteps.append(timestep)
        nrho_vals.append(nrho[idx])
        press_vals.append(press[idx])
    
    return np.array(timesteps), np.array(nrho_vals), np.array(press_vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', action='store_true', help='If set, save CSV to outputs/ and skip plotting')
    args = ap.parse_args()

    os.makedirs('outputs', exist_ok=True)

    # find altitude directories
    alt_dirs = sorted(glob.glob('dumps/alt_*km'))
    if not alt_dirs:
        print('No altitude directories found under dumps/alt_*km')
        return

    # collect per-alt origin data
    alt_list = []
    per_alt_data = {}
    all_timesteps = set()

    for d in alt_dirs:
        alt = int(d.split('_')[1].replace('km',''))
        alt_list.append(alt)
        
        grid_files = glob.glob(os.path.join(d, 'grid_origin.*.dat'))
        if not grid_files:
            print(f'Warning: no grid_origin files found in {d}')
            continue
        
        t, nrho, press = load_grid_origin_data(grid_files)
        per_alt_data[alt] = {
            't': t, 
            'nrho': nrho,
            'press': press
        }
        all_timesteps.update(t.tolist())

    alts = sorted(alt_list)
    timesteps = sorted(all_timesteps)

    # build 2D arrays: nrho_origin, press_origin [time_index, altitude_index]
    nrho_origin = np.full((len(timesteps), len(alts)), np.nan)
    press_origin = np.full((len(timesteps), len(alts)), np.nan)
    
    for j, alt in enumerate(alts):
        if alt not in per_alt_data:
            continue
        t = per_alt_data[alt]['t']
        nrho = per_alt_data[alt]['nrho']
        press = per_alt_data[alt]['press']
        
        # map into arrays
        for idx, ti in enumerate(t):
            i = timesteps.index(int(ti))
            nrho_origin[i, j] = nrho[idx]
            press_origin[i, j] = press[idx]

    # if CSV requested, save and exit (no plot)
    if args.csv:
        # wide format: first column is timestep, subsequent columns one per altitude (in ascending order)
        header_cols = ['timestep'] + [f'alt_{a}km' for a in alts]
        header = ','.join(header_cols)
        
        # save density at origin
        M_nrho = np.column_stack([np.array(timesteps, dtype=int)] + [nrho_origin[:, j] for j in range(len(alts))])
        out_csv = os.path.join('outputs', 'density_at_origin_vs_altitude.csv')
        np.savetxt(out_csv, M_nrho, delimiter=',', header=header, comments='', fmt='%g')
        print(f'Saved density at origin CSV to {out_csv}')
        
        # save pressure at origin
        M_press = np.column_stack([np.array(timesteps, dtype=int)] + [press_origin[:, j] for j in range(len(alts))])
        out_csv_press = os.path.join('outputs', 'pressure_at_origin_vs_altitude.csv')
        np.savetxt(out_csv_press, M_press, delimiter=',', header=header, comments='', fmt='%g')
        print(f'Saved pressure at origin CSV to {out_csv_press}')
        
        return

    # create two animated plots: density and pressure vs altitude
    
    # density plot
    fig_nrho, ax_nrho = plt.subplots(figsize=(10, 7))
    line_nrho, = ax_nrho.plot([], [], color='blue', label='Density at Origin', linewidth=2, marker='o')
    
    ax_nrho.set_xlim(min(alts)-1, max(alts)+1)
    valid_nrho = nrho_origin[~np.isnan(nrho_origin)]
    if len(valid_nrho) > 0:
        ax_nrho.set_ylim(min(valid_nrho)*0.9, max(valid_nrho)*1.1)
    
    ax_nrho.set_xlabel('Altitude (km)')
    ax_nrho.set_ylabel('Number Density (#/m³)')
    ax_nrho.set_title('')
    ax_nrho.grid(True, alpha=0.3)
    ax_nrho.legend(loc='best')
    
    def init_nrho():
        line_nrho.set_data([], [])
        return [line_nrho]
    
    def update_nrho(i):
        y = nrho_origin[i, :]
        x = alts
        mask = ~np.isnan(y)
        line_nrho.set_data(np.array(x)[mask], y[mask])
        ax_nrho.set_title(f'Density at Origin vs Altitude | timestep {timesteps[i]}')
        return [line_nrho]
    
    ani_nrho = FuncAnimation(fig_nrho, update_nrho, frames=len(timesteps), init_func=init_nrho,
                            interval=200, blit=False)
    
    # pressure plot
    fig_press, ax_press = plt.subplots(figsize=(10, 7))
    line_press, = ax_press.plot([], [], color='red', label='Pressure at Origin', linewidth=2, marker='s')
    
    ax_press.set_xlim(min(alts)-1, max(alts)+1)
    valid_press = press_origin[~np.isnan(press_origin)]
    if len(valid_press) > 0:
        ax_press.set_ylim(min(valid_press)*0.9, max(valid_press)*1.1)
    
    ax_press.set_xlabel('Altitude (km)')
    ax_press.set_ylabel('Pressure (Pa)')
    ax_press.set_title('')
    ax_press.grid(True, alpha=0.3)
    ax_press.legend(loc='best')
    
    def init_press():
        line_press.set_data([], [])
        return [line_press]
    
    def update_press(i):
        y = press_origin[i, :]
        x = alts
        mask = ~np.isnan(y)
        line_press.set_data(np.array(x)[mask], y[mask])
        ax_press.set_title(f'Pressure at Origin vs Altitude | timestep {timesteps[i]}')
        return [line_press]
    
    ani_press = FuncAnimation(fig_press, update_press, frames=len(timesteps), init_func=init_press,
                            interval=200, blit=False)

    # save both animations
    ani_nrho.save('outputs/density_at_origin_vs_altitude.mp4', fps=10, dpi=150)
    print('Saved outputs/density_at_origin_vs_altitude.mp4')
    
    ani_press.save('outputs/pressure_at_origin_vs_altitude.mp4', fps=10, dpi=150)
    print('Saved outputs/pressure_at_origin_vs_altitude.mp4')


if __name__ == '__main__':
    main()
