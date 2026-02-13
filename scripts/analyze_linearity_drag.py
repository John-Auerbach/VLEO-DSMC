#!/usr/bin/env python3
"""
Analyze linearity test SPARTA results: animate direct drag vs panel length
Usage:
    python3 scripts/analyze_linearity_drag.py
    python3 scripts/analyze_linearity_drag.py --csv   # CSV only to outputs/, no plot
"""

import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse


def load_direct_drag(drag_file):
    # reads files like:
    # # Time-averaged data for fix dragout
    # # TimeStep c_drag c_drag_xnorm c_drag_walls
    # 0 0 0 0
    # 100 6.0 4.5 1.5
    data = np.loadtxt(drag_file)
    if data.ndim == 1:
        data = data.reshape((1, -1))
    t = data[:, 0].astype(int)
    total_drag = data[:, 1]
    ram_drag = data[:, 2] if data.shape[1] > 2 else None
    skin_friction = data[:, 3] if data.shape[1] > 3 else None
    return t, total_drag, ram_drag, skin_friction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', action='store_true', help='If set, save CSV to outputs/ and skip plotting')
    args = ap.parse_args()

    os.makedirs('outputs', exist_ok=True)

    # find length directories
    length_dirs = sorted(glob.glob('dumps/length_*m'))
    if not length_dirs:
        print('No length directories found under dumps/length_*m')
        return

    # collect per-length drag series
    length_list = []
    per_length_data = {}
    all_timesteps = set()

    for d in length_dirs:
        length = int(d.split('_')[1].replace('m',''))
        length_list.append(length)
        drag_path = os.path.join(d, 'direct_drag.dat')
        t, total_drag, ram_drag, skin_friction = load_direct_drag(drag_path)
        per_length_data[length] = {
            't': t, 
            'total_drag': total_drag,
            'ram_drag': ram_drag,
            'skin_friction': skin_friction
        }
        all_timesteps.update(t.tolist())

    lengths = sorted(length_list)
    timesteps = sorted(all_timesteps)

    # build 2D arrays: D_total, D_ram, D_skin [time_index, length_index]
    D_total = np.full((len(timesteps), len(lengths)), np.nan)
    D_ram = np.full((len(timesteps), len(lengths)), np.nan)
    D_skin = np.full((len(timesteps), len(lengths)), np.nan)
    
    has_components = False
    for j, length in enumerate(lengths):
        t = per_length_data[length]['t']
        total_drag = per_length_data[length]['total_drag']
        ram_drag = per_length_data[length]['ram_drag']
        skin_friction = per_length_data[length]['skin_friction']
        
        if ram_drag is not None and skin_friction is not None:
            has_components = True
        
        # map into arrays
        for idx, ti in enumerate(t):
            i = timesteps.index(int(ti))
            D_total[i, j] = total_drag[idx]
            if ram_drag is not None:
                D_ram[i, j] = ram_drag[idx]
            if skin_friction is not None:
                D_skin[i, j] = skin_friction[idx]

    # if CSV requested, save and exit (no plot)
    if args.csv:
        # wide format: first column is timestep, subsequent columns one per length (in ascending order)
        # save total drag
        header_cols = ['timestep'] + [f'length_{L}m' for L in lengths]
        header = ','.join(header_cols)
        M_total = np.column_stack([np.array(timesteps, dtype=int)] + [D_total[:, j] for j in range(len(lengths))])
        out_csv = os.path.join('outputs', 'drag_vs_length.csv')
        np.savetxt(out_csv, M_total, delimiter=',', header=header, comments='', fmt='%g')
        print(f'Saved total drag CSV to {out_csv}')
        
        # save ram drag if available
        if has_components:
            M_ram = np.column_stack([np.array(timesteps, dtype=int)] + [D_ram[:, j] for j in range(len(lengths))])
            out_csv_ram = os.path.join('outputs', 'ram_drag_vs_length.csv')
            np.savetxt(out_csv_ram, M_ram, delimiter=',', header=header, comments='', fmt='%g')
            print(f'Saved ram drag CSV to {out_csv_ram}')
            
            M_skin = np.column_stack([np.array(timesteps, dtype=int)] + [D_skin[:, j] for j in range(len(lengths))])
            out_csv_skin = os.path.join('outputs', 'skin_friction_vs_length.csv')
            np.savetxt(out_csv_skin, M_skin, delimiter=',', header=header, comments='', fmt='%g')
            print(f'Saved skin friction CSV to {out_csv_skin}')
        
        return

    # create two animated plots (linear and log)
    data_arrays = [D_total, D_ram, D_skin]
    labels = ['Total Drag', 'Ram Drag (x-normals)', 'Skin Friction (walls)']
    colors = ['blue', 'red', 'green']
    
    # linear
    fig_lin, ax_lin = plt.subplots(figsize=(10, 7))
    lines_lin = []
    
    for D, label, color in zip(data_arrays, labels, colors):
        line, = ax_lin.plot([], [], color=color, label=label, linewidth=2)
        lines_lin.append(line)
    
    ax_lin.set_xlim(min(lengths)-0.5, max(lengths)+0.5)
    all_valid = []
    for D in data_arrays:
        valid = D[~np.isnan(D)]
        if len(valid) > 0:
            all_valid.extend(valid)
    if all_valid:
        ax_lin.set_ylim(min(all_valid)*0.9, max(all_valid)*1.1)
    
    ax_lin.set_xlabel('Panel Length (m)')
    ax_lin.set_ylabel('Drag (N)')
    ax_lin.set_title('')
    ax_lin.grid(True, alpha=0.3)
    ax_lin.legend(loc='best')
    
    def init_lin():
        for line in lines_lin:
            line.set_data([], [])
        return lines_lin
    
    def update_lin(i):
        for line, D in zip(lines_lin, data_arrays):
            y = D[i, :]
            x = lengths
            mask = ~np.isnan(y)
            line.set_data(np.array(x)[mask], y[mask])
        ax_lin.set_title(f'Drag Components vs Panel Length | timestep {timesteps[i]}')
        return lines_lin
    
    ani_lin = FuncAnimation(fig_lin, update_lin, frames=len(timesteps), init_func=init_lin,
                            interval=200, blit=False)
    
    # log
    fig_log, ax_log = plt.subplots(figsize=(10, 7))
    lines_log = []
    
    for D, label, color in zip(data_arrays, labels, colors):
        line, = ax_log.plot([], [], color=color, label=label, linewidth=2)
        lines_log.append(line)
    
    ax_log.set_xlim(min(lengths)-0.5, max(lengths)+0.5)
    all_valid_pos = []
    for D in data_arrays:
        valid = D[~np.isnan(D) & (D > 0)]
        if len(valid) > 0:
            all_valid_pos.extend(valid)
    if all_valid_pos:
        ax_log.set_ylim(min(all_valid_pos)*0.5, max(all_valid_pos)*2.0)
    else:
        raise ValueError("No positive drag values found; cannot create log-scale plot")
    
    ax_log.set_xlabel('Panel Length (m)')
    ax_log.set_ylabel('Drag (N)')
    ax_log.set_yscale('log')
    ax_log.set_title('')
    ax_log.grid(True, alpha=0.3, which='both')
    ax_log.legend(loc='best')
    
    def init_log():
        for line in lines_log:
            line.set_data([], [])
        return lines_log
    
    def update_log(i):
        for line, D in zip(lines_log, data_arrays):
            y = D[i, :]
            x = lengths
            mask = ~np.isnan(y) & (y > 0)  # only positive values for log scale
            line.set_data(np.array(x)[mask], y[mask])
        ax_log.set_title(f'Drag Components vs Panel Length (Log Scale) | timestep {timesteps[i]}')
        return lines_log
    
    ani_log = FuncAnimation(fig_log, update_log, frames=len(timesteps), init_func=init_log,
                            interval=200, blit=False)

    # save both animations
    ani_lin.save('outputs/linearity_drag_evolution.mp4', fps=10, dpi=150)
    print('Saved outputs/linearity_drag_evolution.mp4')
    
    ani_log.save('outputs/linearity_drag_evolution_log.mp4', fps=10, dpi=150)
    print('Saved outputs/linearity_drag_evolution_log.mp4')


if __name__ == '__main__':
    main()

