#!/usr/bin/env python3
"""
Analyze multi-altitude SPARTA results: animate direct drag vs altitude
Usage: python3 tools/analyze_multi_altitude_drag.py
"""

import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def load_direct_drag(drag_file):
    # reads files like:
    # # Time-averaged data for fix dragout
    # # TimeStep c_drag
    # 0 0
    # 100 6.0
    data = np.loadtxt(drag_file)
    if data.ndim == 1:
        data = data.reshape((1, -1))
    t = data[:, 0].astype(int)
    drag = data[:, 1]
    return t, drag


def main():
    os.makedirs('outputs', exist_ok=True)

    # Find altitude directories
    alt_dirs = sorted(glob.glob('dumps/alt_*km'))
    if not alt_dirs:
        print('No altitude directories found under dumps/alt_*km')
        return

    # collect per-alt drag series
    alt_list = []
    per_alt_data = {}
    all_timesteps = set()

    for d in alt_dirs:
        alt = int(d.split('_')[1].replace('km',''))
        alt_list.append(alt)
        drag_path = os.path.join(d, 'direct_drag.dat')
        t, drag = load_direct_drag(drag_path)
        per_alt_data[alt] = {'t': t, 'drag': drag}
        all_timesteps.update(t.tolist())

    alts = sorted(alt_list)
    timesteps = sorted(all_timesteps)

    # Build a 2D array D[time_index, altitude_index]
    D = np.full((len(timesteps), len(alts)), np.nan)
    for j, alt in enumerate(alts):
        t = per_alt_data[alt]['t']
        drag = per_alt_data[alt]['drag']
        # map into D
        for ti, val in zip(t, drag):
            i = timesteps.index(int(ti))
            D[i, j] = val

    # animation: one moving line of drag vs altitude
    fig, ax = plt.subplots(figsize=(8,6))
    line, = ax.plot([], [], '-o')
    ax.set_xlim(min(alts)-1, max(alts)+1)
    # y limits (drag)
    valid = D[~np.isnan(D)]
    ax.set_ylim(np.nanmin(valid)*0.9, np.nanmax(valid)*1.1)
    ax.set_xlabel('Altitude (km)')
    ax.set_ylabel('Drag (N)')
    title = ax.set_title('')
    ax.grid(True)

    def init():
        line.set_data([], [])
        return line, title

    def update(i):
        y = D[i, :]
        x = alts
        # mask NaNs
        mask = ~np.isnan(y)
        line.set_data(np.array(x)[mask], y[mask])
        title.set_text(f'Drag vs Altitude | timestep {timesteps[i]}')
        return line, title

    ani = FuncAnimation(fig, update, frames=len(timesteps), init_func=init,
                        interval=200, blit=False)

    ani.save('outputs/multi_altitude_drag_evolution.mp4', fps=10, dpi=150)
    print('Saved outputs/multi_altitude_drag_evolution.mp4')


if __name__ == '__main__':
    main()
