#!/usr/bin/env python3
"""
Pressure distribution heatmap in z cross-section
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import os
import re
import sys
import argparse
sys.path.append(os.path.expanduser("~/AMPT/tools"))
from load_dumps import load_parquet_timesteps, load_parquet_single

# create outputs directory
os.makedirs('outputs', exist_ok=True)

# parse command line arguments
parser = argparse.ArgumentParser(description='Create grid pressure heatmap')
parser.add_argument('folder', nargs='?', default='dumps', 
                   help='Folder containing dump files (default: dumps)')
args = parser.parse_args()

# get timesteps
folder_path = args.folder
timesteps = load_parquet_timesteps("grid", folder_path)
print(f"Found {len(timesteps)} grid timesteps in {args.folder}")

# load first frame to get grid setup
first_step, first_df, box0 = load_parquet_single("grid", timesteps[0], folder_path)
xlim = (box0["xlo"], box0["xhi"])
ylim = (box0["ylo"], box0["yhi"])
zlim = (box0["zlo"], box0["zhi"])

# choose slice thickness about z=0
slice_frac = 0.05
delta_z = slice_frac * (zlim[1] - zlim[0])

# derive native grid spacing from first frame
xc0 = np.sort(first_df["xc"].unique())
yc0 = np.sort(first_df["yc"].unique())
dx = np.min(np.diff(xc0))
dy = np.min(np.diff(yc0))
x_edges = np.concatenate(([xc0[0] - dx / 2], (xc0[:-1] + xc0[1:]) / 2, [xc0[-1] + dx / 2]))
y_edges = np.concatenate(([yc0[0] - dy / 2], (yc0[:-1] + yc0[1:]) / 2, [yc0[-1] + dy / 2]))
nx, ny = len(xc0), len(yc0)

# pressure column name: c_prop_grid_press[1] is pressure
press_col = [c for c in first_df.columns if "c_prop_grid_press" in c][0]

def pressure_hist(df):
    """Return 2D array of mean pressure for |z| < delta_z, binned on (x,y) using cell centres."""
    xc = df["xc"].to_numpy()
    yc = df["yc"].to_numpy()
    zc = df["zc"].to_numpy()
    mask = np.abs(zc) <= delta_z
    if not mask.any():
        return np.full((ny, nx), np.nan)

    # get pressure
    press = df[press_col].to_numpy()[mask]
    xs, ys = xc[mask], yc[mask]

    sum_p, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges], weights=press)
    cnt_p, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_p = np.where(cnt_p > 0, sum_p / cnt_p, np.nan)
    
    result = mean_p.T  # shape (ny, nx) for imshow
    return result

# precompute vmin/vmax
print("Computing min/max for color scale...")
all_vals = []
for step in timesteps:
    _, df, _ = load_parquet_single("grid", step, folder_path)
    img = pressure_hist(df)
    if np.isfinite(img).any():
        all_vals.extend(img[np.isfinite(img)])
vmin = np.percentile(all_vals, 5)
vmax = np.percentile(all_vals, 95)
print(f"vmin={vmin:.2e} Pa, vmax={vmax:.2e} Pa")

fig, ax = plt.subplots(figsize=(6, 3))
cmap = plt.cm.get_cmap("viridis").copy()
cmap.set_bad('black')

im = ax.imshow(
    np.ma.masked_all((ny, nx)),
    extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
    origin="lower",
    aspect="auto",
    vmin=vmin, vmax=vmax,
    cmap=cmap,
    interpolation="none"
)
cbar = fig.colorbar(im, ax=ax, label="Pressure (Pa)")
title = ax.set_title("")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

def extract_tstep_from_input(path):
    with open(path, "r") as f:
        for line in f:
            m = re.match(r"variable\s+tstep\s+equal\s+([eE\d.+-]+)", line)
            if m:
                return float(m.group(1))
    raise ValueError("tstep not found in input")

tstep = extract_tstep_from_input("in.ampt")

def init():
    im.set_data(np.ma.masked_all((ny, nx)))
    return im,

def update(i):
    step = timesteps[i]
    _, df, _ = load_parquet_single("grid", step, folder_path)
    img = pressure_hist(df)
    img_masked = np.ma.masked_invalid(img)
    im.set_data(img_masked)
    title.set_text(f"pressure heatmap |z| ≤ {slice_frac:.2f}H  |  time = {step * tstep:.2e} s")
    return im, title

ani = FuncAnimation(fig, update, frames=len(timesteps), init_func=init, blit=False, interval=200)
ani.save("outputs/grid_pressure_heatmap.mp4", fps=30, dpi=300)
print("Saved outputs/grid_pressure_heatmap.mp4")
