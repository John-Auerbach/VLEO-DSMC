import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import re
import sys
sys.path.append(os.path.expanduser("~/AMPT/tools"))
from load_dumps import load_parquet_timesteps, load_parquet_single

# I/O
timesteps = load_parquet_timesteps("particle")
print(f"Found {len(timesteps)} particle timesteps")

# Get box extents from first frame
first_step, first_df, box0 = load_parquet_single("particle", timesteps[0])
xlim = (box0['xlo'], box0['xhi'])
ylim = (box0['ylo'], box0['yhi'])
zlim = (box0['zlo'], box0['zhi'])

# choose slice thickness about z=0
slice_frac = 0.05                                      # _% of box height
delta_z = slice_frac * (zlim[1] - zlim[0])

# grid for the heat-map
nx, ny = 500, 300
x_edges = np.linspace(*xlim, nx + 1)
y_edges = np.linspace(*ylim, ny + 1)

def speed_hist(df):
    """Return 2‑D array of mean speed in each (x,y) bin for |z|<delta_z."""
    in_slice = np.abs(df["z"].values) <= delta_z
    if not in_slice.any():
        return np.full((ny, nx), np.nan)
    xs, ys = df["x"].values[in_slice], df["y"].values[in_slice]
    vmag = np.linalg.norm(df[["vx", "vy", "vz"]].values[in_slice], axis=1)

    # accumulate sum of speeds and counts, then divide
    sum_v, _, _ = np.histogram2d(xs, ys, [x_edges, y_edges], weights=vmag)
    cnt_v, _, _ = np.histogram2d(xs, ys, [x_edges, y_edges])
    with np.errstate(invalid='ignore'):
        mean_v = np.divide(sum_v, cnt_v, where=cnt_v > 0)
    return np.flipud(mean_v.T)  # flip y for imshow’s origin='lower'

# precompute vmin/vmax
print("Computing min/max for color scale...")
all_vals = []
for step in timesteps:
    _, df, _ = load_parquet_single("particle", step)
    img = speed_hist(df)
    if np.isfinite(img).any():
        all_vals.extend(img[np.isfinite(img)].flatten())

vmin = 0 #np.percentile(all_vals, 5)
vmax = np.percentile(all_vals, 95)
print(f"vmin={vmin:.2f}, vmax={vmax:.2f}")

fig, ax = plt.subplots(figsize=(6, 3))
im = ax.imshow(
    np.zeros((ny, nx)),
    extent=(*xlim, *ylim),
    origin='lower',
    aspect='auto',
    vmin=vmin, vmax=vmax, cmap='coolwarm'
)
cbar = fig.colorbar(im, ax=ax, label="v (m/s)")
title = ax.set_title("")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

# animation

def extract_tstep_from_input(path):
    with open(path, 'r') as f:
        for line in f:
            m = re.match(r'variable\s+tstep\s+equal\s+([eE\d.+-]+)', line)
            if m:
                return float(m.group(1))
    raise ValueError("tstep not found in input")

tstep = extract_tstep_from_input(os.path.expanduser("~/AMPT/in.ampt"))


def init():
    im.set_data(np.zeros((ny, nx)))
    return im,

def update(i):
    step = timesteps[i]
    _, df, _ = load_parquet_single("particle", step)
    img = speed_hist(df)
    im.set_data(img)
    title.set_text(f"speed heatmap |z| ≤ {slice_frac:.2f}H  |  time = {step * tstep:.2e} s")
    return im, title

ani = FuncAnimation(fig, update, frames=len(timesteps),
                    init_func=init, blit=False, interval=200)

ani.save("velocity_heatmap.mp4", fps=30, dpi=500)
