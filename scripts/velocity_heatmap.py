import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pickle, os
import re

# I/O
traj_path = os.path.expanduser("~/AMPT/dumps/traj.pkl")
if not os.path.exists(traj_path):
    raise FileNotFoundError(f"Pickle file '{traj_path}' not found. Run load_dumps.py first.")

with open(traj_path, "rb") as f:
    traj = pickle.load(f)

traj.sort(key=lambda tup: tup[0])  # sort by timestep

# box extents
_, _, box0 = traj[0]
xlim = (box0['xlo'], box0['xhi'])
ylim = (box0['ylo'], box0['yhi'])
zlim = (box0['zlo'], box0['zhi'])

# choose slice thickness about z=0
delta_z = 0.1 * (zlim[1] - zlim[0])  # 5% of box height

# grid for the heat-map
nx, ny = 500, 300
x_edges = np.linspace(*xlim, nx + 1)
y_edges = np.linspace(*ylim, ny + 1)

# helpers ----------
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

# precompute robust vmin/vmax ----------
print("Computing robust min/max for color scale...")
all_vals = []
for _, df, _ in traj:
    img = speed_hist(df)
    if np.isfinite(img).any():
        all_vals.extend(img[np.isfinite(img)].flatten())

vmin = np.percentile(all_vals, 5)
vmax = np.percentile(all_vals, 95)
print(f"Robust vmin={vmin:.2f}, vmax={vmax:.2f}")

# figure ----------
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

# animation ----------

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
    step, df, _ = traj[i]
    img = speed_hist(df)
    im.set_data(img)
    title.set_text(f"speed heatmap z=0 cross-section  |  time = {step * tstep:.2e} s")
    return im, title

ani = FuncAnimation(fig, update, frames=len(traj),
                    init_func=init, blit=False, interval=200)

ani.save("velocity_heatmap.mp4", fps=30, dpi=300)
