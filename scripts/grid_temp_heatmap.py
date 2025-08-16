import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os, re
import sys
sys.path.append(os.path.expanduser("~/AMPT/tools"))
from load_dumps import load_parquet_timesteps, load_parquet_single
from scipy.ndimage import gaussian_filter

# I/O - Get timesteps without loading data
timesteps = load_parquet_timesteps("grid")
print(f"Found {len(timesteps)} grid timesteps")

# Load just the first frame to get grid setup
first_step, first_df, box0 = load_parquet_single("grid", timesteps[0])
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

# temperature column name
temp_col = [c for c in first_df.columns if c.startswith("c_compute_Tgrid")][0]

def temp_hist(df, smooth=True):
    """Return 2D array of mean Tgrid value for |z| < delta_z, binned on (x,y) using cell centres."""
    xc = df["xc"].to_numpy()
    yc = df["yc"].to_numpy()
    zc = df["zc"].to_numpy()
    mask = np.abs(zc) <= delta_z
    if not mask.any():
        return np.full((ny, nx), np.nan)

    temps = df[temp_col].to_numpy()[mask]
    xs, ys = xc[mask], yc[mask]

    sum_t, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges], weights=temps)
    cnt_t, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
    with np.errstate(invalid="ignore"):
        mean_t = np.divide(sum_t, cnt_t, where=cnt_t > 0)
    
    result = mean_t.T  # shape (ny, nx) for imshow
    
    # Apply light Gaussian smoothing to reduce graininess
    if smooth:
        result = gaussian_filter(result, sigma=0.8, mode='constant', cval=np.nan)
    
    return result

# precompute vmin/vmax
print("Computing min/max for color scale...")
all_vals = []
for step in timesteps:
    _, df, _ = load_parquet_single("grid", step)
    img = temp_hist(df)
    if np.isfinite(img).any():
        all_vals.extend(img[np.isfinite(img)])
vmin = 0 #np.percentile(all_vals, 5)
vmax = np.percentile(all_vals, 95)
print(f"vmin={vmin:.2f}, vmax={vmax:.2f}")

fig, ax = plt.subplots(figsize=(6, 3))
im = ax.imshow(
    np.zeros((ny, nx)),
    extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
    origin="lower",
    aspect="auto",
    vmin=vmin, vmax=vmax,
    cmap="inferno",
    interpolation="bilinear"  # or "bicubic" for even smoother
)
cbar = fig.colorbar(im, ax=ax, label="T (K)")
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

tstep = extract_tstep_from_input(os.path.expanduser("~/AMPT/in.ampt"))

def init():
    im.set_data(np.zeros((ny, nx)))
    return im,

def update(i):
    step = timesteps[i]
    _, df, _ = load_parquet_single("grid", step)
    img = temp_hist(df)
    im.set_data(img)
    title.set_text(f"temperature heatmap |z| ≤ {slice_frac:.2f}H  |  time = {step * tstep:.2e} s")
    return im, title

ani = FuncAnimation(fig, update, frames=len(timesteps), init_func=init, blit=False, interval=200)
ani.save("grid_temp_heatmap.mp4", fps=30, dpi=300)
