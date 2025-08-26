import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os, re
import sys
import argparse
import copy as _copy
import warnings
sys.path.append(os.path.expanduser("~/AMPT/tools"))
from load_dumps import load_parquet_timesteps, load_parquet_single
from scipy.ndimage import gaussian_filter

# create outputs directory
os.makedirs('outputs', exist_ok=True)

# parse command line arguments
parser = argparse.ArgumentParser(description='Create grid temperature heatmap animation')
parser.add_argument('folder', nargs='?', default='dumps', 
                   help='Folder containing dump files (default: dumps)')
args = parser.parse_args()

# I/O - Get timesteps without loading data from specified folder
folder_path = os.path.expanduser(f"~/AMPT/{args.folder}")
timesteps = load_parquet_timesteps("grid", folder_path)
print(f"Found {len(timesteps)} grid timesteps in {args.folder}")

# load just the first frame to get grid setup
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

# temperature column name
temp_col = [c for c in first_df.columns if c.startswith("c_compute_Tgrid")][0]

def temp_hist(df):
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
    # produce NaN where count is zero so masked_invalid() will mask these cells
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_t = np.where(cnt_t > 0, sum_t / cnt_t, np.nan)
    
    result = mean_t.T  # shape (ny, nx) for imshow
    
    return result

# precompute vmin/vmax
print("Computing min/max for color scale...")
all_vals = []
# suppress runtime warnings during histogram division (empty bins)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    for step in timesteps:
        _, df, _ = load_parquet_single("grid", step, folder_path)
        img = temp_hist(df)
        if np.isfinite(img).any():
            all_vals.extend(img[np.isfinite(img)])
vmin = 0 #np.percentile(all_vals, 5)
vmax = np.percentile(all_vals, 95)
print(f"vmin={vmin:.2f}, vmax={vmax:.2f}")

fig, ax = plt.subplots(figsize=(6, 3))
# use a cmap with black for masked/invalid cells
_cmap = plt.cm.get_cmap("inferno")
try:
    cmap = _cmap.copy()
except Exception:
    cmap = _copy.deepcopy(_cmap)
cmap.set_bad('black')
# initialize with a fully-masked array so empty cells show black
im = ax.imshow(
    np.ma.masked_all((ny, nx)),
    extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
    origin="lower",
    aspect="auto",
    vmin=vmin, vmax=vmax,
    cmap=cmap,
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
    im.set_data(np.ma.masked_all((ny, nx)))
    return im,

def update(i):
    step = timesteps[i]
    _, df, _ = load_parquet_single("grid", step, folder_path)
    img = temp_hist(df)
    # mask invalid/empty bins so they render as black
    img_masked = np.ma.masked_invalid(img)
    im.set_data(img_masked)
    title.set_text(f"temperature heatmap |z| ≤ {slice_frac:.2f}H  |  time = {step * tstep:.2e} s")
    return im, title

ani = FuncAnimation(fig, update, frames=len(timesteps), init_func=init, blit=False, interval=200)
ani.save("outputs/grid_temp_heatmap.mp4", fps=30, dpi=300)
