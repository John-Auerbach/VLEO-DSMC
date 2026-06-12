import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import re
import argparse
import copy as _copy
import warnings
import sys
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(_REPO_ROOT, 'tools'))
from load_dumps import load_parquet_timesteps, load_parquet_single
from anim_utils import compute_payloads_parallel, save_animation

# Create outputs directory
os.makedirs('outputs', exist_ok=True)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Create velocity heatmap animation')
parser.add_argument('folder', nargs='?', default='dumps', 
                   help='Folder containing dump files (default: dumps)')
parser.add_argument('-j', '--jobs', type=int, default=1,
                   help='Parallel worker processes for frame precompute '
                        '(default: 1 = serial). Use the node core count on a '
                        'cluster, e.g. -j 8 (memory scales with jobs).')
args = parser.parse_args()

# I/O
folder_path = os.path.join(_REPO_ROOT, args.folder)
timesteps = load_parquet_timesteps("particle", folder_path)
print(f"Found {len(timesteps)} particle timesteps in {args.folder}")

# Get box extents from first frame
first_step, first_df, box0 = load_parquet_single("particle", timesteps[0], folder_path)
xlim = (box0['xlo'], box0['xhi'])
ylim = (box0['ylo'], box0['yhi'])
zlim = (box0['zlo'], box0['zhi'])

# choose slice thickness about z=0
slice_frac = 0.05                                      # _% of box height
delta_z = slice_frac * (zlim[1] - zlim[0])

# grid for the heat-map
nx, ny = 1500, 1000
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
    # ensure empty bins are NaN (suppress divide warnings)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_v = np.where(cnt_v > 0, sum_v / cnt_v, np.nan)
    return np.flipud(mean_v.T)  # flip y for imshow's origin='lower'

def compute_frame(i):
    """Load one particle frame and return (step, speed image). Top-level so it is
    picklable for parallel workers; relies on inherited module globals."""
    import warnings as _w
    step = timesteps[i]
    with _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        # Read only the columns we need and downcast to float32 to roughly
        # halve per-frame memory (particle frames are multi-GB; the old all-
        # column float64 load OOM'd with several workers).
        _, df, _ = load_parquet_single("particle", step, folder_path,
                                       columns=["x", "y", "z", "vx", "vy", "vz"])
        df = df.astype("float32", copy=False)
        img = speed_hist(df)
    return step, img

# Precompute every frame's image (the expensive load+histogram), optionally in
# parallel, with live progress. This replaces the old serial vmin/vmax pre-pass:
# the colour scale is derived from the cached images below, so each file is read
# only once instead of twice.
print(f"Precomputing {len(timesteps)} frames with {args.jobs} worker(s)...")
frame_imgs = compute_payloads_parallel(range(len(timesteps)), compute_frame,
                                       jobs=args.jobs, label="particle frame")

# colour scale from the already-computed images (no extra file reads)
vmin = np.inf
vmax = -np.inf
for _step, img in frame_imgs.values():
    finite = img[np.isfinite(img)]
    if finite.size:
        vmin = min(vmin, float(finite.min()))
        vmax = max(vmax, float(finite.max()))
if not np.isfinite(vmin):
    vmin, vmax = 0.0, 1.0
print(f"vmin={vmin:.2f}, vmax={vmax:.2f}")

fig, ax = plt.subplots(figsize=(6, 3))
# use a cmap with black for masked/invalid cells
# use the reversed colormap so colors are inverted
_cmap = plt.cm.get_cmap('coolwarm_r')
try:
    cmap = _cmap.copy()
except Exception:
    cmap = _copy.deepcopy(_cmap)
cmap.set_bad('black')
# initialize with a fully-masked array so empty cells show black
im = ax.imshow(
    np.ma.masked_all((ny, nx)),
    extent=(*xlim, *ylim),
    origin='lower',
    aspect='auto',
    vmin=vmin, vmax=vmax, cmap=cmap
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

tstep = extract_tstep_from_input(os.path.join(_REPO_ROOT, 'in.runfile'))


def init():
    im.set_data(np.ma.masked_all((ny, nx)))
    return im,

def update(i):
    step, img = frame_imgs[i]
    img_masked = np.ma.masked_invalid(img)
    im.set_data(img_masked)
    title.set_text(f"speed heatmap |z| ≤ {slice_frac:.2f}H  |  time = {step * tstep:.2e} s")
    return im, title

ani = FuncAnimation(fig, update, frames=len(timesteps),
                    init_func=init, blit=False, interval=200)

save_animation(ani, "outputs/velocity_heatmap.mp4", fps=30, dpi=500)