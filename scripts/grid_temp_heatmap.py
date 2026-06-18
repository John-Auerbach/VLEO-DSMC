import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os, re
import sys
import argparse
import copy as _copy
import warnings
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(_REPO_ROOT, 'tools'))
from load_dumps import load_parquet_timesteps, load_parquet_single
from anim_utils import compute_payloads_parallel, save_animation, render_heatmap_animation
from scipy.ndimage import gaussian_filter

# create outputs directory
os.makedirs('outputs', exist_ok=True)

# parse command line arguments
parser = argparse.ArgumentParser(description='Create grid temperature heatmap animation')
parser.add_argument('folder', nargs='?', default='dumps', 
                   help='Folder containing dump files (default: dumps)')
parser.add_argument('-j', '--jobs', type=int, default=1,
                   help='Parallel worker processes for frame precompute '
                        '(default: 1 = serial). Use the node core count on a '
                        'cluster, e.g. -j 8 (memory scales with jobs).')
parser.add_argument('--log', action='store_true',
                   help='Also write a log-colour-scale version alongside the '
                        'linear one (suffix _log). Frames are computed once and '
                        'reused, so this only adds a second encode pass.')
args = parser.parse_args()

# I/O - Get timesteps without loading data from specified folder
folder_path = os.path.join(_REPO_ROOT, args.folder)
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
temp_col = [c for c in first_df.columns if c.startswith("c_prop_grid_temp")][0]

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

def compute_frame(i):
    """Load one grid frame and return (step, temperature image). Top-level so it
    is picklable for parallel workers; relies on inherited module globals."""
    import warnings as _w
    step = timesteps[i]
    with _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        _, df, _ = load_parquet_single("grid", step, folder_path)
        img = temp_hist(df)
    return step, img

# Precompute every frame's image (the expensive load+histogram), optionally in
# parallel, with live progress. This replaces the old serial vmin/vmax pre-pass:
# the colour scale is derived from the cached images below, so each file is read
# only once instead of twice.
print(f"Precomputing {len(timesteps)} frames with {args.jobs} worker(s)...")
frame_imgs = compute_payloads_parallel(range(len(timesteps)), compute_frame,
                                       jobs=args.jobs, label="grid frame")

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

def extract_tstep_from_input(path):
    with open(path, "r") as f:
        for line in f:
            m = re.match(r"variable\s+tstep\s+equal\s+([eE\d.+-]+)", line)
            if m:
                return float(m.group(1))
    raise ValueError("tstep not found in input")

tstep = extract_tstep_from_input(os.path.join(_REPO_ROOT, 'in.runfile'))

def _title(step):
    return f"temperature heatmap |z| ≤ {slice_frac:.2f}H  |  time = {step * tstep:.2e} s"

_render_kwargs = dict(
    shape=(ny, nx),
    extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
    fps=30, dpi=300,
    cmap_name="inferno",
    cbar_label="T (K)",
    title_fn=_title,
    interpolation="bilinear",
)

# linear scale (default, unchanged output path)
render_heatmap_animation(frame_imgs, outpath="outputs/grid_temp_heatmap.mp4",
                         vmin=vmin, vmax=vmax, log=False, **_render_kwargs)

# optional log scale: reuses the cached frame_imgs, only re-renders/encodes
if args.log:
    render_heatmap_animation(frame_imgs, outpath="outputs/grid_temp_heatmap_log.mp4",
                             log=True, **_render_kwargs)
