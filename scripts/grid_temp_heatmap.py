import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pickle, os, re

# I/O
grid_path = os.path.expanduser("~/AMPT/dumps/grid.pkl")
if not os.path.exists(grid_path):
    raise FileNotFoundError(f"Pickle file '{grid_path}' not found. Run load_dumps.py first.")

with open(grid_path, "rb") as f:
    grid = pickle.load(f)

grid.sort(key=lambda tup: tup[0])  # sort by timestep

# box extents
_, _, box0 = grid[0]
xlim = (box0["xlo"], box0["xhi"])
ylim = (box0["ylo"], box0["yhi"])
zlim = (box0["zlo"], box0["zhi"])

# choose slice thickness about z=0
slice_frac = 0.1
delta_z = slice_frac * (zlim[1] - zlim[0])

# target grid size for heatmap
nx, ny = 300, 200
x_edges = np.linspace(*xlim, nx + 1)
y_edges = np.linspace(*ylim, ny + 1)

# infer number of z-slices
ncells = len(grid[0][1])
nz_guess = round(ncells ** (1 / 3))
if nz_guess ** 3 == ncells:
    nz = nz_guess
else:
    nz = 1
dz = (zlim[1] - zlim[0]) / max(nz, 1)

# temperature column name
temp_col = [c for c in grid[0][1].columns if c != "id"][0]

def temp_hist(df):
    """Return 2D array of mean Tgrid value for |z| < delta_z, binned on (x,y)."""
    ids = df["id"].astype(int).to_numpy() - 1
    ix = ids % nx
    iy = (ids // nx) % ny
    iz = ids // (nx * ny) if nz > 1 else np.zeros_like(ix)

    xc = xlim[0] + (ix + 0.5) * (xlim[1] - xlim[0]) / nx
    yc = ylim[0] + (iy + 0.5) * (ylim[1] - ylim[0]) / ny
    zc = zlim[0] + (iz + 0.5) * dz

    mask = np.abs(zc) <= delta_z
    if not mask.any():
        return np.full((ny, nx), np.nan)

    temps = df[temp_col].to_numpy()
    xs, ys, ts = xc[mask], yc[mask], temps[mask]

    sum_t, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges], weights=ts)
    cnt_t, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
    with np.errstate(invalid="ignore"):
        mean_t = np.divide(sum_t, cnt_t, where=cnt_t > 0)
    return np.flipud(mean_t.T)

# precompute vmin/vmax
print("Computing min/max for color scale...")
all_vals = []
for _, df, _ in grid:
    img = temp_hist(df)
    if np.isfinite(img).any():
        all_vals.extend(img[np.isfinite(img)])
vmin = np.percentile(all_vals, 5)
vmax = np.percentile(all_vals, 95)
print(f"vmin={vmin:.2f}, vmax={vmax:.2f}")

fig, ax = plt.subplots(figsize=(6, 3))
im = ax.imshow(
    np.zeros((ny, nx)),
    extent=(*xlim, *ylim),
    origin="lower",
    aspect="auto",
    vmin=vmin, vmax=vmax,
    cmap="inferno"
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
    step, df, _ = grid[i]
    img = temp_hist(df)
    im.set_data(img)
    title.set_text(f"temperature heatmap |z| ≤ {slice_frac:.2f}H  |  time = {step * tstep:.2e} s")
    return im, title

ani = FuncAnimation(fig, update, frames=len(grid), init_func=init, blit=False, interval=200)
ani.save("grid_temp_heatmap.mp4", fps=30, dpi=300)
