import os
import re
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.expanduser('~/AMPT/tools'))
from load_dumps import load_parquet_timesteps, load_parquet_single

DUMPS_DIR = os.path.expanduser('~/AMPT/dumps')
OUTPUT_PATH = 'outputs/streamlines_2d.png'
SLICE_FRAC = 0.05 # fraction of box height for z slice
EMPTY_SPEED_EPS = 1e-3 # speed below which a cell is considered empty


def read_timestep_size():
    """look up the timestep in in.ampt to label the figure with physical time"""
    input_path = os.path.expanduser('~/AMPT/in.ampt')
    try:
        with open(input_path, 'r', encoding='utf-8') as fh:
            for line in fh:
                match = re.match(r'variable\s+tstep\s+equal\s+([eE\d.+-]+)', line)
                if match:
                    return float(match.group(1))
    except FileNotFoundError:
        return None
    return None


def build_slice_field(df, box):
    """take velocity cells near z=0 and average them onto the natural x-y mesh"""
    delta_z = SLICE_FRAC * (box['zhi'] - box['zlo'])
    in_slice = np.abs(df['zc'].to_numpy()) <= delta_z

    # grab just the slice values needed for coordinates and horizontal velocity
    x = df.loc[in_slice, 'xc'].to_numpy()
    y = df.loc[in_slice, 'yc'].to_numpy()
    u = df.loc[in_slice, 'c_compute_flow[1]'].to_numpy()
    v = df.loc[in_slice, 'c_compute_flow[2]'].to_numpy()

    # drop cells containing no particles and leave blank in the figure
    speed = np.hypot(u, v)
    filled = speed > EMPTY_SPEED_EPS
    x = x[filled]
    y = y[filled]
    u = u[filled]
    v = v[filled]

    if x.size == 0:
        raise RuntimeError('no non-empty cells in slice; check dump or adjust constants')

    nx = np.unique(x).size
    ny = np.unique(y).size
    x_edges = np.linspace(box['xlo'], box['xhi'], nx + 1)
    y_edges = np.linspace(box['ylo'], box['yhi'], ny + 1)

    # average remaining velocity vectors onto each mesh cell
    sum_u, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=u)
    sum_v, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=v)
    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

    sum_u = sum_u.T
    sum_v = sum_v.T
    counts = counts.T

    with np.errstate(divide='ignore', invalid='ignore'):
        u_grid = np.where(counts > 0, sum_u / counts, np.nan)
        v_grid = np.where(counts > 0, sum_v / counts, np.nan)

    speed_grid = np.hypot(u_grid, v_grid)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return x_centers, y_centers, x_edges, y_edges, u_grid, v_grid, speed_grid


def plot_streamlines(x_centers, y_centers, x_edges, y_edges, u_grid, v_grid, speed_grid, step, dt):
    """draw a speed heatmap with white streamlines so the flow direction is obvious"""
    X, Y = np.meshgrid(x_centers, y_centers)
    u_plot = np.ma.masked_invalid(u_grid)
    v_plot = np.ma.masked_invalid(v_grid)
    speed_plot = np.ma.masked_invalid(speed_grid)

    # paint the speed background in color and overlay white streamlines
    fig, ax = plt.subplots(figsize=(7, 3.5))
    cmap = plt.colormaps['viridis'].copy()
    cmap.set_bad('black')
    ax.set_aspect('auto')

    extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
    img = ax.imshow(speed_plot, extent=extent, origin='lower', cmap=cmap, aspect='auto')

    ax.streamplot(
        X,
        Y,
        u_plot,
        v_plot,
        color='white',
        density=1.0,
        linewidth=0.8,
        arrowsize=1.0,
        broken_streamlines=False,
    )

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    time_text = f"time = {step * dt:.2e} s" if dt is not None else f"step = {step}"
    ax.set_title(f"streamlines |z| ≤ {SLICE_FRAC:.2f}H | {time_text}")
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    fig.colorbar(img, ax=ax, label='|u| (m/s)')
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300)
    plt.close(fig)


def main():
    """load the newest flow snapshot, slice it, and save the streamline plot"""
    # pick the newest available flow parquet file in the dumps folder
    timesteps = load_parquet_timesteps('flow', DUMPS_DIR)
    if not timesteps:
        raise RuntimeError('no flow parquet files found; run load_dumps.py first')

    step = timesteps[-1]
    _, df, box = load_parquet_single('flow', step, DUMPS_DIR)

    x_centers, y_centers, x_edges, y_edges, u_grid, v_grid, speed_grid = build_slice_field(df, box)
    dt = read_timestep_size()
    plot_streamlines(x_centers, y_centers, x_edges, y_edges, u_grid, v_grid, speed_grid, step, dt)
    print(f'saved {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
