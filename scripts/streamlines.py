import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys

sys.path.append(os.path.expanduser('~/AMPT/tools'))
from load_dumps import load_parquet_timesteps, load_parquet_single, read_sparta_grid_dump

DUMPS_DIR = os.path.expanduser('~/AMPT/dumps')
OUTPUT_PATH = 'outputs/streamlines_2d.png'
ANIM_OUTPUT_PATH = 'outputs/streamlines_anim.mp4'
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
    u = df.loc[in_slice, 'c_prop_grid_flow[1]'].to_numpy()
    v = df.loc[in_slice, 'c_prop_grid_flow[2]'].to_numpy()

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


def discover_timesteps(prefix='grid', dumps_dir=DUMPS_DIR):
    """return sorted timesteps from parquet or text dumps for the given prefix."""
    timesteps = load_parquet_timesteps(prefix, dumps_dir)
    if timesteps:
        return timesteps

    files = [f for f in os.listdir(dumps_dir) if f.startswith(f"{prefix}.") and f.endswith('.dat')]
    if not files:
        return []
    steps = []
    for fname in files:
        try:
            steps.append(int(fname.split('.')[-2]))
        except (IndexError, ValueError):
            continue
    return sorted(steps)


def load_flow_frame(prefix, step, dumps_dir=DUMPS_DIR):
    """load a single flow frame as (step, df, box) with parquet/text fallback"""
    try:
        return load_parquet_single(prefix, step, dumps_dir)
    except Exception:
        path = os.path.join(dumps_dir, f"{prefix}.{step}.dat")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dump for step {step}: {path}")
        return read_sparta_grid_dump(path)

# --------------------------------------------------------------------------------------

def _safe_start_points(xc, yc, n_tracks):
    """Generate start points strictly inside data bounds to satisfy streamplot."""
    # small offsets to avoid touching domain extrema
    if xc.size > 1:
        dx = float(xc[1] - xc[0])
    else:
        dx = 0.0
    if yc.size > 1:
        dy = float(yc[1] - yc[0])
    else:
        dy = 0.0
    epsx = 0.25 * dx
    epsy = 0.25 * dy
    x0 = float(xc[0] + epsx)
    y_lo = float(yc[0] + epsy)
    y_hi = float(yc[-1] - epsy) if yc.size > 1 else float(yc[0])
    if y_hi < y_lo:
        y_hi = y_lo
    start_y = np.linspace(y_lo, y_hi, n_tracks, endpoint=True)
    return np.column_stack((np.full_like(start_y, x0), start_y))

# --------------------------------------------------------------------------------------

def plot_streamlines(x_centers, y_centers, x_edges, y_edges, u_grid, v_grid, speed_grid, step, dt, output_path=OUTPUT_PATH):
    """draw a speed heatmap with white streamlines so the flow direction is obvious"""
    X, Y = np.meshgrid(x_centers, y_centers)
    u_plot = np.ma.masked_invalid(u_grid)
    v_plot = np.ma.masked_invalid(v_grid)
    speed_plot = np.ma.masked_invalid(speed_grid)

    # paint the speed background in color and overlay white streamlines
    fig, ax = plt.subplots(figsize=(7, 3.5))
    cmap = plt.colormaps['coolwarm_r'].copy()
    cmap.set_bad('black')
    ax.set_aspect('auto')

    extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
    img = ax.imshow(speed_plot, extent=extent, origin='lower', cmap=cmap, aspect='auto')

    n_tracks = 100
    start_points = _safe_start_points(x_centers, y_centers, n_tracks)

    ax.streamplot(
        X,
        Y,
        u_plot,
        v_plot,
        color='white',
        density=1.0,
        linewidth=0.3,
        arrowsize=0.5,
        broken_streamlines=False,
        start_points=start_points,
    )

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    time_text = f"time = {step * dt:.2e} s" if dt is not None else f"step = {step}"
    ax.set_title(f"streamlines |z| ≤ {SLICE_FRAC:.2f}H | {time_text}")
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    # highlight AMPT box outline -------------------------------------------------------------------
    outline = Rectangle((-0.5, -0.1), 1.0, 0.2, fill=False, edgecolor='red', linewidth=1.5)
    ax.add_patch(outline)
    # ----------------------------------------------------------------------------------------------
    fig.colorbar(img, ax=ax, label='v (m/s)')
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def animate_streamlines(dump_prefix='grid', dumps_dir=DUMPS_DIR, out_path=ANIM_OUTPUT_PATH, fps=10, dpi=200, sample_percentile=95):
    """animation of streamlines over available timesteps"""
    dumps_dir = os.path.expanduser(dumps_dir)
    timesteps = discover_timesteps(dump_prefix, dumps_dir)

    if not timesteps:
        raise RuntimeError('no flow frames found for animation')

    # compute global vmin/vmax across frames (sampled)
    vals = []
    for step in timesteps:
        _, df, box = load_flow_frame(dump_prefix, step, dumps_dir)

        _, _, _, _, _, _, speed_grid = build_slice_field(df, box)
        finite = np.isfinite(speed_grid)
        if finite.any():
            vals.append(np.nanpercentile(speed_grid[finite], sample_percentile))

    vmax = float(np.nanmax(vals)) if vals else 1.0
    vmin = 0.0

    # prepare figure
    fig, ax = plt.subplots(figsize=(7, 3.5))
    cmap = plt.colormaps['coolwarm_r'].copy()
    cmap.set_bad('black')
    ax.set_aspect('auto')

    # initial frame
    # load first available frame
    first = timesteps[0]
    _, df0, box0 = load_flow_frame(dump_prefix, first, dumps_dir)

    x_centers, y_centers, x_edges, y_edges, u_grid, v_grid, speed_grid = build_slice_field(df0, box0)
    X, Y = np.meshgrid(x_centers, y_centers)
    speed_plot = np.ma.masked_invalid(speed_grid)
    extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
    im = ax.imshow(speed_plot, extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    # create initial streamlines
    n_tracks = 100
    start_y = np.linspace(y_centers[0], y_centers[-1], n_tracks, endpoint=True)
    start_points = np.column_stack((np.full_like(start_y, x_centers[0]), start_y))

    before = set(ax.get_children())
    ax.streamplot(X, Y, np.ma.masked_invalid(u_grid), np.ma.masked_invalid(v_grid), color='white', density=1.0, linewidth=0.3, arrowsize=0.5, broken_streamlines=False, start_points=start_points)
    stream_artists = [a for a in ax.get_children() if a not in before]

    outline = Rectangle((-0.5, -0.1), 1.0, 0.2, fill=False, edgecolor='red', linewidth=1.5)
    ax.add_patch(outline)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    title = ax.set_title('')
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    cbar = fig.colorbar(im, ax=ax, label='v (m/s)')
    fig.tight_layout()

    dt = read_timestep_size()

    def fresh_start_points(xc, yc):
        return _safe_start_points(xc, yc, n_tracks)

    def update(frame_idx):
        step = timesteps[frame_idx]
        _, df_f, box_f = load_flow_frame(dump_prefix, step, dumps_dir)

        x_centers_f, y_centers_f, x_edges_f, y_edges_f, u_grid_f, v_grid_f, speed_grid_f = build_slice_field(df_f, box_f)

        # update image data
        im.set_data(np.ma.masked_invalid(speed_grid_f))
        im.set_extent((x_edges_f[0], x_edges_f[-1], y_edges_f[0], y_edges_f[-1]))

        # remove previous stream artists
        nonlocal stream_artists
        for a in stream_artists:
            try:
                a.remove()
            except Exception:
                pass
        stream_artists = []

        # redraw streamlines
        Xf, Yf = np.meshgrid(x_centers_f, y_centers_f)
        before = set(ax.get_children())
        ax.streamplot(Xf, Yf, np.ma.masked_invalid(u_grid_f), np.ma.masked_invalid(v_grid_f), color='white', density=1.0, linewidth=0.3, arrowsize=0.5, broken_streamlines=False, start_points=fresh_start_points(x_centers_f, y_centers_f))
        stream_artists = [a for a in ax.get_children() if a not in before]

        # update title
        if dt is not None:
            time_text = f"time = {step * dt:.2e} s"
        else:
            time_text = f"step = {step}"
        title.set_text(f"streamlines |z| ≤ {SLICE_FRAC:.2f}H | {time_text}")
        return [im, title] + stream_artists

    from matplotlib.animation import FuncAnimation
    ani = FuncAnimation(fig, update, frames=len(timesteps), interval=200, blit=False)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ani.save(out_path, fps=fps, dpi=dpi)
    plt.close(fig)
    print(f'saved {out_path}')


def create_snapshot(dump_prefix='flow', dumps_dir=DUMPS_DIR, out_path=OUTPUT_PATH):
    """generate a single streamline snapshot for the latest timestep"""
    dumps_dir = os.path.expanduser(dumps_dir)
    timesteps = discover_timesteps(dump_prefix, dumps_dir)
    if not timesteps:
        raise RuntimeError('no flow frames found for snapshot')

    step = timesteps[-1]
    _, df, box = load_flow_frame(dump_prefix, step, dumps_dir)

    x_centers, y_centers, x_edges, y_edges, u_grid, v_grid, speed_grid = build_slice_field(df, box)
    dt = read_timestep_size()
    plot_streamlines(x_centers, y_centers, x_edges, y_edges, u_grid, v_grid, speed_grid, step, dt, output_path=out_path)
    print(f'saved {out_path}')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Generate streamline visuals from SPARTA flow dumps.')
    parser.add_argument('--anim', action='store_true', help='Generate an animated MP4 instead of a single PNG snapshot.')
    args = parser.parse_args(argv)

    if args.anim:
        animate_streamlines()
    else:
        create_snapshot()


if __name__ == '__main__':
    main()
