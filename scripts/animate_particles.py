import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import argparse
import sys
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(_REPO_ROOT, 'tools'))
from load_dumps import load_parquet_timesteps, load_parquet_single
from anim_utils import compute_payloads_parallel, save_animation

# create outputs directory
os.makedirs('outputs', exist_ok=True)

# parse command line arguments
parser = argparse.ArgumentParser(description='Animate particle trajectories')
parser.add_argument('folder', nargs='?', default='dumps', 
                   help='Folder containing dump files (default: dumps)')
parser.add_argument('-j', '--jobs', type=int, default=1,
                   help='Parallel worker processes for frame precompute '
                        '(default: 1 = serial). Use the node core count on a '
                        'cluster, e.g. -j 8 (memory scales with jobs).')
args = parser.parse_args()

# load timestep list (memory efficient)
folder_path = os.path.join(_REPO_ROOT, args.folder)
timesteps = load_parquet_timesteps("particle", folder_path)
print(f"Found {len(timesteps)} particle timesteps in {args.folder}")

# get box extents from first frame
first_step, first_df, box0 = load_parquet_single("particle", timesteps[0], folder_path)

# species color mapping
type_to_color = {1: 0, 2: 1}  # map species type to color index
colors_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

# subsample for speed
subsample = 5000
def get_frame_data(df):
    if subsample and len(df) > subsample:
        return df.sample(subsample, random_state=0)
    return df

def compute_frame(i):
    """Load + subsample one particle frame, returning (step, (x, y, z, colors)).
    Top-level so it is picklable for parallel workers."""
    step = timesteps[i]
    _, df, _ = load_parquet_single("particle", step, folder_path)
    dff = get_frame_data(df)
    x = dff['x'].values
    y = dff['y'].values
    z = dff['z'].values
    t = dff['type'].astype(int).map(type_to_color).fillna(0).astype(int).values
    c = [colors_cycle[k % len(colors_cycle)] for k in t]
    return step, (x, y, z, c)

xlim = (box0['xlo'], box0['xhi'])
ylim = (box0['ylo'], box0['yhi'])
zlim = (box0['zlo'], box0['zhi'])

# Precompute every (subsampled) frame, optionally in parallel, with live
# progress, so frames are read once and the animation assembles quickly.
print(f"Precomputing {len(timesteps)} frames with {args.jobs} worker(s)...")
frame_data = compute_payloads_parallel(range(len(timesteps)), compute_frame,
                                       jobs=args.jobs, label="particle frame")

# plot
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(projection='3d')
scat = ax.scatter([], [], [], s=2)

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_zlim(*zlim)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")
title = ax.set_title("")

def init():
    scat._offsets3d = ([], [], [])
    return scat,

def update(frame_idx):
    step, (x, y, z, c) = frame_data[frame_idx]
    scat._offsets3d = (x, y, z)
    scat.set_color(c)
    title.set_text(f"Step {step}")
    return scat, title

ani = FuncAnimation(fig, update, frames=len(timesteps), init_func=init, blit=False, interval=200)
save_animation(ani, "outputs/particle_anim.mp4", fps=5, dpi=150)
print("Animation saved to outputs/particle_anim.mp4")
