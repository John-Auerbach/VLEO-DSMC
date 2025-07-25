import glob, os, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from matplotlib import gridspec

# I/O
dump_glob = os.path.expanduser("~/AMPT/dumps/surf.*.dat")
input_sparta = os.path.expanduser("~/AMPT/in.ampt")
outfile = "surface_temp_heatmap.mp4"
fps = 30

# timestep size from input file
def get_tstep(path):
    with open(path) as f:
        for line in f:
            m = re.match(r"variable\s+tstep\s+equal\s+([eE\d.+-]+)", line)
            if m:
                return float(m.group(1))
    raise RuntimeError("tstep not found")

dt = get_tstep(input_sparta)

# parse one surf dump: return step, list of triangle vertices, temperature array
def read_surf(fname):
    with open(fname) as f:
        lines = f.readlines()
    step = int(lines[1].strip())
    for i, line in enumerate(lines):
        if line.startswith("ITEM: SURFS"):
            headers = line.split()[2:]
            start = i + 1
            break
    data = np.loadtxt(lines[start:])
    cols = {h: data[:, j] for j, h in enumerate(headers)}

    v1 = np.column_stack((cols["v1x"], cols["v1y"], cols["v1z"]))
    v2 = np.column_stack((cols["v2x"], cols["v2y"], cols["v2z"]))
    v3 = np.column_stack((cols["v3x"], cols["v3y"], cols["v3z"]))
    triangles = [np.array([a, b, c]) for a, b, c in zip(v1, v2, v3)]

    return step, triangles, cols["s_Tsurf"]

# Load all frames
files = sorted(glob.glob(dump_glob),
               key=lambda s: int(re.search(r"\.(\d+)\.dat$", s).group(1)))
frames = [read_surf(f) for f in files]
steps = np.array([f[0] for f in frames])
t_phys = steps * dt

# Geometry (constant triangles)
triangles = frames[0][1]
nt = len(frames)
ntri = len(triangles)

# Temperatures (nt × ntriangles)
temps = np.array([f[2] for f in frames])
vmin, vmax = np.percentile(temps, [5, 95])

# Colormap
cmap = cm.get_cmap("inferno")
norm = plt.Normalize(vmin, vmax)

# Figure setup
fig = plt.figure(figsize=(7, 4))
gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1])
ax = fig.add_subplot(gs[0], projection='3d')
cax = fig.add_subplot(gs[1])
collection = Poly3DCollection(triangles, array=temps[0], cmap=cmap, norm=norm)
collection.set_edgecolor('none')
ax.add_collection3d(collection)

fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="Tsurf (K)")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")
title = ax.set_title("")

# Fixed axis limits
ax.set_xlim([-0.5, 0.5])
ax.set_ylim([-0.5, 0.5])
ax.set_zlim([-0.5, 0.5])

# Animation update
def update(i):
    collection.set_array(temps[i])
    ax.view_init(elev=30, azim=(360 * i / nt)+180) # complete 1 full rotation from left
    title.set_text(f"t = {t_phys[i]:.2e} s")
    return collection, title

ani = FuncAnimation(fig, update, frames=nt, interval=1000 / fps, blit=False)
ani.save(outfile, writer=writers["ffmpeg"](fps=fps, bitrate=1800))
print(f"wrote {outfile}")
