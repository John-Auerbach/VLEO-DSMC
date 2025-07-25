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
fps = 25

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

    return step, triangles, cols["s_Tsurf"], cols["f_flux[*]"]

# load all frames
files = sorted(glob.glob(dump_glob),
               key=lambda s: int(re.search(r"\.(\d+)\.dat$", s).group(1)))
frames = [read_surf(f) for f in files]
steps = np.array([f[0] for f in frames])
t_phys = steps * dt

triangles = frames[0][1]
nt = len(frames)
ntri = len(triangles)

temps = np.array([f[2] for f in frames])
fluxes = np.array([f[3] for f in frames])
vmin, vmax = np.percentile(temps, [0, 100])

# Colormap
cmap = cm.get_cmap("inferno")
norm = plt.Normalize(vmin, vmax)

fig = plt.figure(figsize=(9, 4.5))
gs = gridspec.GridSpec(1, 3, width_ratios=[5, 20, 1], wspace=0.01)
text_ax = fig.add_subplot(gs[0])
ax = fig.add_subplot(gs[1], projection='3d')
fig.subplots_adjust(left=0.05, right=0.95)
cax = fig.add_axes([0.86, 0.15, 0.015, 0.7])
text_ax.set_xlim(0, 1)
text_ax.set_ylim(0, 1)
text_ax.axis("off")

collection = Poly3DCollection(triangles, array=temps[0], cmap=cmap, norm=norm)
collection.set_edgecolor('none')
ax.add_collection3d(collection)

fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="Tsurf (K)")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")
title = ax.set_title("")
text_display = text_ax.text(0.18, 0.5, "", fontsize=12, va="center", ha="left", family="monospace")

ax.set_xlim([-0.5, 0.5])
ax.set_ylim([-0.5, 0.5])
ax.set_zlim([-0.5, 0.5])

# animation update
def update(i):
    collection.set_array(temps[i])
    ax.view_init(elev=30, azim=60 + (180 * np.cos((i / nt)*np.pi)))
    title.set_text(f"t = {t_phys[i]:.2e} s")
    flux_lines = [f"{j:2d}: {fluxes[i][j]:.2e} W/m²" for j in range(ntri)]
    text_display.set_text("\n".join(flux_lines))
    return collection, title, text_display

ani = FuncAnimation(fig, update, frames=nt, interval=1000 / fps, blit=False)
ani.save(outfile, writer=writers["ffmpeg"](fps=fps, bitrate=1800))
print(f"wrote {outfile}")
