import glob, os, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers

# I/O
dump_glob = os.path.expanduser("~/AMPT/dumps/surf.*.dat")
input_sparta = os.path.expanduser("~/AMPT/in.ampt")
outfile = "surface_temp_anim.mp4"
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

# parse one surf dump (id v1 v2 v3 s_Tsurf → centroid + temp)
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

    # compute centroid from triangle vertices
    xc = (cols["v1x"] + cols["v2x"] + cols["v3x"]) / 3.0
    yc = (cols["v1y"] + cols["v2y"] + cols["v3y"]) / 3.0
    zc = (cols["v1z"] + cols["v2z"] + cols["v3z"]) / 3.0

    return step, np.column_stack((xc, yc, zc)), cols["s_Tsurf"]

# load all frames
files = sorted(glob.glob(dump_glob),
               key=lambda s: int(re.search(r"\.(\d+)\.dat$", s).group(1)))
frames = [read_surf(f) for f in files]
steps = np.array([f[0] for f in frames])
t_phys = steps * dt

# geometry (constant)
xyz = frames[0][1]
xc, yc, zc = xyz[:, 0], xyz[:, 1], xyz[:, 2]

# temperatures (nt × nfacets)
temps = np.array([fr[2] for fr in frames])

# robust color scale (5th to 95th percentile)
vmin, vmax = np.percentile(temps, [5, 95])

# figure setup
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(xc, yc, zc, c=temps[0], cmap="inferno", vmin=vmin, vmax=vmax, s=4)
fig.colorbar(sc, ax=ax, label="Tsurf (K)")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
title = ax.set_title("")

# animation callbacks
def update(i):
    sc.set_array(temps[i])
    title.set_text(f"t = {t_phys[i]:.2e} s")
    return sc, title

ani = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
ani.save(outfile, writer=writers["ffmpeg"](fps=fps, bitrate=1800))
print(f"wrote {outfile}")
