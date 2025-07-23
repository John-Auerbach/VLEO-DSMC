import glob
import numpy as np
import pandas as pd
import pickle
import os

def read_sparta_particle_dump(fname):
    step = natoms = None
    box = {}
    cols = None
    data_start = None
    lines = open(fname).read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("ITEM: TIMESTEP"):
            step = int(lines[i + 1].strip())
            i += 2
            continue
        if line.startswith("ITEM: NUMBER OF ATOMS"):
            natoms = int(lines[i + 1].strip())
            i += 2
            continue
        if line.startswith("ITEM: BOX BOUNDS"):
            xlo, xhi = map(float, lines[i + 1].split())
            ylo, yhi = map(float, lines[i + 2].split())
            zlo, zhi = map(float, lines[i + 3].split())
            box = dict(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi, zlo=zlo, zhi=zhi)
            i += 4
            continue
        if line.startswith("ITEM: ATOMS"):
            cols = line.split()[2:]
            data_start = i + 1
            break
        i += 1
    raw = np.loadtxt(lines[data_start:], dtype=float, max_rows=natoms)
    raw = raw.reshape(1, -1) if raw.ndim == 1 else raw
    df = pd.DataFrame(raw, columns=cols)
    return step, df, box

def read_sparta_grid_dump(fname):
    step = ncells = None
    box = {}
    cols = None
    data_start = None
    lines = open(fname).read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("ITEM: TIMESTEP"):
            step = int(lines[i + 1].strip())
            i += 2
            continue
        if line.startswith("ITEM: NUMBER OF CELLS"):
            ncells = int(lines[i + 1].strip())
            i += 2
            continue
        if line.startswith("ITEM: BOX BOUNDS"):
            xlo, xhi = map(float, lines[i + 1].split())
            ylo, yhi = map(float, lines[i + 2].split())
            zlo, zhi = map(float, lines[i + 3].split())
            box = dict(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi, zlo=zlo, zhi=zhi)
            i += 4
            continue
        if line.startswith("ITEM: CELLS"):
            cols = line.split()[2:]
            data_start = i + 1
            break
        i += 1
    raw = np.loadtxt(lines[data_start:], dtype=float, max_rows=ncells)
    raw = raw.reshape(1, -1) if raw.ndim == 1 else raw
    df = pd.DataFrame(raw, columns=cols)
    return step, df, box

def read_sparta_surface_dump(fname):
    step = nsurf = None
    box = {}
    cols = None
    data_start = None
    lines = open(fname).read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("ITEM: TIMESTEP"):
            step = int(lines[i + 1].strip())
            i += 2
            continue
        if line.startswith("ITEM: NUMBER OF SURFS"):
            nsurf = int(lines[i + 1].strip())
            i += 2
            continue
        if line.startswith("ITEM: BOX BOUNDS"):
            xlo, xhi = map(float, lines[i + 1].split())
            ylo, yhi = map(float, lines[i + 2].split())
            zlo, zhi = map(float, lines[i + 3].split())
            box = dict(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi, zlo=zlo, zhi=zhi)
            i += 4
            continue
        if line.startswith("ITEM: SURFS"):
            cols = line.split()[2:]
            data_start = i + 1
            break
        i += 1
    raw = np.loadtxt(lines[data_start:], dtype=float, max_rows=nsurf)
    raw = raw.reshape(1, -1) if raw.ndim == 1 else raw
    df = pd.DataFrame(raw, columns=cols)
    return step, df, box

def load_all(pattern, reader):
    files = sorted(glob.glob(os.path.expanduser(pattern)))
    return [reader(f) for f in files]

particle_data = load_all("~/AMPT/dumps/part.*.dat", read_sparta_particle_dump)
'''
Output is list of snapshots containing: 
    integer timestep number, 
    pandas dataframe of all particle positions and velocities,
    dictionary of simulation bounds; i.e. {'xlo':0.0, 'xhi':5.0, ...}

Example:

traj = [
  (step_0, df_0, box_0),
  (step_1, df_1, box_1),
  ...
]
'''
grid_data = load_all("~/AMPT/dumps/grid.*.dat", read_sparta_grid_dump)
'''
Output is list of snapshots containing: 
    integer timestep number, 
    pandas dataframe of all particle positions and velocities,
    dictionary of simulation bounds; i.e. {'xlo':0.0, 'xhi':5.0, ...}

Example:

traj = [
  (step_0, df_0, box_0),
  (step_1, df_1, box_1),
  ...
]
'''
surf_data = load_all("~/AMPT/dumps/surf.*.dat", read_sparta_surface_dump)
'''
Output is list of snapshots containing: 
    integer timestep number, 
    pandas dataframe of all particle positions and velocities,
    dictionary of simulation bounds; i.e. {'xlo':0.0, 'xhi':5.0, ...}

Example:

traj = [
  (step_0, df_0, box_0),
  (step_1, df_1, box_1),
  ...
]
'''
pickle.dump(particle_data, open(os.path.expanduser("~/AMPT/dumps/traj.pkl"), "wb"))
pickle.dump(grid_data, open(os.path.expanduser("~/AMPT/dumps/grid.pkl"), "wb"))
pickle.dump(surf_data, open(os.path.expanduser("~/AMPT/dumps/surf.pkl"), "wb"))

print(f"particle frames: {len(particle_data)}")
print(f"grid frames: {len(grid_data)}")
print(f"surface frames: {len(surf_data)}")
