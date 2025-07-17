import glob
import numpy as np
import pandas as pd
import re
import pickle
import os

def read_sparta_particle_dump(fname):
    # Read one text-format SPARTA particle dump file into (step, df, box)
    step = None
    natoms = None
    box = {}  # dict: {'xlo':..,'xhi':.., etc.}
    cols = None
    data_start = None
    rows = []

    with open(fname, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("ITEM: TIMESTEP"):
            step = int(lines[i+1].strip()); i += 2; continue

        if line.startswith("ITEM: NUMBER OF ATOMS"):
            natoms = int(lines[i+1].strip()); i += 2; continue

        if line.startswith("ITEM: BOX BOUNDS"):
            # format: ITEM: BOX BOUNDS xx yy zz  then 3 lines of lo hi
            xlo, xhi = map(float, lines[i+1].split())
            ylo, yhi = map(float, lines[i+2].split())
            zlo, zhi = map(float, lines[i+3].split())
            box = dict(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi, zlo=zlo, zhi=zhi)
            i += 4
            continue

        if line.startswith("ITEM: ATOMS"):
            # header columns follow 'ATOMS'
            cols = line.split()[2:]  # after ITEM: ATOMS
            data_start = i + 1
            break

        i += 1

    # Load data block
    if data_start is None:
        raise RuntimeError(f"No ATOMS section in {fname}")
    raw = np.loadtxt(lines[data_start:], dtype=float, max_rows=natoms)
    # If only one atom, np.loadtxt returns 1D
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    df = pd.DataFrame(raw, columns=cols)
    return step, df, box

def load_all_sparta_dumps():
    files = sorted(glob.glob(os.path.expanduser("~/AMPT/dumps/part.*.dat")))
    traj = []
    for f in files:
        step, df, box = read_sparta_particle_dump(f)
        traj.append((step, df, box))
    return traj

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

# Load data into cached pickle file once for easy access;
# doesn't need to be reloaded any time analysis scripts are run

traj_path = os.path.expanduser("~/AMPT/dumps/traj.pkl")

traj = load_all_sparta_dumps()
with open(traj_path, "wb") as f:
    pickle.dump(traj, f)
print(f"Parsed and cached {len(traj)} frames to dumps/traj.pkl.")
