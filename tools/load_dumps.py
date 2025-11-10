import glob
import numpy as np
import pandas as pd
import os
import sys
import argparse

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
    results = []
    total = len(files)
    for i, f in enumerate(files, 1):
        print(f"{i:02d}/{total:02d}: {os.path.basename(f)}")
        results.append(reader(f))
    return results

def process_and_save_dumps(pattern, reader, prefix, output_dir=None):
    """Process dump files one by one and save as Parquet to avoid memory overload"""
    files = sorted(glob.glob(os.path.expanduser(pattern)))
    if output_dir is None:
        dumps_dir = os.path.dirname(os.path.expanduser(pattern))
    else:
        dumps_dir = os.path.expanduser(output_dir)
    total = len(files)
    
    for i, f in enumerate(files, 1):
        print(f"{i:02d}/{total:02d}: {os.path.basename(f)}")
        step, df, box = reader(f)
        
        # save dataframe immediately
        df_path = os.path.join(dumps_dir, f"{prefix}_{step:08d}.parquet")
        df.to_parquet(df_path, index=False)
        
        # save box info
        box_df = pd.DataFrame([box])
        box_path = os.path.join(dumps_dir, f"{prefix}_box_{step:08d}.parquet")
        box_df.to_parquet(box_path, index=False)
        
        # clear from memory immediately
        del df, box_df
    
    return total

def save_to_parquet(data, prefix):
    """Save trajectory data as individual parquet files per timestep"""
    dumps_dir = os.path.expanduser("~/AMPT/dumps")
    for step, df, box in data:
        # save dataframe
        df_path = os.path.join(dumps_dir, f"{prefix}_{step:08d}.parquet")
        df.to_parquet(df_path, index=False)
        
        # save box info as separate parquet (small overhead)
        box_df = pd.DataFrame([box])
        box_path = os.path.join(dumps_dir, f"{prefix}_box_{step:08d}.parquet")
        box_df.to_parquet(box_path, index=False)

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Convert SPARTA dumps to Parquet format')
    parser.add_argument('dumps_dir', nargs='?', default='~/AMPT/dumps', 
                       help='Directory containing dump files (default: ~/AMPT/dumps)')
    args = parser.parse_args()
    
    dumps_path = os.path.expanduser(args.dumps_dir)
    
    print(f"Processing dumps from: {dumps_path}")
    print("Processing particle dumps (1/4)...")
    particle_count = process_and_save_dumps(f"{dumps_path}/part.*.dat", read_sparta_particle_dump, "particle", dumps_path)

    print("\nProcessing grid dumps (2/4)...")
    grid_count = process_and_save_dumps(f"{dumps_path}/grid.*.dat", read_sparta_grid_dump, "grid", dumps_path)

    print("\nProcessing surface dumps (3/4)...")
    surf_count = process_and_save_dumps(f"{dumps_path}/surf.*.dat", read_sparta_surface_dump, "surf", dumps_path)

    print("\nProcessing flow dumps (4/4)...")
    flow_count = process_and_save_dumps(f"{dumps_path}/flow.*.dat", read_sparta_grid_dump, "flow", dumps_path)

    print(f"\nCompleted:")
    print(f"particle frames: {particle_count}")
    print(f"grid frames: {grid_count}")
    print(f"surface frames: {surf_count}")
    print(f"flow frames: {flow_count}")

def load_parquet_data(prefix, dumps_dir="~/AMPT/dumps"):
    """Load parquet data back into the original format - MEMORY EFFICIENT VERSION"""
    dumps_dir = os.path.expanduser(dumps_dir)
    data_files = sorted(glob.glob(os.path.join(dumps_dir, f"{prefix}_[0-9]*.parquet")))
    
    print(f"Loading {len(data_files)} {prefix} timesteps from Parquet...")
    if len(data_files) > 50:
        print(f"WARNING: Loading {len(data_files)} timesteps may use significant RAM.")
        print(f"Consider using load_parquet_single() for individual timesteps.")
    
    def data_generator():
        """Return a generator that gives one timestep at a time"""
        for i, df_path in enumerate(data_files, 1):
            if i % 10 == 0 or i == len(data_files):
                print(f"  Loading {i}/{len(data_files)}")
            
            # extract timestep from filename
            step = int(os.path.basename(df_path).split('_')[1].split('.')[0])
            
            # load dataframe
            df = pd.read_parquet(df_path)
            
            # load box info
            box_path = os.path.join(dumps_dir, f"{prefix}_box_{step:08d}.parquet")
            box_df = pd.read_parquet(box_path)
            box = box_df.iloc[0].to_dict()
            
            yield (step, df, box)
    
    # convert generator to list (same interface as before)
    return list(data_generator())

def load_parquet_timesteps(prefix, dumps_dir="~/AMPT/dumps"):
    """Get list of available timesteps without loading data"""
    dumps_dir = os.path.expanduser(dumps_dir)
    data_files = glob.glob(os.path.join(dumps_dir, f"{prefix}_[0-9]*.parquet"))
    timesteps = []
    for df_path in data_files:
        step = int(os.path.basename(df_path).split('_')[1].split('.')[0])
        timesteps.append(step)
    return sorted(timesteps)

def load_parquet_single(prefix, timestep, dumps_dir="~/AMPT/dumps"):
    """Load a single timestep from parquet"""
    dumps_dir = os.path.expanduser(dumps_dir)
    
    # Load dataframe
    df_path = os.path.join(dumps_dir, f"{prefix}_{timestep:08d}.parquet")
    df = pd.read_parquet(df_path)
    
    # Load box info
    box_path = os.path.join(dumps_dir, f"{prefix}_box_{timestep:08d}.parquet")
    box_df = pd.read_parquet(box_path)
    box = box_df.iloc[0].to_dict()
    
    return timestep, df, box
