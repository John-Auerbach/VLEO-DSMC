import glob
import numpy as np
import pandas as pd
import os
import sys
import argparse
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

def _read_sparta_dump(fname, count_prefix, data_prefix):
    """Stream a SPARTA dump file.

    Parses the header line-by-line and feeds the data section directly from the
    open file handle into np.loadtxt. This avoids materialising the whole file
    as a Python list of strings (and the extra sliced copy), keeping peak memory
    at roughly the size of the final data array rather than several times the
    file size. Required for large particle/grid dumps that would otherwise OOM.
    """
    step = n = None
    box = {}
    cols = None
    raw = None
    with open(fname) as f:
        while True:
            line = f.readline()
            if not line:
                break
            s = line.strip()
            if s.startswith("ITEM: TIMESTEP"):
                step = int(f.readline().strip())
            elif s.startswith(count_prefix):
                n = int(f.readline().strip())
            elif s.startswith("ITEM: BOX BOUNDS"):
                xlo, xhi = map(float, f.readline().split())
                ylo, yhi = map(float, f.readline().split())
                zlo, zhi = map(float, f.readline().split())
                box = dict(xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi, zlo=zlo, zhi=zhi)
            elif s.startswith(data_prefix):
                cols = s.split()[2:]
                # read directly from the file stream, no intermediate line list
                raw = np.loadtxt(f, dtype=float, max_rows=n)
                break
    if raw is None:
        raw = np.empty((0, len(cols) if cols else 0), dtype=float)
    raw = raw.reshape(1, -1) if raw.ndim == 1 else raw
    df = pd.DataFrame(raw, columns=cols)
    return step, df, box

def read_sparta_particle_dump(fname):
    return _read_sparta_dump(fname, "ITEM: NUMBER OF ATOMS", "ITEM: ATOMS")

def read_sparta_grid_dump(fname):
    return _read_sparta_dump(fname, "ITEM: NUMBER OF CELLS", "ITEM: CELLS")

def read_sparta_surface_dump(fname):
    return _read_sparta_dump(fname, "ITEM: NUMBER OF SURFS", "ITEM: SURFS")

def load_all(pattern, reader):
    files = sorted(glob.glob(os.path.expanduser(pattern)))
    results = []
    total = len(files)
    for i, f in enumerate(files, 1):
        print(f"{i:02d}/{total:02d}: {os.path.basename(f)}")
        results.append(reader(f))
    return results

def _convert_one_file(fname, reader, prefix, dumps_dir):
    """Read a single dump file and write its Parquet outputs.

    Module-level (picklable) so it can be dispatched to worker processes.
    Returns the basename processed for progress reporting.

    Writes each Parquet to a temporary file in the same directory and then
    atomically renames it into place. This guarantees a reader never sees a
    half-written (and thus corrupt) Parquet, e.g. when an analysis job runs
    concurrently with conversion.
    """
    step, df, box = reader(fname)

    df_path = os.path.join(dumps_dir, f"{prefix}_{step:08d}.parquet")
    _atomic_to_parquet(df, df_path)

    box_df = pd.DataFrame([box])
    box_path = os.path.join(dumps_dir, f"{prefix}_box_{step:08d}.parquet")
    _atomic_to_parquet(box_df, box_path)

    return os.path.basename(fname)

def _atomic_to_parquet(df, path):
    """Write a DataFrame to Parquet atomically (temp file + os.replace)."""
    tmp = f"{path}.tmp.{os.getpid()}"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _step_from_dump_name(fname):
    """Extract the integer timestep from a SPARTA dump filename.

    e.g. 'part.1800.dat' -> 1800, 'grid.0.dat' -> 0. Returns None if no numeric
    step component is present.
    """
    parts = os.path.basename(fname).split('.')
    for token in reversed(parts):
        if token.isdigit():
            return int(token)
    return None


def _outputs_exist(prefix, step, dumps_dir):
    """True if both the data and box Parquet for this step already exist and are
    non-empty (a 0-byte file is treated as missing so it gets rewritten)."""
    if step is None:
        return False
    df_path = os.path.join(dumps_dir, f"{prefix}_{step:08d}.parquet")
    box_path = os.path.join(dumps_dir, f"{prefix}_box_{step:08d}.parquet")
    return (os.path.exists(df_path) and os.path.getsize(df_path) > 0 and
            os.path.exists(box_path) and os.path.getsize(box_path) > 0)


def process_and_save_dumps(pattern, reader, prefix, output_dir=None, jobs=1, overwrite=False):
    """Process dump files one by one and save as Parquet to avoid memory overload.

    jobs=1 (default) processes files serially, identical to the original
    behaviour. jobs>1 fans the per-file conversion across that many worker
    processes (one file per worker). Files are independent, so this scales
    nearly linearly with cores; peak memory is roughly jobs x one-frame size,
    so only raise jobs on a high-memory node.

    By default (``overwrite=False``) any dump whose Parquet outputs already
    exist is skipped, so conversion is incremental and resumable: you can run it
    while a simulation is still producing dumps to get preliminary results, then
    re-run later to convert only the new frames. Pass ``overwrite=True`` to
    force re-conversion of every file.
    """
    files = sorted(glob.glob(os.path.expanduser(pattern)))
    if output_dir is None:
        dumps_dir = os.path.dirname(os.path.expanduser(pattern))
    else:
        dumps_dir = os.path.expanduser(output_dir)

    found = len(files)
    if not overwrite:
        kept = [f for f in files if not _outputs_exist(prefix, _step_from_dump_name(f), dumps_dir)]
        skipped = found - len(kept)
        if skipped:
            print(f"  skipping {skipped} already-converted {prefix} frame(s); {len(kept)} to do")
        files = kept
    total = len(files)

    if total == 0:
        return found

    if jobs <= 1:
        for i, f in enumerate(files, 1):
            print(f"{i:02d}/{total:02d}: {os.path.basename(f)}")
            _convert_one_file(f, reader, prefix, dumps_dir)
        return found

    worker = partial(_convert_one_file, reader=reader, prefix=prefix, dumps_dir=dumps_dir)
    done = 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futures = {ex.submit(worker, f): f for f in files}
        for fut in as_completed(futures):
            name = fut.result()
            done += 1
            print(f"{done:02d}/{total:02d}: {name}")
    return found

def save_to_parquet(data, prefix):
    """Save trajectory data as individual parquet files per timestep"""
    _TOOL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dumps_dir = os.path.join(_TOOL_ROOT, 'dumps')
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
    _TOOL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    _DEFAULT_DUMPS = os.path.join(_TOOL_ROOT, 'dumps')
    parser.add_argument('dumps_dir', nargs='?', default=_DEFAULT_DUMPS, 
                       help='Directory containing dump files (default: dumps/)')
    parser.add_argument('-j', '--jobs', type=int, default=1,
                       help='Number of parallel worker processes (default: 1 = serial). '
                            'Use the node core count on ROAR, e.g. -j 48.')
    parser.add_argument('--force', action='store_true',
                       help='Re-convert every dump, overwriting existing Parquet. '
                            'By default frames whose Parquet already exists are skipped '
                            '(incremental/resumable conversion).')
    args = parser.parse_args()
    
    dumps_path = os.path.expanduser(args.dumps_dir)
    jobs = max(1, args.jobs)
    overwrite = args.force
    
    print(f"Processing dumps from: {dumps_path}")
    if jobs > 1:
        print(f"Using {jobs} parallel workers")
    if not overwrite:
        print("Incremental mode: skipping frames already converted (use --force to redo all)")
    print("Processing particle dumps (1/3)...")
    particle_count = process_and_save_dumps(f"{dumps_path}/part.*.dat", read_sparta_particle_dump, "particle", dumps_path, jobs=jobs, overwrite=overwrite)

    print("\nProcessing grid dumps (2/3)...")
    grid_count = process_and_save_dumps(f"{dumps_path}/grid.*.dat", read_sparta_grid_dump, "grid", dumps_path, jobs=jobs, overwrite=overwrite)

    print("\nProcessing surface dumps (3/3)...")
    surf_count = process_and_save_dumps(f"{dumps_path}/surf.*.dat", read_sparta_surface_dump, "surf", dumps_path, jobs=jobs, overwrite=overwrite)

    print(f"\nCompleted:")
    print(f"particle frames: {particle_count}")
    print(f"grid frames: {grid_count}")
    print(f"surface frames: {surf_count}")

def load_parquet_data(prefix, dumps_dir=None):
    """Load parquet data back into the original format - MEMORY EFFICIENT VERSION"""
    if dumps_dir is None:
        dumps_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'dumps')
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

def load_parquet_timesteps(prefix, dumps_dir=None):
    """Get list of available timesteps without loading data"""
    if dumps_dir is None:
        dumps_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'dumps')
    dumps_dir = os.path.expanduser(dumps_dir)
    data_files = glob.glob(os.path.join(dumps_dir, f"{prefix}_[0-9]*.parquet"))
    timesteps = []
    for df_path in data_files:
        step = int(os.path.basename(df_path).split('_')[1].split('.')[0])
        timesteps.append(step)
    return sorted(timesteps)

def load_parquet_single(prefix, timestep, dumps_dir=None, columns=None):
    """Load a single timestep from parquet.

    columns : list[str] or None
        If given, only those columns are read from the data Parquet. Projecting
        to just the needed columns substantially lowers peak memory for the
        large particle frames (e.g. read only x,y,z,vx,vy,vz instead of all 8).
    """
    if dumps_dir is None:
        dumps_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'dumps')
    dumps_dir = os.path.expanduser(dumps_dir)
    
    # Load dataframe
    df_path = os.path.join(dumps_dir, f"{prefix}_{timestep:08d}.parquet")
    df = pd.read_parquet(df_path, columns=columns)
    
    # Load box info
    box_path = os.path.join(dumps_dir, f"{prefix}_box_{timestep:08d}.parquet")
    box_df = pd.read_parquet(box_path)
    box = box_df.iloc[0].to_dict()
    
    return timestep, df, box
