import sys
sys.path.append('tools')
from load_dumps import load_parquet_single, load_parquet_timesteps
import numpy as np

timesteps = load_parquet_timesteps('grid')
if timesteps:
    step, df, box = load_parquet_single('grid', timesteps[0])
    xc0 = np.sort(df['xc'].unique())
    yc0 = np.sort(df['yc'].unique())
    zc0 = np.sort(df['zc'].unique())
    print(f'Grid dimensions: {len(xc0)} x {len(yc0)} x {len(zc0)} = {len(xc0) * len(yc0) * len(zc0):,} cells')
    print(f'DataFrame shape: {df.shape}')
    print(f'2D slice (nx x ny): {len(xc0)} x {len(yc0)} = {len(xc0) * len(yc0):,} cells in heatmap')
else:
    print('No grid timesteps found')
