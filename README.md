# How to Run

## 1. Install SPARTA

```bash
sudo apt update
sudo apt install build-essential gfortran mpich
git clone https://github.com/sparta/sparta.git
cd sparta/src
make serial
echo 'export PATH=$PATH:$HOME/sparta/src' >> ~/.bashrc
source ~/.bashrc
sparta -h  # should print help
```

## 2. Set Up Python Environment

```bash
cd ~/AMPT
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You must run `source .venv/bin/activate` **every time you open a new terminal**  
(unless you're using system-wide Python via `apt`, which doesn't require activation)
I have it set up like this to isolate dependencies and guarantee it will run on any machine with just requirements.txt

To auto-activate in **VS Code**:
- Press `Ctrl+Shift+P`
- Type `Python: Select Interpreter`
- Choose `.venv/bin/python` from the list

## 3. Run Simulations

### Single Altitude Simulation

```bash
# Generate atmospheric data for specific altitude (70-300 km)
python3 tools/load_atm_data.py 150

# Run SPARTA simulation (single core)
sparta < in.ampt

# OR run with multiple cores for faster execution
./run_sparta.sh    # Uses 8 cores by default. Modify this to work with your setup.
```

**Performance:** Multi-core execution is typically 4-6x faster for DSMC simulations. 

### Multi-Altitude Analysis

```bash
# Single core (default)
python3 multi_altitude.py

# Multi-core parallel execution (faster)
python3 multi_altitude.py --cores 8
python3 multi_altitude.py -c 4

# Edit altitude list in multi_altitude.py (default: 75, 80, 85, 90, 95, 100 km)
```

This will:
- Run SPARTA simulations at each altitude
- Save results to `dumps/alt_XXkm/` directories  
- **Automatically convert dumps to Parquet format** for memory-efficient analysis  

**Performance:** Using `--cores 8` is typically 4-6x faster than single core for DSMC simulations.

### Analyze Multi-Altitude Results

After running multi-altitude simulations:

```bash
python3 tools/analyze_multi_altitude.py
```

This will:
- Generate a plot of surface temperature vs altitude for each triangle
- Export data to `multi_altitude_results.csv` spreadsheet
- Save plot as `surface_temps_vs_altitude.png`
- All output files are saved to the `outputs/` folder

## 4. Convert Dump Data for Memory-Efficient Analysis

After running simulations, convert dump data to Parquet format for memory-efficient analysis:

```bash
# Convert dumps from default directory (dumps/)
python3 tools/load_dumps.py

# Convert dumps from specific directory
python3 tools/load_dumps.py dumps/alt_XXkm/

# For large datasets, process with limited memory usage
python3 tools/load_dumps.py dumps/alt_XXkm/ --memory-limit 4GB
```

This will:
- Parse raw dump files (part.*.dat, grid.*.dat, surf.*.dat) 
- Save memory-efficient Parquet files (.parquet) in the same directory (uses ~60% less RAM than pickle files)
- Can handle 10GB+ datasets without crashes
- Python analysis scripts automatically use Parquet files for streaming data access

Large SPARTA simulations can generate GB of particle data. The old pickle format was simpler in that it generated a single usable file for each set of dumps, but would cause RAM crashes when loading entire datasets. Parquet enables streaming access, loading only one timestep at a time, which reduces memory usage.

## 5. Visualization Scripts

All visualization scripts default to `dumps/` folder but can analyze specific altitude data:

```bash
# Default (uses dumps/ folder)
python3 scripts/animate_particles.py
python3 scripts/surface_temp_heatmap.py
python3 scripts/grid_temp_heatmap.py
python3 scripts/velocity_heatmap.py

# Analyze specific altitude results
python3 scripts/animate_particles.py dumps/alt_80km
python3 scripts/surface_temp_heatmap.py dumps/alt_100km
python3 scripts/grid_temp_heatmap.py dumps/alt_75km
python3 scripts/velocity_heatmap.py dumps/alt_95km
```

**Note:** 
- Run `python3 tools/load_dumps.py <folder>` first to convert dump data to Parquet format
- Scripts now use streaming data access, preventing RAM crashes on large datasets
- All output files (.mp4, .png, .csv) are saved to the `outputs/` folder