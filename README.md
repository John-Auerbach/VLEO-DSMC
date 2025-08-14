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
- Generate a plot of surface temperature vs altitude for each triangle
- Save plot as `surface_temps_vs_altitude.png`

**Performance:** Using `--cores 8` is typically 4-6x faster than single core for DSMC simulations.

## 4. Load and Cache Dump Data

After running simulations, you can cache dump data for analysis with the included python scripts:

```bash
# Load dumps from default directory (dumps/)
python3 tools/load_dumps.py

# Load dumps from specific directory
python3 tools/load_dumps.py dumps/alt_XXkm/

```

This will:
- Parse raw dump files (part.*.dat, grid.*.dat, surf.*.dat) 
- Save binary cache files (traj.pkl, grid.pkl, surf.pkl) in the same directory
- Python analysis scripts can read from .pkl files for much faster access

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

**Note:** Run `python3 tools/load_dumps.py <folder>` first to cache dump data for faster visualization.