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

## 2. Run Simulation

```bash
cd ~/AMPT
sparta < in.ampt
```

- This creates timestep output in the `dumps/` directory.

## 3. Set Up Python Environment

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

## 4. Load and Cache Dump Data

After running the simulation:

```
python scripts/load_dumps.py
```

- This parses raw dump files in `dumps/` and saves a binary cache as `dumps/traj.pkl`
- Python analysis scripts will read from `traj.pkl` for fast access