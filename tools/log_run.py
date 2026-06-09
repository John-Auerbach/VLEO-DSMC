#!/usr/bin/env python3
"""Append/update a row in data/ampt_box_log.tsv from a finished SPARTA run.

Pulls fields from:
  - log.sparta              (SPARTA log)
  - in.ampt_box_Roar        (SPARTA input)
  - job_sparta.sh           (SLURM submit script -> partition, cores)
  - sacct $SLURM_JOB_ID     (when running inside a SLURM job -> credits)

Altitude is read from the comment header of data/atm.sparta written by
tools/load_atm_data.py. Rows are keyed by altitude: an existing row at the
same altitude is replaced, otherwise the row is appended. After the TSV is
updated, the markdown table in README.md (between AMPT_BOX_LOG markers) is
rewritten.

Run with no arguments:
    python tools/log_run.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TSV = ROOT / "data" / "ampt_box_log.tsv"
ATM = ROOT / "data" / "atm.sparta"
LOG = ROOT / "log.sparta"
INPUT = ROOT / "in.ampt_box_Roar"
JOBSCRIPT = ROOT / "job_sparta.sh"
README = ROOT / "README.md"
README_START = "<!-- AMPT_BOX_LOG_START -->"
README_END = "<!-- AMPT_BOX_LOG_END -->"

# Reference frontal area for C_d (ram face of 0.2 x 0.2 x 1.0 m box) [m^2]
A_REF = 0.04


# ---------- helpers ----------

def _find(pattern: str, text: str, group: int = 1, flags: int = 0) -> str | None:
    m = re.search(pattern, text, flags)
    return m.group(group) if m else None


def _findall(pattern: str, text: str, flags: int = 0) -> list[str]:
    return re.findall(pattern, text, flags)


def _fmt_particles(n: int) -> str:
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{v:.0f}M" if v.is_integer() else f"{v:.2f}M"
    if n >= 1_000:
        v = n / 1_000
        return f"{v:.0f}K" if v.is_integer() else f"{v:.2f}K"
    return str(n)


# ---------- extractors ----------

def parse_altitude(atm_path: Path) -> str:
    """Pull altitude (km) from atm.sparta header comment."""
    if not atm_path.exists():
        return ""
    head = atm_path.read_text().splitlines()[:3]
    for line in head:
        m = re.search(r"([\d.]+)\s*km", line)
        if m:
            v = float(m.group(1))
            return str(int(v)) if v.is_integer() else str(v)
    return ""


def parse_atm_props(atm_path: Path) -> dict:
    """Pull rho (kg/m^3) and vx (m/s) from atm.sparta variable definitions."""
    out: dict = {}
    if not atm_path.exists():
        return out
    text = atm_path.read_text()
    rho = _find(r"variable\s+rho\s+equal\s+([\deE.+-]+)", text)
    vx = _find(r"variable\s+vx\s+equal\s+([\deE.+-]+)", text)
    if rho:
        out["rho"] = float(rho)
    if vx:
        out["vx"] = float(vx)
    return out


def parse_input(input_path: Path) -> dict:
    """Extract Ns_target and total `run` steps from the SPARTA input."""
    out: dict = {}
    if not input_path.exists():
        return out
    text = input_path.read_text()
    ns = _find(r"Ns_target\s+equal\s+([\deE.+-]+)", text)
    if ns:
        out["Ns_target"] = int(float(ns))
    # last `run N` directive (uncommented)
    runs = _findall(r"^\s*run\s+(\d+)", text, flags=re.MULTILINE)
    if runs:
        out["total_steps"] = int(runs[-1])
    return out


def parse_log(log_path: Path) -> dict:
    """Extract grid, particles, cell/timestep, speed/step, drag, cores."""
    out: dict = {}
    if not log_path.exists():
        return out
    text = log_path.read_text()

    cores = _find(r"Running on\s+(\d+)\s+MPI task", text)
    if cores:
        out["cores"] = int(cores)

    box = re.search(
        r"Created orthogonal box\s*=\s*\(([-\d.eE+ ]+)\) to \(([-\d.eE+ ]+)\)",
        text,
    )
    grid = re.search(r"Created\s+(\d+)\s+child grid cells", text)
    nx_ny_nz = re.search(r"create_grid\s+(\d+)\s+(\d+)\s+(\d+)\s+block", text)
    if nx_ny_nz:
        nx, ny, nz = (int(v) for v in nx_ny_nz.groups())
        out["grid_str"] = f"{nx}x{ny}x{nz}"
        if box:
            lo = [float(v) for v in box.group(1).split()]
            hi = [float(v) for v in box.group(2).split()]
            dx = (hi[0] - lo[0]) / nx
            out["cell_actual"] = f"{dx:.4g}"
    if grid:
        out["total_cells"] = int(grid.group(1))

    cell_req = _find(r"CELL SIZE MUST BE LESS THAN\s+([\d.eE+-]+)\s*m", text)
    if cell_req:
        out["cell_req"] = f"{float(cell_req):.3g}"

    ts_req = _find(r"TIMESTEP MUST BE\s*<\s*([\d.eE+-]+)\s*s", text)
    if ts_req:
        out["ts_req"] = f"{float(ts_req):.3g}"

    ts_act = _find(r"^\s*timestep\s+([\d.eE+-]+)\s*$", text, flags=re.MULTILINE)
    if ts_act:
        out["ts_actual"] = f"{float(ts_act):g}"

    created = _find(r"Created\s+(\d+)\s+particles", text)
    if created:
        out["particles_created"] = int(created)

    loop = re.search(
        r"Loop time of\s+([\d.eE+-]+)\s+on\s+(\d+)\s+procs\s+for\s+(\d+)\s+steps",
        text,
    )
    if loop:
        loop_s = float(loop.group(1))
        steps = int(loop.group(3))
        out["loop_time_s"] = loop_s
        out["loop_steps"] = steps
        # ms per step
        out["speed_per_step"] = f"{(loop_s / steps) * 1000:.2f} ms"

    # final drag value: last column-c_drag from the last Step line
    # stats line: "Step CPU Np Natt Ncoll c_Tbox c_drag c_drag_xnorm c_drag_walls"
    header_m = re.search(r"^Step\s+CPU\s+Np\s+Natt.*$", text, flags=re.MULTILINE)
    if header_m:
        cols = header_m.group(0).split()
        try:
            drag_idx = cols.index("c_drag")
        except ValueError:
            drag_idx = None
        if drag_idx is not None:
            # collect numeric data lines after the header
            tail = text[header_m.end():]
            last_drag = None
            for line in tail.splitlines():
                toks = line.strip().split()
                if len(toks) > drag_idx and toks[0].lstrip("-").isdigit():
                    try:
                        last_drag = float(toks[drag_idx])
                    except ValueError:
                        pass
            if last_drag is not None:
                out["drag"] = f"{last_drag:.3g}"
    return out


def parse_jobscript(script_path: Path) -> dict:
    out: dict = {}
    if not script_path.exists():
        return out
    text = script_path.read_text()
    part = _find(r"^#SBATCH\s+--partition[= ]+(\S+)", text, flags=re.MULTILINE)
    if part:
        out["partition"] = part
    nt = _find(r"^#SBATCH\s+--ntasks[= ]+(\d+)", text, flags=re.MULTILINE)
    if nt:
        out["cores"] = int(nt)
    return out


def parse_elapsed(jobid: str) -> str:
    """Return wallclock elapsed time as 'HH:MM:SS' from sacct, or ''."""
    try:
        res = subprocess.run(
            ["sacct", "-X", "-j", str(jobid), "-P", "-n", "--format=Elapsed"],
            capture_output=True, text=True, timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    if res.returncode != 0:
        return ""
    for line in res.stdout.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def parse_credits(jobid: str) -> str:
    """Use Roar's credit_estimate tool for authoritative credit cost."""
    try:
        res = subprocess.run(
            ["credit_estimate", "-j", str(jobid)],
            capture_output=True, text=True, timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    if res.returncode != 0:
        return ""
    m = re.search(r"Estimated Cost:\s*([\d.]+)\s*credits", res.stdout)
    return m.group(1) if m else ""


def parse_memory(jobid: str) -> str:
    """Return peak resident memory used by the job from sacct MaxRSS.

    sacct reports MaxRSS per task; we take the max across all steps and
    format as a human-readable string (e.g. '12.3 GB').
    """
    try:
        res = subprocess.run(
            ["sacct", "-j", str(jobid), "-P", "-n", "--format=MaxRSS"],
            capture_output=True, text=True, timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    if res.returncode != 0:
        return ""
    max_kb = 0.0
    for line in res.stdout.splitlines():
        tok = line.strip()
        if not tok:
            continue
        m = re.match(r"([\d.]+)\s*([KMGT]?)", tok)
        if not m:
            continue
        val = float(m.group(1))
        unit = m.group(2)
        mult = {"": 1 / 1024, "K": 1, "M": 1024, "G": 1024**2, "T": 1024**3}
        kb = val * mult.get(unit, 1)
        if kb > max_kb:
            max_kb = kb
    if max_kb <= 0:
        return ""
    gb = max_kb / (1024 * 1024)
    if gb >= 1.0:
        return f"{gb:.2f} GB"
    mb = max_kb / 1024
    return f"{mb:.1f} MB"


# ---------- TSV append ----------

def append_row(row: list[str]) -> bool:
    """Append row to TSV unless an identical row already exists. Returns True if appended."""
    lines = TSV.read_text().splitlines() if TSV.exists() else []
    if not lines:
        raise SystemExit(f"{TSV} has no header")
    header = lines[0]
    body = [ln for ln in lines[1:] if ln.strip()]
    new_line = "\t".join(row)
    if new_line in body:
        return False
    body.append(new_line)
    TSV.write_text("\n".join([header, *body]) + "\n")
    return True


# ---------- README table ----------

def update_readme() -> None:
    """Rewrite the markdown table between README markers from the TSV."""
    if not README.exists():
        return
    text = README.read_text()
    if README_START not in text or README_END not in text:
        print(f"WARN: markers not found in {README}; skipping README update",
              file=sys.stderr)
        return
    rows = [ln.split("\t") for ln in TSV.read_text().splitlines() if ln != ""]
    if not rows:
        return
    header, body = rows[0], rows[1:]
    ncols = len(header)
    md = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * ncols) + "|",
    ]
    for r in body:
        r = (r + [""] * ncols)[:ncols]
        md.append("| " + " | ".join(c if c else " " for c in r) + " |")
    pre, rest = text.split(README_START, 1)
    _, post = rest.split(README_END, 1)
    README.write_text(
        f"{pre}{README_START}\n" + "\n".join(md) + f"\n{README_END}{post}"
    )
    print(f"Updated table in {README}")


# ---------- main ----------

def main() -> None:
    altitude = parse_altitude(ATM)
    if not altitude:
        print("WARN: could not determine altitude from atm.sparta", file=sys.stderr)
        altitude = "?"

    inp_d = parse_input(INPUT)
    log_d = parse_log(LOG)
    job_d = parse_jobscript(JOBSCRIPT)
    atm_d = parse_atm_props(ATM)
    jobid = os.environ.get("SLURM_JOB_ID", "")
    credits = parse_credits(jobid) if jobid else ""
    runtime = parse_elapsed(jobid) if jobid else ""
    memory = parse_memory(jobid) if jobid else ""
    if jobid:
        print(f"Using SLURM job id {jobid} for runtime/credits")

    # Compose fields
    drag = log_d.get("drag", "")
    cd = ""
    if log_d.get("drag") and atm_d.get("rho") and atm_d.get("vx"):
        try:
            cd_val = float(log_d["drag"]) / (
                0.5 * atm_d["rho"] * atm_d["vx"] ** 2 * A_REF
            )
            cd = f"{cd_val:.3g}"
        except (ValueError, ZeroDivisionError):
            pass
    cell = ""
    if log_d.get("cell_req") and log_d.get("cell_actual"):
        cell = f"{log_d['cell_req']}/{log_d['cell_actual']}"
    ts = ""
    if log_d.get("ts_req") and log_d.get("ts_actual"):
        ts = f"{log_d['ts_req']}/{log_d['ts_actual']}"

    grid_str = log_d.get("grid_str", "")
    particles = (
        _fmt_particles(inp_d["Ns_target"]) if inp_d.get("Ns_target") else
        (_fmt_particles(log_d["particles_created"]) if log_d.get("particles_created") else "")
    )
    ppc = ""
    if inp_d.get("Ns_target") and log_d.get("total_cells"):
        ppc = f"{inp_d['Ns_target'] / log_d['total_cells']:.2f}"
    partition = job_d.get("partition", "")
    cores = str(job_d.get("cores") or log_d.get("cores") or "")
    speed = log_d.get("speed_per_step", "")
    total_steps = str(inp_d.get("total_steps") or log_d.get("loop_steps") or "")

    row = [
        altitude, drag, cd, cell, ts, grid_str, particles,
        ppc, partition, cores, speed, total_steps, runtime, memory, credits,
    ]
    print("Row:", row)

    if append_row(row):
        print(f"Appended row for altitude={altitude} km to {TSV}")
    else:
        print(f"Duplicate row for altitude={altitude} km; not appended")

    update_readme()


if __name__ == "__main__":
    main()
