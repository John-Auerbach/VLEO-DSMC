#!/usr/bin/env python3
"""
Ethan_drag_theory.py

This script was copied from Ethan's original MATLAB script and converted to Python. 
Thanks Ethan!

Compare analytical solutions of a cube in continuum and free-molecular flow
(FMF) regimes to DSMC drag results to verify the accuracy of the DSMC code
and identify where the continuum and FMF assumptions break down.

Atmospheric conditions vs altitude (70-300 km) are obtained from the
NRLMSISE-00 model via the `pymsis` package.

"""

import csv
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

from pymsis import msis


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NA = 6.02214076e23           # Avogadro's number [1/mol]
kB = 1.380649e-23            # Boltzmann constant [J/K]
R_E = 6371.0                 # Earth radius [km]
MU_E = 3.986e5               # Earth gravitational parameter [km^3/s^2]


# ---------------------------------------------------------------------------
# FMF and continuum analytical functions
# ---------------------------------------------------------------------------
def fmf_flat_plate(v, rho_row, MW, T_atm, xsec, AoA_deg, sigma, T_w, kB, NA):
    """
    Free-molecular-flow drag on a flat plate.

    Equations from "Aerodynamic drag analysis and reduction strategy for
    satellites in Very Low Earth Orbit", Yifan Jiang, 2023.

    Parameters
    ----------
    v        : orbital velocity [m/s]
    rho_row  : 1-D array of number/mass densities at this altitude
               [He, O, N2, O2, Ar, total_mass, H, N, anom-O]  (indices 0..8)
    MW       : molecular weights for the first 8 species [g/mol]
    T_atm    : atmospheric temperature [K]
    xsec     : reference cross-sectional area [m^2]
    AoA_deg  : angle of attack [deg]
    sigma    : accommodation coefficient
    T_w      : wall temperature [K]
    """
    # Specific gas constant from number-weighted molecular weight
    n_species = rho_row[0:8]
    R_sp = (kB * np.sum(n_species)) / np.sum(n_species * (np.asarray(MW) / NA))

    S = v / np.sqrt(2.0 * R_sp * T_atm)
    S_r = v / np.sqrt(2.0 * R_sp * T_w)

    AoA = np.deg2rad(AoA_deg)
    sinA = np.sin(AoA)
    cos2A = np.cos(2.0 * AoA)

    Cd = (1.0 / S**2) * (
        (S / np.sqrt(np.pi))
        * (4.0 * sinA**2 + 2.0 * sigma * cos2A)
        * np.exp(-(S * sinA) ** 2)
        + sinA
        * (1.0 + 2.0 * S**2 + (1.0 - sigma) * (1.0 - 2.0 * S**2 * cos2A))
        * erf(S * sinA)
        + sigma * np.sqrt(np.pi) * (S**2 / S_r) * sinA**2
    )

    rho_mass = rho_row[5]  # total mass density [kg/m^3]
    drag = 0.5 * rho_mass * v**2 * xsec * Cd
    return drag


def prism_drag(mu_0, T_atm, T0, rho_mass, v, A, L):
    """
    Continuum-regime drag on a cube using the Clift empirical Cd(Re) fit.
    Returns NaN if Re is outside [0.1, 3e5] where the correlation is valid.
    Sutherland's law is used for dynamic viscosity.
    """
    dyn_visc = mu_0 * (T_atm / T0) ** 1.5 * ((T0 + 110.4) / (T_atm + 110.4))
    Re = (rho_mass * v * L) / dyn_visc

    if Re < 0.1 or Re > 3e5:
        return np.nan

    Cd = (24.0 / Re) * (1.0 + 0.15 * Re**0.687) + 0.42 / (
        1.0 + 4.25e4 * Re ** (-1.16)
    )
    return 0.5 * rho_mass * v**2 * Cd * A


# ---------------------------------------------------------------------------
# Atmospheric data via NRLMSISE-00 (pymsis)
# ---------------------------------------------------------------------------
def get_atmosphere(alt_m):
    """
    Returns:
      T_atm  : (N,) translational temperature [K]
      rho    : (N, 9) array with columns matching the MATLAB ordering:
               [He, O, N2, O2, Ar, total_mass[kg/m^3], H, N, anom-O]
               (number densities in [1/m^3] except column 5 which is mass)
    """
    alt_km = alt_m / 1e3

    lat = 40.7934
    lon = -77.8600
    utc = datetime.datetime(2011, 4, 10, 0, 0)  # Day 100, midnight
    times = np.array([utc], dtype="datetime64")

    f107a = 250.0
    f107 = 250.0
    aps = [[4, 4, 4, 4, 4, 4, 4]]

    # pymsis.msis.run returns shape (ndates, nlons, nlats, nalts, 11) with
    # columns: [mass_rho, N2, O2, O, He, H, Ar, N, anomO, NO, T]
    atm = msis.run(
        times, [lon], [lat], alt_km, f107, f107a, aps, version=0
    )
    atm = np.squeeze(atm)  # -> (nalts, 11)

    rho_mass = atm[:, 0]
    n_N2 = atm[:, 1]
    n_O2 = atm[:, 2]
    n_O = atm[:, 3]
    n_He = atm[:, 4]
    n_H = atm[:, 5]
    n_Ar = atm[:, 6]
    n_N = atm[:, 7]
    n_anomO = atm[:, 8]
    T_atm = atm[:, 10]

    # Replace NaNs (pymsis returns NaN for species above their model ceiling)
    for arr in (n_He, n_O, n_N2, n_O2, n_Ar, n_H, n_N, n_anomO):
        np.nan_to_num(arr, copy=False, nan=0.0)

    rho = np.column_stack(
        [n_He, n_O, n_N2, n_O2, n_Ar, rho_mass, n_H, n_N, n_anomO]
    )
    return T_atm, rho


def load_ampt_box_log(path):
    """
    Load altitude [km] and drag [N] from data/ampt_box_log.tsv.

    Returns two 1-D numpy arrays (alt_km, drag_N). Returns empty arrays if
    the file is missing or contains no valid numeric rows.
    """
    if not os.path.isfile(path):
        return np.array([]), np.array([])

    alts, drags = [], []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            a_str, d_str = row[0].strip(), row[1].strip()
            if not a_str or not d_str:
                continue
            try:
                alts.append(float(a_str))
                drags.append(float(d_str))
            except ValueError:
                continue
    return np.array(alts), np.array(drags)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    alt = np.arange(70e3, 300e3 + 1, 1e3)  # [m]

    T_atm, rho = get_atmosphere(alt)

    # Molecular weights of the first 8 constituents [g/mol]
    MW = np.array([4.0003, 16.0, 28.014, 32.0, 39.95, 0.0, 1.008, 14.007])

    # Orbital velocity [m/s]
    v = 1000.0 * np.sqrt(MU_E / (alt / 1000.0 + R_E))

    T_wall = 300.0  # [K]

    # FMF drag: one ram face (xsec = 0.04 m^2) + four lateral faces (0.2 m^2)
    drag_FMF = np.zeros_like(alt)
    for n in range(len(alt)):
        drag_FMF[n] = (
            fmf_flat_plate(
                v[n], rho[n, :], MW, T_atm[n], 0.04, 0.0, 0.9, T_wall, kB, NA
            )
            + 4.0
            * fmf_flat_plate(
                v[n], rho[n, :], MW, T_atm[n], 0.2, 90.0, 0.9, T_wall, kB, NA
            )
        )

    # Continuum drag (Clift correlation; Sutherland viscosity)
    mu_0 = 1.716e-5  # [kg/m*s]
    T0 = 298.15      # [K]

    drag_C = np.zeros_like(alt)
    for n in range(len(alt)):
        drag_C[n] = prism_drag(mu_0, T_atm[n], T0, rho[n, 5], v[n], 0.04, 0.2) # 0.2x0.2x1 prism with 0.04 m^2 frontal area and 0.2 m length

    # Cube Cd in continuum regime (reference area = ram face)
    with np.errstate(invalid="ignore"):
        Cd_C = drag_C / (0.5 * rho[:, 5] * v**2 * 0.04)

    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------
    fig1, ax1 = plt.subplots()
    ax1.plot(alt / 1000.0, drag_FMF, label="FMF")
    ax1.plot(alt / 1000.0, drag_C, label="Continuum")

    # Overlay DSMC results from data/ampt_box_log.tsv if present
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(
        os.path.dirname(script_dir), "data", "ampt_box_log.tsv"
    )
    dsmc_alt, dsmc_drag = load_ampt_box_log(log_path)
    if dsmc_alt.size > 0:
        ax1.scatter(dsmc_alt, dsmc_drag, marker="*", color="k", label="DSMC")
    else:
        print(f"No DSMC data rows found in {log_path}")

    ax1.set_xlabel("Altitude [km]")
    ax1.set_ylabel("Drag [N]")
    ax1.set_yscale("log")
    ax1.grid(True, which="both")
    ax1.legend()
    fig1.patch.set_facecolor("w")

    fig2, ax2 = plt.subplots()
    ax2.plot(alt / 1000.0, Cd_C)
    ax2.set_xlabel("Altitude [km]")
    ax2.set_ylabel(r"$C_d$")
    ax2.set_xlim([70, 100])
    ax2.grid(True)
    fig2.patch.set_facecolor("w")

    plt.show()


if __name__ == "__main__":
    main()
