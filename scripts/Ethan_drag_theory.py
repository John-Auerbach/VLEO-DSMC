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
D_MOL = 3.7e-10              # effective molecular (collision) diameter [m]


# ---------------------------------------------------------------------------
# Flow-regime definitions (by Knudsen number)
# ---------------------------------------------------------------------------
# Each entry: (label, Kn_lower, Kn_upper, shading color)
FLOW_REGIMES = [
    ("Continuum Flow (Kn < 0.01)", 0.0, 0.01, "#ff7f0e"),
    ("Slip Flow (0.01 < Kn < 0.1)", 0.01, 0.1, "#ffd700"),
    ("Transitional Flow (0.1 < Kn < 10)", 0.1, 10.0, "#2ca02c"),
    ("Free Molecular / Knudsen Flow (Kn > 10)", 10.0, np.inf, "#1f77b4"),
]


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


def knudsen_number(rho, d, L):
    """
    Knudsen number Kn = lambda / L for each altitude.

    The hard-sphere mean free path lambda = 1 / (sqrt(2) * pi * d^2 * n_tot)
    uses the total number density summed over all atmospheric species.

    Parameters
    ----------
    rho : (N, 9) array with the MATLAB column ordering
          [He, O, N2, O2, Ar, total_mass, H, N, anom-O]; columns other than
          index 5 are number densities [1/m^3].
    d   : effective molecular collision diameter [m]
    L   : characteristic body length [m]

    Returns
    -------
    mfp : (N,) mean free path [m]
    Kn  : (N,) Knudsen number [-]
    """
    number_density_cols = [0, 1, 2, 3, 4, 6, 7, 8]  # all columns except mass (5)
    n_tot = np.sum(rho[:, number_density_cols], axis=1)
    mfp = 1.0 / (np.sqrt(2.0) * np.pi * d**2 * n_tot)
    return mfp, mfp / L


def shade_flow_regimes(ax, alt_km, Kn, alpha=0.18):
    """
    Shade the altitude span of each flow regime (per Knudsen number) on a plot
    whose x-axis is altitude [km]. Boundaries are found by interpolating the
    altitude at which Kn crosses each regime threshold (Kn increases
    monotonically with altitude).
    """
    order = np.argsort(Kn)
    kn_sorted = Kn[order]
    alt_sorted = alt_km[order]
    for label, kn_lo, kn_hi, color in FLOW_REGIMES:
        lo_alt = np.interp(kn_lo, kn_sorted, alt_sorted)
        hi_alt = np.interp(kn_hi, kn_sorted, alt_sorted)
        if hi_alt > lo_alt:
            ax.axvspan(lo_alt, hi_alt, color=color, alpha=alpha,
                       zorder=0, label=label)


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
    Load altitude [km], drag [N], and C_d from data/ampt_box_log.tsv.

    Returns three 1-D numpy arrays (alt_km, drag_N, c_d). Returns empty
    arrays if the file is missing or contains no valid numeric rows.
    """
    if not os.path.isfile(path):
        return np.array([]), np.array([]), np.array([])

    alts, drags, cds = [], [], []
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
                a = float(a_str)
                d = float(d_str)
            except ValueError:
                continue
            alts.append(a)
            drags.append(d)
            cd_val = np.nan
            if len(row) >= 3 and row[2].strip():
                try:
                    cd_val = float(row[2].strip())
                except ValueError:
                    pass
            cds.append(cd_val)
    return np.array(alts), np.array(drags), np.array(cds)


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

    # Body geometry: rectangular prism. x is the flow (ram) direction.
    # Set the three dimensions and every reference area / length is derived.
    body_L = 1.0   # length along flow, x [m]
    body_W = 0.2   # width, y [m]
    body_H = 0.2   # height, z [m]

    A_ref = body_W * body_H     # frontal / ram area (x-normal face) [m^2]
    A_face_y = body_L * body_H  # area of each y-normal (lateral) face [m^2]
    A_face_z = body_L * body_W  # area of each z-normal (lateral) face [m^2]
    L_char = body_W             # characteristic length for Kn / Re [m]

    # Knudsen number vs altitude for this body
    mfp, Kn = knudsen_number(rho, D_MOL, L_char)

    # FMF drag: one ram face (broadside -> AoA=90 deg) + the four lateral
    # faces (edge-on -> AoA=0 deg), all areas derived from body dimensions.
    drag_FMF = np.zeros_like(alt)
    for n in range(len(alt)):
        drag_FMF[n] = (
            fmf_flat_plate(
                v[n], rho[n, :], MW, T_atm[n], A_ref, 90.0, 0.9, T_wall, kB, NA
            )
            + 2.0
            * fmf_flat_plate(
                v[n], rho[n, :], MW, T_atm[n], A_face_y, 0.0, 0.9, T_wall, kB, NA
            )
            + 2.0
            * fmf_flat_plate(
                v[n], rho[n, :], MW, T_atm[n], A_face_z, 0.0, 0.9, T_wall, kB, NA
            )
        )

    # Continuum drag (Clift correlation; Sutherland viscosity)
    mu_0 = 1.716e-5  # [kg/m*s]
    T0 = 298.15      # [K]

    drag_C = np.zeros_like(alt)
    for n in range(len(alt)):
        drag_C[n] = prism_drag(mu_0, T_atm[n], T0, rho[n, 5], v[n], A_ref, L_char)

    # Cube Cd in continuum regime (reference area = ram face)
    with np.errstate(invalid="ignore", divide="ignore"):
        Cd_C = drag_C / (0.5 * rho[:, 5] * v**2 * A_ref)
        Cd_FMF = drag_FMF / (0.5 * rho[:, 5] * v**2 * A_ref)

    # Report regime boundary altitudes (where Kn crosses each threshold)
    print(f"Knudsen number (L = {L_char} m): "
          f"Kn={Kn.min():.3g} at {alt.min()/1e3:.0f} km -> "
          f"Kn={Kn.max():.3g} at {alt.max()/1e3:.0f} km")
    for kn_thr in (0.01, 0.1, 10.0):
        if Kn.min() < kn_thr < Kn.max():
            alt_cross = np.interp(kn_thr, np.sort(Kn),
                                  (alt / 1000.0)[np.argsort(Kn)])
            print(f"  Kn = {kn_thr:>5} at {alt_cross:6.1f} km")

    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------
    fig1, ax1 = plt.subplots()
    shade_flow_regimes(ax1, alt / 1000.0, Kn)
    ax1.plot(alt / 1000.0, drag_FMF, label="FMF")
    ax1.plot(alt / 1000.0, drag_C, label="Continuum")

    # Overlay DSMC results from data/ampt_box_log.tsv if present
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(
        os.path.dirname(script_dir), "data", "ampt_box_log.tsv"
    )
    dsmc_alt, dsmc_drag, dsmc_cd = load_ampt_box_log(log_path)
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

    # Linear-scale drag, zoomed to 70-100 km
    fig_lin, ax_lin = plt.subplots()
    shade_flow_regimes(ax_lin, alt / 1000.0, Kn)
    ax_lin.plot(alt / 1000.0, drag_FMF, label="FMF")
    ax_lin.plot(alt / 1000.0, drag_C, label="Continuum")
    if dsmc_alt.size > 0:
        ax_lin.scatter(dsmc_alt, dsmc_drag, marker="*", color="k", label="DSMC")
    ax_lin.set_xlim(70.0, 100.0)
    zoom = (alt / 1000.0 >= 70.0) & (alt / 1000.0 <= 100.0)
    finite_drag = np.concatenate([drag_FMF[zoom], drag_C[zoom]])
    finite_drag = finite_drag[np.isfinite(finite_drag)]
    if finite_drag.size:
        ax_lin.set_ylim(0.0, 1.05 * finite_drag.max())
    ax_lin.set_xlabel("Altitude [km]")
    ax_lin.set_ylabel("Drag [N]")
    ax_lin.grid(True, which="both")
    ax_lin.legend()
    fig_lin.patch.set_facecolor("w")

    fig2, ax2 = plt.subplots()
    alt_km_all = alt / 1000.0
    shade_flow_regimes(ax2, alt_km_all, Kn)
    ax2.plot(alt_km_all, Cd_FMF, label="FMF")
    ax2.plot(alt_km_all, Cd_C, label="Continuum")
    if dsmc_alt.size > 0:
        mask = np.isfinite(dsmc_cd)
        if mask.any():
            ax2.scatter(
                dsmc_alt[mask], dsmc_cd[mask],
                marker="*", color="k", label="DSMC",
            )
    ax2.set_xlabel("Altitude [km]")
    ax2.set_ylabel(r"$C_d$")
    ax2.set_yscale("log")
    ax2.grid(True, which="both")
    ax2.legend()
    fig2.patch.set_facecolor("w")

    # Cd vs Knudsen number (regimes shaded directly on the Kn axis)
    fig3, ax3 = plt.subplots()
    kn_min, kn_max = Kn.min(), Kn.max()
    for label, kn_lo, kn_hi, color in FLOW_REGIMES:
        lo = max(kn_lo, kn_min)
        hi = min(kn_hi, kn_max)
        if hi > lo:
            ax3.axvspan(lo, hi, color=color, alpha=0.18, zorder=0, label=label)
    ax3.plot(Kn, Cd_FMF, label="FMF")
    ax3.plot(Kn, Cd_C, label="Continuum")
    if dsmc_alt.size > 0:
        mask = np.isfinite(dsmc_cd)
        if mask.any():
            order = np.argsort(alt_km_all)
            dsmc_kn = np.interp(dsmc_alt[mask], alt_km_all[order], Kn[order])
            ax3.scatter(dsmc_kn, dsmc_cd[mask],
                        marker="*", color="k", label="DSMC")
    ax3.set_xlabel("Knudsen number")
    ax3.set_ylabel(r"$C_d$")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlim(kn_min, kn_max)
    ax3.grid(True, which="both")
    ax3.legend()
    fig3.patch.set_facecolor("w")

    out_dir = os.path.join(os.path.dirname(script_dir), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    drag_path = os.path.join(out_dir, "Ethan_drag_theory_drag.png")
    drag_log_path = os.path.join(out_dir, "Ethan_drag_theory_drag_log.png")
    cd_path = os.path.join(out_dir, "Ethan_drag_theory_Cd.png")
    cd_kn_path = os.path.join(out_dir, "Ethan_drag_theory_Cd_vs_Kn.png")
    fig_lin.savefig(drag_path, dpi=150, bbox_inches="tight")
    fig1.savefig(drag_log_path, dpi=150, bbox_inches="tight")
    fig2.savefig(cd_path, dpi=150, bbox_inches="tight")
    fig3.savefig(cd_kn_path, dpi=150, bbox_inches="tight")
    print(f"Saved {drag_path}")
    print(f"Saved {drag_log_path}")
    print(f"Saved {cd_path}")
    print(f"Saved {cd_kn_path}")

    if os.environ.get("DISPLAY"):
        plt.show()


if __name__ == "__main__":
    main()
