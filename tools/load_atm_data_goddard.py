#!/usr/bin/env python3
import numpy as np
import os
import sys
import math
import argparse
import datetime

msis_version = 2.1  # NRLMSIS-2.1

# shared location / time used by BOTH NRLMSIS (neutrals) and IRI (ions)
# Matches the SPIS IRI pull (scripts/iri_ion_current_table.py): NASA GSFC.
# GSFC coordinates: 38deg59'N 76deg51'W.
LAT = 38 + 59 / 60.0           # 38.983 N
LON = -(76 + 51 / 60.0)        # -76.85 (76deg51'W); IRI uses 283.15 E
UTC = datetime.datetime(2026, 6, 22, 17, 0)

# command-line interface
parser = argparse.ArgumentParser(
    description="Generate a SPARTA atmosphere include file from NRLMSIS (neutrals) "
                "and, with --ions, ionospheric ions (O+, O2+, NO+, N+) from IRI. "
                "GSFC location/time variant matching the SPIS IRI settings.")
parser.add_argument("altitude", nargs="?", type=float, default=85.0,
                    help="altitude in km (default: 85)")
parser.add_argument("--ions", action="store_true",
                    help="include IRI ions for ambipolar DSMC (adds ion species + electron 'e')")
args = parser.parse_args()

alt = args.altitude
include_ions = args.ions

print(f"Using altitude: {alt} km")
if include_ions:
    print("Ion inclusion: ENABLED (IRI model)")

# determine correct paths based on script location
script_dir = os.path.dirname(os.path.abspath(__file__))
ampt_dir = os.path.dirname(script_dir)  # Parent of tools directory
data_dir = os.path.join(ampt_dir, 'data')

# check if NRLMSIS data file exists, if not create it
# (Goddard-specific cache so it never reuses the State College nrlmsis.dat.)
data_file = os.path.join(data_dir, 'nrlmsis_goddard.dat')
if not os.path.exists(data_file):
    print(f"NRLMSIS data file not found. Generating fresh data...")
    
    try:
        import pymsis
        import scipy.constants as const
        import matplotlib.pyplot as plt
        
        # constants
        R_E = 6378e3  # m
        mu_E = 3.986e14 # m^3/s^2
        k_B = 1.380649e-23 # J/K
        R = 287.05 # specific gas constant, air (J / kg*K)
        
        lat = LAT
        lon = LON
        altitudes = np.linspace(70e3, 300e3, 501)  # m
        alt_km = altitudes * 1e-3  # km
        
        # load NRLMSIS Data
        utc = UTC
        times = np.array([utc], dtype='datetime64')
        
        # Solar/geomagnetic parameters (solar max conditions)
        f107a = 250  # 81-day average F10.7 flux (solar max ~250, solar min ~70)
        f107 = 250   # daily F10.7 flux for previous day
        aps = [[4, 4, 4, 4, 4, 4, 4]]  # Ap indices: [daily, 0h, 3h, 6h, 9h, 12-33h avg, 36-57h avg] (quiet)
        
        #atmosphere = pymsis.calculate(times, [lon], [lat], alt_km,
        #                              f107, f107a, aps,
        #                              version=msis_version)

        # For real historical conditions, drop f107/f107a/aps (auto-fetched for utc). Comment out above, use this instead:
        atmosphere = pymsis.calculate(times, [lon], [lat], alt_km, version=msis_version)

        T = np.squeeze(atmosphere[..., pymsis.Variable.TEMPERATURE])
        rho = np.squeeze(atmosphere[..., pymsis.Variable.MASS_DENSITY])
        P = rho * R * T
        P_torr = P / 133.322
        v = np.sqrt(mu_E / (R_E + altitudes))
        N = P / (k_B * T)

        # species number densities
        n_N2 = np.squeeze(atmosphere[..., pymsis.Variable.N2])
        n_O2 = np.squeeze(atmosphere[..., pymsis.Variable.O2])
        n_O  = np.squeeze(atmosphere[..., pymsis.Variable.O])
        n_He = np.squeeze(atmosphere[..., pymsis.Variable.HE])
        n_Ar = np.squeeze(atmosphere[..., pymsis.Variable.AR])
        n_N  = np.squeeze(atmosphere[..., pymsis.Variable.N])

        plt.figure(figsize=(8, 5))
        plt.plot(alt_km, rho)
        plt.xlabel('Altitude (km)')
        plt.ylabel('Density (kg/m³)')
        plt.title('Density vs Altitude')
        plt.grid(True)
        
        # save data to file
        data = np.column_stack((alt_km, T, rho, P, P_torr, v, N, n_N2, n_O2, n_O, n_He, n_Ar, n_N))
        header = "Altitude_km Temperature_K Density_kg_m3 Pressure_Pa Pressure_Torr Orbital_Velocity_m_s Number_Density_per_m3 n_N2 n_O2 n_O n_He n_Ar n_N"
        
        os.makedirs(data_dir, exist_ok=True)
        np.savetxt(data_file, data, header=header, fmt='%.6e', delimiter=' ')
        print(f"Generated NRLMSIS data and saved to {data_file}")
        
    except ImportError:
        print("ERROR: pymsis not available. Using existing data file if present, or create data manually.")
        if not os.path.exists(data_file):
            print(f"ERROR: No NRLMSIS data file found at {data_file}")
            exit(1)

# load NRLMSIS data
d = np.loadtxt(data_file)

# interpolate values for specified altitude
os.makedirs(data_dir, exist_ok=True)
rho_val = np.interp(alt, d[:,0], d[:,2])
nrho_val = np.interp(alt, d[:,0], d[:,6])
T_val = np.interp(alt, d[:,0], d[:,1])
vx_val = np.interp(alt, d[:,0], d[:,5])

# species fractions
# columns: 7=n_N2, 8=n_O2, 9=n_O, 10=n_He, 11=n_Ar, 12=n_N
n_N2 = np.interp(alt, d[:,0], d[:,7])
n_O2 = np.interp(alt, d[:,0], d[:,8])
n_O  = np.interp(alt, d[:,0], d[:,9])
n_He = np.interp(alt, d[:,0], d[:,10])
n_Ar = np.interp(alt, d[:,0], d[:,11])
n_N  = np.interp(alt, d[:,0], d[:,12])

# replace nan with 0 (pymsis returns nan for some species at low altitudes)
n_N2 = np.nan_to_num(n_N2)
n_O2 = np.nan_to_num(n_O2)
n_O  = np.nan_to_num(n_O)
n_He = np.nan_to_num(n_He)
n_Ar = np.nan_to_num(n_Ar)
n_N  = np.nan_to_num(n_N)

# ============================================================================
# IONS (optional) - International Reference Ionosphere via the iri2020 package
# ============================================================================
n_Op = n_O2p = n_NOp = n_Np = 0.0
ne_val = Ti_val = Te_val = 0.0
if include_ions:
    # iri2020 builds a Fortran library on first use and needs CMake on PATH.
    # If no system cmake, fall back to the pip 'cmake' wheel's bundled binary.
    import shutil
    if shutil.which("cmake") is None:
        try:
            import cmake as _cmake_pkg
            _cmbin = os.path.join(os.path.dirname(_cmake_pkg.__file__), "data", "bin")
            if os.path.isdir(_cmbin):
                os.environ["PATH"] = _cmbin + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass

    iri_file = os.path.join(data_dir, 'iri_goddard.dat')
    if not os.path.exists(iri_file):
        print("IRI data file not found. Generating fresh data "
              "(IRI Fortran build may take ~1 min on first run)...")
        try:
            import iri2020
        except ImportError:
            print("ERROR: --ions requested but the 'iri2020' package is not installed.")
            print("       Install it in the active environment:")
            print("           python -m pip install git+https://github.com/space-physics/iri2020 cmake")
            sys.exit(1)

        iono = iri2020.IRI(UTC, [70, 300, 1], LAT, LON)
        alt_iri = np.asarray(iono.coords['alt_km'].values, dtype=float)

        def _clean(name, clip_neg=True):
            a = np.asarray(iono[name].values, dtype=float)
            a = np.nan_to_num(a, nan=0.0)
            if clip_neg:
                a[a < 0.0] = 0.0  # IRI uses negative sentinels for "not computed"
            return a

        ne_i   = _clean('ne')
        Ti_i   = _clean('Ti', clip_neg=False)
        Te_i   = _clean('Te', clip_neg=False)
        nOp_i  = _clean('nO+')
        nO2p_i = _clean('nO2+')
        nNOp_i = _clean('nNO+')
        nNp_i  = _clean('nN+')

        out = np.column_stack((alt_iri, ne_i, Ti_i, Te_i, nOp_i, nO2p_i, nNOp_i, nNp_i))
        hdr = ("IRI ion data (%s UTC, lat=%.3f lon=%.3f)\n"
               "Altitude_km ne Ti Te nO+ nO2+ nNO+ nN+") % (UTC.isoformat(), LAT, LON)
        np.savetxt(iri_file, out, header=hdr, fmt='%.6e', delimiter=' ')
        print(f"Generated IRI data and saved to {iri_file}")

    di = np.loadtxt(iri_file)
    # columns: 0=alt 1=ne 2=Ti 3=Te 4=nO+ 5=nO2+ 6=nNO+ 7=nN+
    ne_val = max(float(np.interp(alt, di[:, 0], di[:, 1])), 0.0)
    Ti_val = float(np.interp(alt, di[:, 0], di[:, 2]))
    Te_val = float(np.interp(alt, di[:, 0], di[:, 3]))
    n_Op   = max(float(np.interp(alt, di[:, 0], di[:, 4])), 0.0)
    n_O2p  = max(float(np.interp(alt, di[:, 0], di[:, 5])), 0.0)
    n_NOp  = max(float(np.interp(alt, di[:, 0], di[:, 6])), 0.0)
    n_Np   = max(float(np.interp(alt, di[:, 0], di[:, 7])), 0.0)

# ----------------------------------------------------------------------------
# assemble species lists and number-density-weighted mixture fractions
# ----------------------------------------------------------------------------
neutral_species = [('N2', n_N2), ('O2', n_O2), ('O', n_O),
                   ('He', n_He), ('Ar', n_Ar), ('N', n_N)]
ion_species = []
if include_ions:
    ion_species = [('O+', n_Op), ('O2+', n_O2p), ('NO+', n_NOp), ('N+', n_Np)]

n_neutral_total = sum(n for _, n in neutral_species)
n_ion_total = sum(n for _, n in ion_species)
n_total = n_neutral_total + n_ion_total

# species command line
neutral_names = [s for s, _ in neutral_species]
if include_ions:
    # N2+ is defined (referenced by 'fix ambipolar') even though IRI provides none
    species_line = ("species         species/air.species "
                    + " ".join(neutral_names + ['O+', 'O2+', 'NO+', 'N+', 'N2+', 'e']))
else:
    species_line = "species         species/air.species " + " ".join(neutral_names)

# write SPARTA include file with all atmospheric data
out_path = os.path.join(data_dir, 'atm_goddard.sparta')
with open(out_path, 'w') as f:
    f.write(f"# NRLMSIS-{msis_version} atmospheric data for {alt} km altitude\n")
    f.write(f"# GSFC location {LAT:.3f} N, {LON:.3f} E-equiv; {UTC.isoformat()} UTC\n")
    if include_ions:
        f.write("# Ions from IRI (O+, O2+, NO+, N+; N2+ defined but ~0). "
                "Electron 'e' defined for ambipolar.\n")
    f.write(f"# Generated by load_atm_data_goddard.py\n\n")
    f.write(f"variable        rho  equal {rho_val:.12e}\n")
    f.write(f"variable        nrho equal {n_total:.6e}\n")
    f.write(f"variable        T    equal {T_val:.6f}\n")
    f.write(f"variable        vx   equal {vx_val:.1f}\n")
    if include_ions:
        f.write(f"variable        Ti   equal {Ti_val:.6f}   # IRI ion temperature (K)\n")
        f.write(f"variable        Te   equal {Te_val:.6f}   # IRI electron temperature (K)\n")
        f.write(f"variable        ne   equal {ne_val:.6e}   # IRI electron density (m^-3)\n")
    f.write("\n")
    f.write(species_line + "\n")

    # Write a frac for every species. The densest species is the "filler": it
    # is listed with no explicit frac, so SPARTA assigns it the remaining
    # fraction and the mixture closes to exactly 1.0 (SPARTA errors if the
    # explicit fracs sum to > 1.0). High precision keeps tiny ion fractions
    # from truncating to zero.
    all_species = neutral_species + ion_species
    filler = max(all_species, key=lambda kv: kv[1])[0]
    for name, dens in all_species:
        if name == filler:
            f.write(f"mixture         atm {name}\n")
        else:
            fr = math.floor((dens / n_total) * 1e10) / 1e10
            f.write(f"mixture         atm {name} frac {fr:.10f}\n")

    if include_ions:
        f.write("\n")
        f.write("# --- ambipolar setup reminder (add to the input script AFTER the\n")
        f.write("#     'mixture atm nrho ... vstream ... temp ...' line): ---\n")
        f.write("#   fix ambi ambipolar e O+ N+ O2+ N2+ NO+\n")
        f.write("#   mixture atm copy noelectron\n")
        f.write("#   mixture noelectron delete e\n")
        f.write("#   collide_modify ambipolar yes\n")
        f.write("#   (use 'noelectron' in create_particles and fix emit/face)\n")

# console summary
print(f'Loaded atmospheric data for {alt} km altitude')
print(f'  rho: {rho_val:.3e} kg/m^3')
if include_ions:
    print(f'  nrho: {n_total:.3e} m^-3  (neutrals + ions)')
else:
    print(f'  nrho: {n_total:.3e} m^-3  (sum of species)')
print(f'  T: {T_val:.1f} K')
print(f'  vx: {vx_val:.1f} m/s')
neutral_frac_str = "  ".join(f"{s}: {n / n_total:.4f}" for s, n in neutral_species)
print(f'  neutrals -> {neutral_frac_str}')
if include_ions:
    ion_frac_str = "  ".join(f"{s}: {n / n_total:.3e}" for s, n in ion_species)
    print(f'  ions     -> {ion_frac_str}')
    print(f'  ion fraction (n_ion/n_total): {n_ion_total / n_total:.3e}')
    print(f'  Ti: {Ti_val:.1f} K   Te: {Te_val:.1f} K   ne(IRI): {ne_val:.3e} m^-3')
