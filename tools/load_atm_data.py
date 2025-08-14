#!/usr/bin/env python3
import numpy as np
import os
import sys

# Get altitude from command line or use default
if len(sys.argv) > 1:
    alt = float(sys.argv[1])
else:
    alt = 150.0  # Default altitude

print(f"Using altitude: {alt} km")

# Determine correct paths based on script location
script_dir = os.path.dirname(os.path.abspath(__file__))
ampt_dir = os.path.dirname(script_dir)  # Parent of tools directory
data_dir = os.path.join(ampt_dir, 'data')

# Check if NRLMSIS data file exists, if not create it
data_file = os.path.join(data_dir, 'nrlmsis.dat')
if not os.path.exists(data_file):
    print(f"NRLMSIS data file not found. Generating fresh data...")
    
    try:
        import pymsis
        import datetime
        import scipy.constants as const
        
        # Constants
        R_E = 6378e3  # m
        mu_E = 3.986e14 # m^3/s^2
        k_B = 1.380649e-23 # J/K
        R = 287.05 # specific gas constant, air (J / kg*K)
        
        lat = 0
        lon = 0
        altitudes = np.linspace(70, 300, 100) * 1e3 # m
        alt_km = altitudes * 1e-3 # km
        
        # Load NRLMSIS Data
        utc = datetime.datetime(2020, 1, 1, 12, 0)
        times = np.array([utc], dtype='datetime64')
        
        atmosphere = pymsis.calculate(times, [lon], [lat], alt_km)
        
        T = np.squeeze(atmosphere[..., pymsis.Variable.TEMPERATURE])
        rho = np.squeeze(atmosphere[..., pymsis.Variable.MASS_DENSITY])
        P = rho * R * T
        P_torr = P / 133.322
        v = np.sqrt(mu_E / (R_E + altitudes))
        N = P / (k_B * T)
        
        # Save data to file
        data = np.column_stack((alt_km, T, rho, P, P_torr, v, N))
        header = "Altitude_km Temperature_K Density_kg_m3 Pressure_Pa Pressure_Torr Orbital_Velocity_m_s Number_Density_per_m3"
        
        os.makedirs(data_dir, exist_ok=True)
        np.savetxt(data_file, data, header=header, fmt='%.6e', delimiter=' ')
        print(f"Generated NRLMSIS data and saved to {data_file}")
        
    except ImportError:
        print("ERROR: pymsis not available. Using existing data file if present, or create data manually.")
        if not os.path.exists(data_file):
            print(f"ERROR: No NRLMSIS data file found at {data_file}")
            exit(1)

# Load NRLMSIS data
d = np.loadtxt(data_file)

# Interpolate and write to files (in data directory)
os.makedirs(data_dir, exist_ok=True)
open(os.path.join(data_dir, 'rho.dat'),'w').write(f'{np.interp(alt,d[:,0],d[:,2]):.12f}')
open(os.path.join(data_dir, 'nrho.dat'),'w').write(f'{np.interp(alt,d[:,0],d[:,6]):.0f}')
open(os.path.join(data_dir, 'T.dat'),'w').write(f'{np.interp(alt,d[:,0],d[:,1]):.6f}')
open(os.path.join(data_dir, 'vx.dat'),'w').write(f'{np.interp(alt,d[:,0],d[:,5]):.1f}')

print(f'Loaded atmospheric data for {alt} km altitude')
rho_val = np.interp(alt,d[:,0],d[:,2])
nrho_val = np.interp(alt,d[:,0],d[:,6])
T_val = np.interp(alt,d[:,0],d[:,1])
vx_val = np.interp(alt,d[:,0],d[:,5])
print(f'  rho: {rho_val:.3e} kg/m³')
print(f'  nrho: {nrho_val:.3e} m⁻³')
print(f'  T: {T_val:.1f} K')
print(f'  vx: {vx_val:.1f} m/s')
