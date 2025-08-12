import numpy as np
import datetime
import math
import scipy.constants as const
from scipy.integrate import quad

try:
    import pymsis
except ImportError:
    print("Error: pymsis module not found.")
    print("Please install it using: pip install pymsis")
    print("or: conda install -c conda-forge pymsis")
    exit(1)

# Constants
R_E = 6378e3  # m
mu_E = 3.986e14 # m^3/s^2
k_B = 1.380649e-23 # J/K
R = 287.05 # specific gas constant, air (J / kg*K)

[i for i in pymsis.Variable]

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

np.savetxt('/home/scien/AMPT/data/nrlmsis.dat', data, header=header, fmt='%.6e', delimiter=' ')

print("Atmospheric data saved to /home/scien/AMPT/data/nrlmsis.dat")