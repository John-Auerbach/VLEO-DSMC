#!/usr/bin/env python3
"""plot drag vs timestep from dumps/direct_drag.dat

Usage:
  python3 scripts/plot_drag.py [--show] [--out png] [--csv out.csv]

"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_drag(path):
	"""
	reads dumps/direct_drag.dat: timestep, total_drag, ram_drag, skin_friction columns
	c_drag is running-averaged total x-force on ampt surface group in newtons
	returns (timesteps, total_drag, ram_drag, skin_friction) as float arrays
	"""
	if not os.path.exists(path):
		raise FileNotFoundError(f"Drag file not found: {path}")
	data = np.loadtxt(path)
	if data.ndim == 1:
		# single line
		data = data.reshape((1, -1))
	if data.shape[1] < 2:
		raise ValueError("Expected at least two columns: timestep and drag")
	
	timesteps = data[:, 0]
	total_drag = data[:, 1]
	ram_drag = data[:, 2] if data.shape[1] > 2 else None
	skin_friction = data[:, 3] if data.shape[1] > 3 else None
	
	return timesteps, total_drag, ram_drag, skin_friction


def read_flux_file(path):
	"""
	Read a plane flux file with timestep, M_dot (kg/s), E_dot (W), N_dot (1/s).
	Also derive an effective normal speed v_eff = sqrt(2 * E_dot / M_dot).
	Returns: timesteps, M_dot, E_dot, v_eff, N_dot.
	"""
	if not os.path.exists(path):
		raise FileNotFoundError(f"Pressure file not found: {path}")
	data = np.loadtxt(path)
	if data.ndim == 1:
		# single line
		data = data.reshape((1, -1))
	if data.shape[1] < 3:
		raise ValueError("Expected at least three columns: timestep, M_dot, E_dot")
	
	timesteps = data[:, 0]
	m_dot = data[:, 1]  # total mass flow (kg/s)
	e_dot = data[:, 2]  # total KE flow (W)
	n_dot = data[:, 3]  # total number flow (1/s)
	
	# Debug: print flux values
	print(f"\n{path}:")
	print(f"  M_dot mean: {np.mean(m_dot):.6e} kg/s")
	print(f"  E_dot mean: {np.mean(e_dot):.6e} W")
	print(f"  N_dot mean: {np.mean(n_dot):.6e} 1/s")

	# Calculate velocity: v = sqrt(2*Φ_KE / Φ_m)
	# Avoid division by zero
	with np.errstate(divide='ignore', invalid='ignore'):
		velocity = np.sqrt(2.0 * e_dot / m_dot)

	return timesteps, m_dot, e_dot, velocity, n_dot


def read_in_ampt_area(path='in.ampt'):
	"""
	parses variable definitions from in.ampt file to extract ymin/ymax/zmin/zmax
	computes cross-sectional area Ly*Lz normal to x-direction for pressure-to-force conversion
	returns area in square meters
	"""
	# read xmin/xmax/ymin/ymax/zmin/zmax from in.ampt and compute Ly*Lz area
	txt = open(path).read().splitlines()
	vals = {}
	for line in txt:
		line = line.strip()
		if line.startswith('variable'):
			parts = line.split()
			# expecting: variable name equal value
			if len(parts) >= 4 and parts[2] == 'equal':
				name = parts[1]
				expr = parts[3]
				try:
					vals[name] = float(expr)
				except:
					# ignore non-numeric
					pass
	Ly = vals['ymax'] - vals['ymin']
	Lz = vals['zmax'] - vals['zmin']
	return Ly * Lz


def main():
	p = argparse.ArgumentParser()
	p.add_argument('--file', '-f', default='dumps/direct_drag.dat')
	p.add_argument('--out', '-o', default='outputs/drag.png')
	p.add_argument('--csv', help='optional CSV output file')
	p.add_argument('--show', action='store_true')
	args = p.parse_args()

	t, drag, ram_drag, skin_friction = read_drag(args.file) # timesteps, drag forces

	# boundary-derived drag from separate pressure files
	xlo_file = 'dumps/xlo_flux.dat'
	xhi_file = 'dumps/xhi_flux.dat'
	
	# read total flow data from xlo and xhi boundary surfaces
	t_xlo, m_lo_dot, e_lo_dot, v_lo, n_lo_dot = read_flux_file(xlo_file)
	t_xhi, m_hi_dot, e_hi_dot, v_hi, n_hi_dot = read_flux_file(xhi_file)
	
	area = read_in_ampt_area('in.ampt') # cross-sectional area in m^2
	# convert totals to per-area fluxes
	m_lo = m_lo_dot / area
	m_hi = m_hi_dot / area
	n_lo = n_lo_dot / area
	n_hi = n_hi_dot / area

	# Momentum-flux-based boundary drag: F ≈ (Π_lo - Π_hi) * A, where Π = Φ_m * v
	pi_lo = m_lo * v_lo
	pi_hi = m_hi * v_hi
	bdrag = (pi_lo - pi_hi) * area

	# Compute average rho and vx at each plane from fluxes (Φ_m = ρ v)
	mask_lo = np.isfinite(m_lo) & np.isfinite(v_lo) & (m_lo > 0) & (v_lo > 0)
	mask_hi = np.isfinite(m_hi) & np.isfinite(v_hi) & (m_hi > 0) & (v_hi > 0)

	if np.any(mask_lo):
		vx_lo_avg = float(np.mean(v_lo[mask_lo]))
		rho_lo_avg = float(np.mean(m_lo[mask_lo] / v_lo[mask_lo]))
		nrho_lo_avg = float(np.mean(n_lo[mask_lo] / v_lo[mask_lo]))
	else:
		vx_lo_avg = float('nan')
		rho_lo_avg = float('nan')
		nrho_lo_avg = float('nan')

	if np.any(mask_hi):
		vx_hi_avg = float(np.mean(v_hi[mask_hi]))
		rho_hi_avg = float(np.mean(m_hi[mask_hi] / v_hi[mask_hi]))
		nrho_hi_avg = float(np.mean(n_hi[mask_hi] / v_hi[mask_hi]))
	else:
		vx_hi_avg = float('nan')
		rho_hi_avg = float('nan')
		nrho_hi_avg = float('nan')

	# Calculate average velocities (excluding NaN/inf values from zero flux)
	v_lo_valid = v_lo[np.isfinite(v_lo)]
	v_hi_valid = v_hi[np.isfinite(v_hi)]
	
	v_lo_avg = np.mean(v_lo_valid) if len(v_lo_valid) > 0 else 0.0
	v_hi_avg = np.mean(v_hi_valid) if len(v_hi_valid) > 0 else 0.0
	v_total_avg = (v_lo_avg + v_hi_avg) / 2.0

	print(f"Cross-sectional area (Ly*Lz): {area} m^2")
	print("xlo boundary averages:")
	print(f"  rho: {rho_lo_avg:.3e} kg/m³")
	print(f"  vx: {vx_lo_avg:.1f} m/s")
	print(f"  nrho: {nrho_lo_avg:.3e} 1/m³")
	print("xhi boundary averages:")
	print(f"  rho: {rho_hi_avg:.3e} kg/m³")
	print(f"  vx: {vx_hi_avg:.1f} m/s")
	print(f"  nrho: {nrho_hi_avg:.3e} 1/m³")

	os.makedirs(os.path.dirname(args.out), exist_ok=True)

	fig, ax = plt.subplots(figsize=(10,6))
	ax.plot(t, drag, '-o', markersize=3, label='Total Direct Drag', linewidth=2)
	if ram_drag is not None:
		ax.plot(t, ram_drag, '-^', markersize=3, label='Ram Drag (direct)', alpha=0.7)
	if skin_friction is not None:
		ax.plot(t, skin_friction, '-v', markersize=3, label='Skin Friction (direct)', alpha=0.7)
	ax.plot(t, bdrag, '-s', markersize=3, label='Boundary Momentum-Flux', alpha=0.7)
	ax.set_xlabel('Timestep')
	ax.set_ylabel('Drag (N)')
	ax.set_title('Drag vs Timestep')
	ax.grid(True, alpha=0.3)
	ax.legend()
	fig.tight_layout()
	fig.savefig(args.out, dpi=150)
	print(f"Saved plot to {args.out}")

	if args.csv:
		csv_data = [t, drag]
		csv_header = 'timestep,total_drag'
		if ram_drag is not None:
			csv_data.append(ram_drag)
			csv_header += ',ram_drag'
		if skin_friction is not None:
			csv_data.append(skin_friction)
			csv_header += ',skin_friction'
		csv_data.append(bdrag)
		csv_header += ',boundary_drag'
		np.savetxt(args.csv, np.column_stack(csv_data), header=csv_header, delimiter=',', comments='')
		print(f"Saved CSV to {args.csv}")

	if args.show:
		# show interactively
		matplotlib.use('TkAgg')
		plt.show()


if __name__ == '__main__':
	main()
