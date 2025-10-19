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
	reads dumps/direct_drag.dat: timestep and c_drag columns
	c_drag is running-averaged total x-force on ampt surface group in newtons
	returns (timesteps, drag_forces) as float arrays
	"""
	if not os.path.exists(path):
		raise FileNotFoundError(f"Drag file not found: {path}")
	data = np.loadtxt(path)
	if data.ndim == 1:
		# single line
		data = data.reshape((1, -1))
	if data.shape[1] < 2:
		raise ValueError("Expected at least two columns: timestep and drag")
	return data[:, 0], data[:, 1]


def read_flux_file(path):
	"""
	reads a flux data file with timestep, nflux, and momentum flux columns
	returns (timesteps, nflux, momentum_flux) as float arrays
	"""
	if not os.path.exists(path):
		raise FileNotFoundError(f"Flux file not found: {path}")
	data = np.loadtxt(path)
	if data.ndim == 1:
		# single line
		data = data.reshape((1, -1))
	if data.shape[1] < 3:
		raise ValueError("Expected at least three columns: timestep, nflux, and momentum flux")
	return data[:, 0], data[:, 1], data[:, 2]


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

	t, drag = read_drag(args.file) # timesteps, drag forces

	# boundary-derived drag from separate pressure files
	xlo_file = 'dumps/xlo_pressure.dat'
	xhi_file = 'dumps/xhi_pressure.dat'
	
	# read pressure data from xlo and xhi boundary surfaces
	t_xlo, p_lo = read_pressure_file(xlo_file)  # timesteps, pressure at xlo
	t_xhi, p_hi = read_pressure_file(xhi_file)  # timesteps, pressure at xhi
	
	area = read_in_ampt_area('in.ampt') # cross-sectional area in m^2
	bdrag = (p_lo - p_hi) * area

	print(f"Cross-sectional area (Ly*Lz): {area} m^2")

	os.makedirs(os.path.dirname(args.out), exist_ok=True)

	fig, ax = plt.subplots(figsize=(8,4))
	ax.plot(t, drag, '-o', markersize=3, label='direct surf-sum')
	ax.plot(t, bdrag, '-s', markersize=3, label='boundary-derived')
	ax.set_xlabel('Timestep')
	ax.set_ylabel('Drag (N)')
	ax.set_title('Drag vs Timestep')
	ax.grid(True, alpha=0.3)
	ax.legend()
	fig.tight_layout()
	fig.savefig(args.out, dpi=150)
	print(f"Saved plot to {args.out}")

	if args.csv:
		np.savetxt(args.csv, np.column_stack((t, drag)), header='timestep,drag', delimiter=',', comments='')
		print(f"Saved CSV to {args.csv}")

	if args.show:
		# show interactively
		matplotlib.use('TkAgg')
		plt.show()


if __name__ == '__main__':
	main()
