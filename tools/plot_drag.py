#!/usr/bin/env python3
"""plot drag vs timestep from dumps/direct_drag.dat

Usage:
  python3 tools/plot_drag.py [--show] [--out png] [--csv out.csv]

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


def read_boundary(path):
	"""
	parses boundary file with blocks: timestep nrows, then nrows lines of 'row val1 val2 val3'.
	extracts pressure at xlo boundary (row=1) and xhi boundary (row=2) from val3 column:

	(example file format where val1 = c_bnd[1] = n, val2 = c_bnd[2] = mflux, val3 = c_bnd[3] = mom flux / pressure)
	# header comments
	12500 6          # timestep nrows
	1 val1 val2 val3  # row=1 (xlo boundary data)
	2 val1 val2 val3  # row=2 (xhi boundary data) 
	3 val1 val2 val3  # row=3 (ylo boundary data)
	4 val1 val2 val3  # row=4 (yhi boundary data)
	5 val1 val2 val3  # row=5 (zlo boundary data)
	6 val1 val2 val3  # row=6 (zhi boundary data)
	12600 6          # next timestep block
	1 val1 val2 val3
	...

	returns (timesteps, pressure_xlo, pressure_xhi) as arrays.
	"""
	f = open(path, 'r')
	lines = [l.strip() for l in f if l.strip()]  # read all non-blank lines
	f.close()
	# skip header lines starting with '#'
	idx = 0
	while lines[idx].startswith('#'):  # find first data line
		idx += 1
	ts = []
	press_xlo = []
	press_xhi = []
	while idx < len(lines):
		parts = lines[idx].split() # 12500 6 -> ["12500", "6"]
		t = float(parts[0])  # timestep
		nrows = int(parts[1])  # number of boundary rows to read
		idx += 1
		# read nrows lines
		p_lo = None
		p_hi = None
		for i in range(nrows):
			parts = lines[idx+i].split()
			row = int(parts[0])  # boundary id (1=xlo, 2=xhi, 3=ylo, 4=yhi, 5=zlo, 6=zhi)
			# values: c_bnd[1], c_bnd[2], c_bnd[3]
			val1 = float(parts[1])  # n
			val2 = float(parts[2])  # mflux
			val3 = float(parts[3])  # press
			if row == 1:  # xlo boundary
				p_lo = val3
			if row == 2:  # xhi boundary
				p_hi = val3
		idx += nrows  # move to next timestep block
		ts.append(t)
		press_xlo.append(p_lo)
		press_xhi.append(p_hi)
	return np.array(ts), np.array(press_xlo), np.array(press_xhi)


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

	# boundary-derived drag
	# find a boundary file
	bfile = 'dumps/boundary_drag.dat'
	tb, p_lo, p_hi = read_boundary(bfile) # timesteps, pressure_xlo, pressure_xhi
	area = read_in_ampt_area('in.ampt') # cross-sectional area in m^2
	bdrag = (p_hi - p_lo) * area
	# interpolate boundary drag onto direct-drag timesteps
	bdrag_interp = np.interp(t, tb, bdrag)

	os.makedirs(os.path.dirname(args.out), exist_ok=True)

	fig, ax = plt.subplots(figsize=(8,4))
	ax.plot(t, drag, '-o', markersize=3, label='direct surf-sum')
	ax.plot(t, bdrag_interp, '-s', markersize=3, label='boundary-derived')
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
