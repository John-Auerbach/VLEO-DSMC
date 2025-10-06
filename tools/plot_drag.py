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
	# parse boundary file blocks of the form:
	# <timestep> <nrows>
	# row val1 val2 val3
	# ... repeated
	f = open(path, 'r')
	lines = [l.strip() for l in f if l.strip()]
	f.close()
	# skip header lines starting with '#'
	idx = 0
	while lines[idx].startswith('#'):
		idx += 1
	ts = []
	press_xlo = []
	press_xhi = []
	while idx < len(lines):
		parts = lines[idx].split()
		t = float(parts[0])
		nrows = int(parts[1])
		idx += 1
		# read nrows lines
		p_lo = None
		p_hi = None
		for i in range(nrows):
			parts = lines[idx+i].split()
			row = int(parts[0])
			# values: c_bnd[1], c_bnd[2], c_bnd[3]
			val1 = float(parts[1])
			val2 = float(parts[2])
			val3 = float(parts[3])
			if row == 1:
				p_lo = val3
			if row == 2:
				p_hi = val3
		idx += nrows
		ts.append(t)
		press_xlo.append(p_lo)
		press_xhi.append(p_hi)
	return np.array(ts), np.array(press_xlo), np.array(press_xhi)


def read_in_ampt_area(path='in.ampt'):
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

	t, drag = read_drag(args.file)

	# boundary-derived drag
	# find a boundary file; prefer dumps/boundary_drag.* (may be multiple), pick the first
	import glob
	bfiles = sorted(glob.glob('dumps/boundary_drag.*'))
	if len(bfiles) == 0:
		bfiles = sorted(glob.glob('dumps/boundary_running.*'))
	bfile = bfiles[0]
	tb, p_lo, p_hi = read_boundary(bfile)
	area = read_in_ampt_area('in.ampt')
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
