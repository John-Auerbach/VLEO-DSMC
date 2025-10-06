#!/usr/bin/env python3
"""plot drag vs timestep from dumps/drag.dat

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


def main():
	p = argparse.ArgumentParser()
	p.add_argument('--file', '-f', default='dumps/drag.dat')
	p.add_argument('--out', '-o', default='outputs/drag.png')
	p.add_argument('--csv', help='optional CSV output file')
	p.add_argument('--show', action='store_true')
	args = p.parse_args()

	try:
		t, drag = read_drag(args.file)
	except Exception as e:
		print(f"Error reading drag file: {e}", file=sys.stderr)
		sys.exit(2)

	os.makedirs(os.path.dirname(args.out), exist_ok=True)

	fig, ax = plt.subplots(figsize=(8,4))
	ax.plot(t, drag, '-o', markersize=3)
	ax.set_xlabel('Timestep')
	ax.set_ylabel('Drag (N)')
	ax.set_title('Drag vs Timestep')
	ax.grid(True, alpha=0.3)
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
