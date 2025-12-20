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

def main():
	p = argparse.ArgumentParser()
	p.add_argument('--file', '-f', default='dumps/direct_drag.dat')
	p.add_argument('--out', '-o', default='outputs/drag.png')
	p.add_argument('--csv', help='optional CSV output file')
	p.add_argument('--show', action='store_true')
	p.add_argument('dir', nargs='?', help='directory containing dump files (optional positional argument)')
	args = p.parse_args()

	# If directory is provided, use it to construct file path
	if args.dir:
		dump_dir = args.dir.rstrip('/')
		args.file = os.path.join(dump_dir, 'direct_drag.dat')

	t, drag, ram_drag, skin_friction = read_drag(args.file) # timesteps, drag forces

	os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)

	fig, ax = plt.subplots(figsize=(10,6))
	ax.plot(t, drag, '-o', markersize=3, label='Total Direct Drag', linewidth=2)
	if ram_drag is not None:
		ax.plot(t, ram_drag, '-^', markersize=3, label='Ram Drag (direct)', alpha=0.7)
	if skin_friction is not None:
		ax.plot(t, skin_friction, '-v', markersize=3, label='Skin Friction (direct)', alpha=0.7)
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
		np.savetxt(args.csv, np.column_stack(csv_data), header=csv_header, delimiter=',', comments='')
		print(f"Saved CSV to {args.csv}")

	if args.show:
		# show interactively
		matplotlib.use('TkAgg')
		plt.show()


if __name__ == '__main__':
	main()
