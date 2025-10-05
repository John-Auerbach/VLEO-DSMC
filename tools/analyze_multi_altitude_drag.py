#!/usr/bin/env python3
"""
Enhanced multi-altitude analysis including drag forces and surface temperatures.
Creates comprehensive plots and exports data for temperature and drag vs altitude.

Usage: python3 tools/analyze_multi_altitude_drag.py
"""

import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def extract_tstep_from_input(path="in.ampt"):
    """Extract timestep from SPARTA input file"""
    with open(path, "r") as f:
        for line in f:
            m = re.match(r"variable\s+tstep\s+equal\s+([eE\d.+-]+)", line)
            if m:
                return float(m.group(1))
    raise ValueError("tstep not found in input")

def load_surface_data(surf_file):
    """Load surface temperature data from a surf file"""
    with open(surf_file) as f:
        lines = f.readlines()
    
    # get timestep from header
    step = int(lines[1].strip())
    
    # find data section
    for i, line in enumerate(lines):
        if 'ITEM: SURFS' in line:
            data_start = i + 1
            break
    
    # load surface temperatures
    data = np.loadtxt(lines[data_start:])
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    # last column is s_Tsurf
    temps = data[:, -1]
    return step, temps

def load_drag_data(drag_file):
    """Load drag analysis results from npz file"""
    try:
        data = np.load(drag_file)
        return {
            'timesteps': data['timesteps'],
            'times': data['times'],
            'drag_force': data['drag_force'],
            'mean_drag': float(data['mean_drag']),
            'std_drag': float(data['std_drag']),
            'final_drag': float(data['final_drag']),
            'inlet_momentum': data['inlet_momentum'],
            'outlet_momentum': data['outlet_momentum']
        }
    except Exception as e:
        print(f"Warning: Could not load drag data from {drag_file}: {e}")
        return None

def analyze_altitude_folder(alt_folder):
    """Analyze a single altitude folder for temperature and drag data"""
    results = {
        'altitude': None,
        'surface_temps': [],
        'drag_data': None,
        'timesteps': [],
        'times': []
    }
    
    # Extract altitude from folder name
    alt_match = re.search(r'alt_(\d+)km', alt_folder)
    if alt_match:
        results['altitude'] = int(alt_match.group(1))
    
    # Load surface temperature data
    surf_files = glob.glob(os.path.join(alt_folder, 'surf.*.dat'))
    surf_files.sort(key=lambda x: int(os.path.basename(x).split('.')[1]))
    
    timestep = extract_tstep_from_input()
    
    for surf_file in surf_files:
        try:
            step, temps = load_surface_data(surf_file)
            time = step * timestep
            results['timesteps'].append(step)
            results['times'].append(time)
            results['surface_temps'].append(np.mean(temps))  # Average surface temp
        except Exception as e:
            print(f"Warning: Error loading {surf_file}: {e}")
    
    # Convert to arrays
    results['timesteps'] = np.array(results['timesteps'])
    results['times'] = np.array(results['times'])
    results['surface_temps'] = np.array(results['surface_temps'])
    
    # Load drag data
    drag_file = os.path.join(alt_folder, 'drag_analysis.npz')
    if os.path.exists(drag_file):
        results['drag_data'] = load_drag_data(drag_file)
    
    return results

def main():
    print("=== Enhanced Multi-Altitude Analysis ===")
    
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    # Find all altitude folders
    alt_folders = glob.glob('dumps/alt_*km')
    alt_folders.sort(key=lambda x: int(re.search(r'alt_(\d+)km', x).group(1)))
    
    if not alt_folders:
        print("No altitude folders found. Run multi_altitude.py first.")
        return
    
    print(f"Found {len(alt_folders)} altitude folders")
    
    # Analyze each altitude
    all_results = []
    for alt_folder in alt_folders:
        print(f"Analyzing {alt_folder}...")
        results = analyze_altitude_folder(alt_folder)
        if results['altitude'] is not None:
            all_results.append(results)
    
    if not all_results:
        print("No valid results found")
        return
    
    # Sort by altitude
    all_results.sort(key=lambda x: x['altitude'])
    altitudes = [r['altitude'] for r in all_results]
    
    print(f"Processed altitudes: {altitudes} km")
    
    # === CREATE COMPREHENSIVE PLOTS ===
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Temperature vs Time for all altitudes
    ax1 = plt.subplot(2, 3, 1)
    for i, result in enumerate(all_results):
        if len(result['surface_temps']) > 0:
            plt.plot(result['times'], result['surface_temps'], 
                    label=f"{result['altitude']} km", alpha=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('Surface Temperature (K)')
    plt.title('Surface Temperature vs Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # 2. Drag vs Time for all altitudes
    ax2 = plt.subplot(2, 3, 2)
    drag_altitudes = []
    final_drags = []
    
    for result in all_results:
        if result['drag_data'] is not None:
            drag_data = result['drag_data']
            plt.plot(drag_data['times'], drag_data['drag_force'], 
                    label=f"{result['altitude']} km", alpha=0.8)
            drag_altitudes.append(result['altitude'])
            final_drags.append(drag_data['final_drag'])
    
    plt.xlabel('Time (s)')
    plt.ylabel('Drag Force (N)')
    plt.title('Drag Force vs Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # 3. Final Temperature vs Altitude
    ax3 = plt.subplot(2, 3, 4)
    final_temps = []
    temp_altitudes = []
    
    for result in all_results:
        if len(result['surface_temps']) > 0:
            # Average of last 10 temperature points
            final_temp = np.mean(result['surface_temps'][-10:])
            final_temps.append(final_temp)
            temp_altitudes.append(result['altitude'])
    
    if final_temps:
        plt.plot(temp_altitudes, final_temps, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Altitude (km)')
        plt.ylabel('Final Surface Temperature (K)')
        plt.title('Surface Temperature vs Altitude')
        plt.grid(True)
    
    # 4. Final Drag vs Altitude
    ax4 = plt.subplot(2, 3, 5)
    if final_drags:
        plt.plot(drag_altitudes, final_drags, 'ro-', linewidth=2, markersize=8)
        plt.xlabel('Altitude (km)')
        plt.ylabel('Final Drag Force (N)')
        plt.title('Drag Force vs Altitude')
        plt.grid(True)
        plt.ticklabel_format(style='scientific', axis='y', scilimits=(-3,3))
    
    # 5. Drag Coefficient vs Altitude (if we can estimate reference area)
    ax5 = plt.subplot(2, 3, 6)
    if final_drags:
        # This is a placeholder - you'll need to define reference area and dynamic pressure
        # based on your specific geometry and atmospheric conditions
        plt.plot(drag_altitudes, np.abs(final_drags), 'go-', linewidth=2, markersize=8)
        plt.xlabel('Altitude (km)')
        plt.ylabel('|Drag Force| (N)')
        plt.title('Absolute Drag Force vs Altitude')
        plt.yscale('log')
        plt.grid(True)
    
    # 6. Summary statistics
    ax6 = plt.subplot(2, 3, 3)
    ax6.axis('off')
    
    summary_text = "=== SUMMARY STATISTICS ===\n\n"
    summary_text += f"Altitudes analyzed: {len(altitudes)}\n"
    summary_text += f"Range: {min(altitudes)}-{max(altitudes)} km\n\n"
    
    if final_temps:
        summary_text += f"Temperature Range:\n"
        summary_text += f"  Min: {min(final_temps):.1f} K at {temp_altitudes[np.argmin(final_temps)]} km\n"
        summary_text += f"  Max: {max(final_temps):.1f} K at {temp_altitudes[np.argmax(final_temps)]} km\n\n"
    
    if final_drags:
        summary_text += f"Drag Force Range:\n"
        summary_text += f"  Min: {min(final_drags):.2e} N at {drag_altitudes[np.argmin(final_drags)]} km\n"
        summary_text += f"  Max: {max(final_drags):.2e} N at {drag_altitudes[np.argmax(final_drags)]} km\n\n"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
             verticalalignment='top', fontfamily='monospace', fontsize=10)
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    plot_file = 'outputs/multi_altitude_comprehensive_analysis.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Comprehensive plot saved to: {plot_file}")
    plt.show()
    
    # === EXPORT DATA ===
    
    # Export temperature data
    if final_temps:
        temp_data = np.column_stack([temp_altitudes, final_temps])
        temp_file = 'outputs/temperature_vs_altitude.csv'
        np.savetxt(temp_file, temp_data, delimiter=',', 
                  header='Altitude_km,Final_Temperature_K', comments='')
        print(f"Temperature data exported to: {temp_file}")
    
    # Export drag data
    if final_drags:
        drag_data = np.column_stack([drag_altitudes, final_drags])
        drag_file = 'outputs/drag_vs_altitude.csv'
        np.savetxt(drag_file, drag_data, delimiter=',', 
                  header='Altitude_km,Final_Drag_Force_N', comments='')
        print(f"Drag data exported to: {drag_file}")
    
    # Export detailed time series for each altitude
    for result in all_results:
        alt = result['altitude']
        
        # Temperature time series
        if len(result['surface_temps']) > 0:
            temp_ts_data = np.column_stack([result['times'], result['surface_temps']])
            temp_ts_file = f'outputs/temperature_timeseries_{alt}km.csv'
            np.savetxt(temp_ts_file, temp_ts_data, delimiter=',',
                      header='Time_s,Surface_Temperature_K', comments='')
        
        # Drag time series
        if result['drag_data'] is not None:
            drag_data = result['drag_data']
            drag_ts_data = np.column_stack([drag_data['times'], drag_data['drag_force']])
            drag_ts_file = f'outputs/drag_timeseries_{alt}km.csv'
            np.savetxt(drag_ts_file, drag_ts_data, delimiter=',',
                      header='Time_s,Drag_Force_N', comments='')
    
    print("\n=== Analysis Complete ===")
    print("Files created in outputs/:")
    for file in sorted(glob.glob('outputs/*.csv')):
        print(f"  {file}")
    print(f"  {plot_file}")

if __name__ == "__main__":
    main()