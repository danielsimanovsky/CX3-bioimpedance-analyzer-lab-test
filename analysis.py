# Save this file as analysis.py
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List

# Suppress pandas warning for sep=None
import warnings
warnings.filterwarnings("ignore", "This regex is not supported by the 'python' engine")


class ImpedanceAnalyzer:
    """
    Handles analysis for a SINGLE experiment/distance folder.
    (e.g., "1cm_100uApp")
    """
    
    def __init__(self, experiment_dir: Path, output_dir: Path, 
                 steak_size: float, z_limit: Optional[float] = None):
        
        if not experiment_dir.is_dir():
            raise NotADirectoryError(f"Experiment directory not found: {experiment_dir}")

        self.experiment_dir = experiment_dir
        self.steak_size = steak_size
        self.z_limit = z_limit

        # Create structured output paths
        self.output_dir = output_dir
        self.plot_dir = self.output_dir / "plots"
        self.csv_dir = self.output_dir / "csv_data" # For summary CSVs
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Analyzer initialized for: {self.experiment_dir.name}")

    def run_spec_to_csv_conversion(self) -> str:
        """
        Deletes all old '*_nf.csv' files and then converts .spec
        files into sequentially named .csv files (0.0_nf, 0.1_nf, etc.).
        """
        print(f"  Processing: {self.experiment_dir.name}")
        
        # --- START OF FIX: CLEANUP ---
        # First, find and delete all old, contaminated CSVs
        old_csvs = list(self.experiment_dir.rglob("*_nf.csv"))
        if old_csvs:
            print(f"    Found and deleting {len(old_csvs)} old '*_nf.csv' files...")
            for f in old_csvs:
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"    Could not delete {f.name}: {e}")
        # --- END OF FIX: CLEANUP ---
        
        # Find all .spec files, no matter how deep
        spec_files = sorted(list(self.experiment_dir.rglob("*.spec")))
        if not spec_files:
            print("    No .spec files found.")
            return "No .spec files found in this folder."
        
        print(f"    Found {len(spec_files)} .spec files. Converting...")

        # Generate the new names based on the sort order (which is by timestamp)
        new_names = [f"{round(i*0.1, 1)}_nf.csv" for i in range(len(spec_files))]
        
        total_converted = 0
        
        for spec_path, new_name in zip(spec_files, new_names):
            
            # Save the new CSV in the *same folder* as the .spec file
            csv_path = spec_path.with_name(new_name)
            
            # --- THIS IS THE CHANGE ---
            # We no longer check if the file exists. We will
            # *always* overwrite. This is crucial for a web server.
            # (Removed 'if csv_path.exists(): continue')
            # --- END OF CHANGE ---

            try:
                df = pd.read_csv(spec_path, sep=None, engine="python")
                df.to_csv(csv_path, index=False)
                total_converted += 1
            except Exception as e:
                print(f"    Failed to parse {spec_path.name}, copying raw: {e}")
                shutil.copy(spec_path, csv_path)

        return f"Successfully created {total_converted} new files (cleaned {len(old_csvs)} old files)."

    def plot_2d_graphs(self) -> Tuple[plt.Figure, plt.Figure]:
        """
        Generates two 2D plots for this experiment folder, with
        all 20 capacitance values on each plot.
        """
        # Find ALL the .csv files we just created
        csv_files = sorted(list(self.experiment_dir.rglob("*_nf.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No *_nf.csv files found in: {self.experiment_dir.name}. Did you run the conversion?")

        plot_data = [] # List of (capacitance, min_magnitude, freqs, magnitudes)
        for file in csv_files:
            cap_val = None
            
            if '_' not in file.name:
                continue
            try:
                cap_val = float(file.name.split('_')[0])
            except ValueError:
                print(f"Skipping plot for file (could not parse cap): {file.name}")
                continue
            
            try:
                df = pd.read_csv(file, skiprows=7)
                freqs = df["frequency[Hz]"].values
                mags = np.sqrt(df["Re[Ohm]"].values**2 + df["Im[Ohm]"].values**2)
                
                if self.z_limit:
                    mask = mags <= self.z_limit
                    freqs, mags = freqs[mask], mags[mask]

                if len(mags) > 0:
                    min_mag = mags.min()
                    plot_data.append((cap_val, min_mag, freqs, mags, file.name))
            except Exception as e:
                print(f"Could not read or process CSV data in {file.name}: {e}")

        if not plot_data:
            raise ValueError(f"No valid data to plot in: {self.experiment_dir.name}")
        
        plot_data.sort(key=lambda x: x[0])
        
        # --- Plot 1: |Z| vs. Frequency (All 20 scatters) ---
        fig_z_freq = plt.figure(figsize=(9, 5))
        ax1 = fig_z_freq.add_subplot(111)
        cmap = plt.get_cmap("viridis", len(plot_data))
        
        for i, (cap, min_mag, freqs, mags, fname) in enumerate(plot_data):
            ax1.scatter(freqs, mags, label=f"{cap} nF", color=cmap(i/len(plot_data)), s=10)

        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("|Z| (Ω)")
        ax1.set_yscale("log")
        ax1.set_title(f"Impedance vs. Frequency for {self.experiment_dir.name}")
        ax1.grid(True, which="both", linestyle="--", alpha=0.5)
        
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        fig_z_freq.subplots_adjust(right=0.75)
            
        fig_z_freq.savefig(self.plot_dir / f"{self.experiment_dir.name}_Z_vs_Freq.png")

        # --- Plot 2: Min |Z| vs. Capacitance (All 20 points) ---
        capacitances = [d[0] for d in plot_data]
        min_magnitudes = [d[1] for d in plot_data]

        fig_z_cap = plt.figure(figsize=(7, 5))
        ax2 = fig_z_cap.add_subplot(111)
        
        ax2.plot(capacitances, min_magnitudes, 'bo', label="Min |Z|") 
        
        ax2.set_xlabel("Capacitance (nF)")
        ax2.set_ylabel("Minimum |Z| (Ω)")
        ax2.set_title(f"Minimum Impedance vs. Capacitance for {self.experiment_dir.name}")
        ax2.grid(True, linestyle="--", alpha=0.7)
        fig_z_cap.tight_layout()
        fig_z_cap.savefig(self.plot_dir / f"{self.experiment_dir.name}_MinZ_vs_Cap.png")
        
        return (fig_z_freq, fig_z_cap)

