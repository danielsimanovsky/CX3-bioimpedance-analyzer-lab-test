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
    Handles analysis for a SINGLE experiment folder.
    """

    def __init__(self, experiment_dir: Path, output_dir: Path,
                 steak_size: float, coil_inductance: float,
                 z_limit: Optional[float] = None):

        if not experiment_dir.is_dir():
            raise NotADirectoryError(f"Experiment directory not found: {experiment_dir}")

        self.experiment_dir = experiment_dir
        self.steak_size = steak_size
        self.coil_inductance = coil_inductance  # Accepted here to prevent crash
        self.z_limit = z_limit

        # Create structured output paths
        self.output_dir = output_dir
        self.plot_dir = self.output_dir / "plots"
        self.csv_dir = self.output_dir / "csv_data"  # For summary CSVs

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)

        print(f"Analyzer initialized for: {self.experiment_dir.name}")

    def run_spec_to_csv_conversion(self, cap_values: List[float]) -> str:
        """
        Deletes all old '*_pf.csv' (and legacy *_nf.csv) files and then converts .spec
        files into named .csv files based on the provided list of capacitance values.
        """
        print(f"  Processing: {self.experiment_dir.name}")

        # --- CLEANUP ---
        for pattern in ["*_pf.csv", "*_nf.csv"]:
            old_csvs = list(self.experiment_dir.rglob(pattern))
            if old_csvs:
                print(f"    Found and deleting {len(old_csvs)} old '{pattern}' files...")
                for f in old_csvs:
                    try:
                        os.remove(f)
                    except Exception as e:
                        print(f"    Could not delete {f.name}: {e}")

        # Find all .spec files
        spec_files = sorted(list(self.experiment_dir.rglob("*.spec")))
        if not spec_files:
            print("    No .spec files found.")
            return "No .spec files found in this folder."

        print(f"    Found {len(spec_files)} .spec files. Converting...")

        # --- UPDATED NAMING LOGIC (Explicit List) ---
        if len(cap_values) < len(spec_files):
            print(
                f"WARNING: Not enough capacitance values provided! Found {len(spec_files)} files but only {len(cap_values)} values.")

        total_converted = 0

        for spec_path, cap_val in zip(spec_files, cap_values):
            new_name = f"{float(cap_val)}_pf.csv"
            csv_path = spec_path.with_name(new_name)

            try:
                df = pd.read_csv(spec_path, sep=None, engine="python")
                df.to_csv(csv_path, index=False)
                total_converted += 1
            except Exception as e:
                print(f"    Failed to parse {spec_path.name}, copying raw: {e}")
                shutil.copy(spec_path, csv_path)

        return f"Successfully created {total_converted} new files (cleaned old files)."

    def plot_2d_graphs(self, plot_type: str, target_freq_mhz: float) -> Tuple[
        plt.Figure, plt.Figure, plt.Figure, plt.Figure, pd.DataFrame, pd.DataFrame]:
        """
        Generates FOUR 2D plots for this experiment folder.
        Returns: (fig_spectrum, fig_max_val, fig_freq_at_max, fig_target_freq, df_summary, df_spectrum)
        """
        csv_files = sorted(list(self.experiment_dir.rglob("*_pf.csv")))
        if not csv_files:
            raise FileNotFoundError(
                f"No *_pf.csv files found in: {self.experiment_dir.name}. Did you run the conversion?")

        plot_data = []  # List to store parsed data
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
                reals = df["Re[Ohm]"].values
                imags = df["Im[Ohm]"].values
                mags = np.sqrt(reals ** 2 + imags ** 2)

                if self.z_limit:
                    mask = mags <= self.z_limit
                    freqs, reals, imags, mags = freqs[mask], reals[mask], imags[mask], mags[mask]

                if len(mags) > 0:
                    # --- PARALLEL CALCULATION ---
                    L_henry = self.coil_inductance * 1e-9  # nH to H
                    C_farad = cap_val * 1e-12  # pF to F
                    epsilon = 1e-12

                    parallel_z = np.zeros_like(freqs)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        safe_freqs = freqs + epsilon
                        safe_reals = reals + epsilon

                        if L_henry <= 0 or C_farad <= 0:
                            parallel_z[:] = np.nan
                        else:
                            XL = 2 * np.pi * safe_freqs * L_henry
                            XC = 1 / (2 * np.pi * safe_freqs * C_farad)
                            term1 = (1 / safe_reals) ** 2
                            term2 = (1 / XL - 1 / XC) ** 2
                            parallel_z = 1 / np.sqrt(term1 + term2)

                    parallel_z[~np.isfinite(parallel_z)] = np.nan

                    plot_data.append((cap_val, freqs, mags, reals, imags, parallel_z))
            except Exception as e:
                print(f"Could not read or process CSV data in {file.name}: {e}")

        if not plot_data:
            raise ValueError(f"No valid data to plot in: {self.experiment_dir.name}")

        plot_data.sort(key=lambda x: x[0])  # Sort by capacitance

        # --- Plot 1 Setup ---
        fig_z_freq = plt.figure(figsize=(9, 5))
        ax1 = fig_z_freq.add_subplot(111)
        cmap = plt.get_cmap("viridis", len(plot_data))

        y_label = ""

        # Data collectors
        plot_values_at_max = []
        plot_freqs_at_max = []
        plot_values_at_target_freq = []

        spectrum_data_list = []

        target_freq_hz = target_freq_mhz * 1e6

        for i, (cap, freqs, mags, reals, imags, parallel_z) in enumerate(plot_data):
            y_data = None

            # --- SELECTION LOGIC ---
            if plot_type == 'MagZ':
                y_data = mags
                y_label = "|Z| (Ω)"
            elif plot_type == 'ReZ':
                y_data = reals
                y_label = "Re(Z) (Ω)"
            elif plot_type == 'ImZ':
                y_data = imags
                y_label = "Im(Z) (Ω)"
            elif plot_type == 'MagZ_Parallel':
                y_data = parallel_z
                y_label = "|Z| Parallel Model (Ω)"

            # 1. Find Peak (Max)
            if y_data is not None and len(y_data) > 0 and not np.all(np.isnan(y_data)):
                max_idx = np.nanargmax(y_data)
                plot_values_at_max.append(y_data[max_idx])
                plot_freqs_at_max.append(freqs[max_idx] / 1e6)

                # 2. Find value at Target Frequency
                # Find index of frequency closest to target
                idx_closest = (np.abs(freqs - target_freq_hz)).argmin()
                val_closest = y_data[idx_closest]
                plot_values_at_target_freq.append(val_closest)

                # Collect Data for Download
                df_temp = pd.DataFrame({
                    "Frequency (Hz)": freqs,
                    "Frequency (MHz)": freqs / 1e6,
                    f"{y_label}": y_data,
                    "Capacitance (pF)": cap
                })
                spectrum_data_list.append(df_temp)

            else:
                plot_values_at_max.append(0)
                plot_freqs_at_max.append(0)
                plot_values_at_target_freq.append(0)

            # Scatter plot for Plot 1
            ax1.scatter(freqs / 1e6, y_data, label=f"{cap} pF", color=cmap(i / len(plot_data)), s=10)

        # --- Prepare DataFrames ---
        capacitances = [d[0] for d in plot_data]
        df_summary = pd.DataFrame({
            "Capacitance (pF)": capacitances,
            f"Peak {y_label}": plot_values_at_max,
            f"Frequency at Peak (MHz)": plot_freqs_at_max,
            f"Value at {target_freq_mhz} MHz": plot_values_at_target_freq
        })

        if spectrum_data_list:
            df_spectrum = pd.concat(spectrum_data_list, ignore_index=True)
        else:
            df_spectrum = pd.DataFrame()

        # --- Plot 1: Spectrum ---
        ax1.set_xlabel("Frequency (MHz)")
        ax1.set_ylabel(y_label)
        ax1.set_yscale("log" if plot_type in ['MagZ', 'MagZ_Parallel'] else "linear")
        ax1.set_title(f"{y_label} vs. Frequency for {self.experiment_dir.name}")
        #ax1.axvline(x=target_freq_mhz, color='r', linestyle='--', alpha=0.5,
         #           label=f"Target: {target_freq_mhz} MHz")  # Visual aid
        ax1.grid(True, which="both", linestyle="--", alpha=0.5)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        fig_z_freq.subplots_adjust(right=0.75)
        fig_z_freq.savefig(self.plot_dir / f"{self.experiment_dir.name}_{plot_type}_vs_Freq.png")

        # --- Plot 2: Peak Value vs. Capacitance ---
        fig_z_cap = plt.figure(figsize=(7, 5))
        ax2 = fig_z_cap.add_subplot(111)
        ax2.plot(capacitances, plot_values_at_max, 'bo-', label=f"Peak Value")
        ax2.set_xlabel("Capacitance (pF)")
        ax2.set_ylabel(f"Peak {y_label}")
        ax2.set_title(f"Peak {y_label} vs. Capacitance")
        ax2.grid(True, linestyle="--", alpha=0.7)
        fig_z_cap.tight_layout()
        fig_z_cap.savefig(self.plot_dir / f"{self.experiment_dir.name}_{plot_type}_PeakVal_vs_Cap.png")

        # --- Plot 3: Frequency at Peak vs. Capacitance ---
        fig_freq_cap = plt.figure(figsize=(7, 5))
        ax3 = fig_freq_cap.add_subplot(111)
        ax3.plot(capacitances, plot_freqs_at_max, 'ro-', label=f"Freq at Peak")
        ax3.set_xlabel("Capacitance (pF)")
        ax3.set_ylabel(f"Frequency at Max {y_label} (MHz)")
        ax3.set_title(f"Freq at Max {y_label} vs. Capacitance")
        ax3.grid(True, linestyle="--", alpha=0.7)
        fig_freq_cap.tight_layout()
        fig_freq_cap.savefig(self.plot_dir / f"{self.experiment_dir.name}_{plot_type}_FreqAtPeak_vs_Cap.png")

        # --- Plot 4: Value at Target Frequency vs. Capacitance ---
        fig_target_val = plt.figure(figsize=(7, 5))
        ax4 = fig_target_val.add_subplot(111)
        ax4.plot(capacitances, plot_values_at_target_freq, 'go-', label=f"Val at {target_freq_mhz} MHz")
        ax4.set_xlabel("Capacitance (pF)")
        ax4.set_ylabel(f"{y_label} at {target_freq_mhz} MHz")
        ax4.set_title(f"{y_label} at {target_freq_mhz} MHz vs. Capacitance")
        ax4.grid(True, linestyle="--", alpha=0.7)
        fig_target_val.tight_layout()
        fig_target_val.savefig(self.plot_dir / f"{self.experiment_dir.name}_{plot_type}_TargetFreqVal_vs_Cap.png")

        return (fig_z_freq, fig_z_cap, fig_freq_cap, fig_target_val, df_summary, df_spectrum)
