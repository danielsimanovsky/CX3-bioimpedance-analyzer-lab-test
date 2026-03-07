import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
from typing import Optional, Tuple, List
import warnings

warnings.filterwarnings("ignore", "This regex is not supported by the 'python' engine")


class ImpedanceAnalyzer:
    """Extracts raw .spec data into DataFrames to be stored in memory."""

    def __init__(self, experiment_dir: Path, coil_inductance: float = 1.0):
        self.experiment_dir = experiment_dir
        self.coil_inductance = coil_inductance

    def extract_timeseries_data(self) -> pd.DataFrame:
        spec_files = sorted(list(self.experiment_dir.rglob("*.spec")))
        if not spec_files:
            return pd.DataFrame()

        data_rows = []
        for f in spec_files:
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                raw_time = lines[6].strip()
                data_line = lines[8].strip().split(',')
                data_rows.append({
                    "Filename": f.name, "Raw_Time": raw_time,
                    "Frequency[Hz]": float(data_line[0]),
                    "Re[Ohm]": float(data_line[1]), "Im[Ohm]": float(data_line[2])
                })
            except Exception:
                pass

        if not data_rows: return pd.DataFrame()
        df = pd.DataFrame(data_rows)

        def parse_time(t_str):
            try:
                clean_str = t_str.replace('.-', '-').replace('a.m.', 'AM').replace('p.m.', 'PM')
                parts = clean_str.split(':')
                if len(parts) >= 4:
                    clean_str = f"{parts[0]}:{parts[1]}:{parts[2]}.{parts[3]}"
                return pd.to_datetime(clean_str, format="%d-%b-%Y %I:%M:%S.%f %p")
            except:
                return pd.NaT

        df['Timestamp'] = df['Raw_Time'].apply(parse_time)
        df = df.dropna(subset=['Timestamp']).sort_values('Timestamp')
        df['Time_Seconds'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds()

        # Calculate magnitudes for potential filtering
        df['|Z|[Ohm]'] = np.sqrt(df['Re[Ohm]'] ** 2 + df['Im[Ohm]'] ** 2)

        # Parallel Z (Safeguard calculation)
        L_henry = self.coil_inductance * 1e-9
        C_farad = 1e-12
        epsilon = 1e-12
        with np.errstate(divide='ignore', invalid='ignore'):
            XL = 2 * np.pi * (df['Frequency[Hz]'] + epsilon) * L_henry
            XC = 1 / (2 * np.pi * (df['Frequency[Hz]'] + epsilon) * C_farad)
            df['Parallel_Z'] = 1 / np.sqrt((1 / (df['Re[Ohm]'] + epsilon)) ** 2 + (1 / XL - 1 / XC) ** 2)

        return df

    def extract_spectrum_data(self, cap_values: List[float]) -> pd.DataFrame:
        spec_files = sorted(list(self.experiment_dir.rglob("*.spec")))
        if not spec_files: return pd.DataFrame()

        data_rows = []
        for spec_path, cap_val in zip(spec_files, cap_values):
            try:
                df_temp = pd.read_csv(spec_path, skiprows=7)
                freqs = df_temp["frequency[Hz]"].values
                reals = df_temp["Re[Ohm]"].values
                imags = df_temp["Im[Ohm]"].values
                mags = np.sqrt(reals ** 2 + imags ** 2)

                L_henry = self.coil_inductance * 1e-9
                C_farad = cap_val * 1e-12
                epsilon = 1e-12

                with np.errstate(divide='ignore', invalid='ignore'):
                    XL = 2 * np.pi * (freqs + epsilon) * L_henry
                    XC = 1 / (2 * np.pi * (freqs + epsilon) * C_farad)
                    parallel_z = 1 / np.sqrt((1 / (reals + epsilon)) ** 2 + (1 / XL - 1 / XC) ** 2)

                for f, r, i, m, p in zip(freqs, reals, imags, mags, parallel_z):
                    data_rows.append({
                        "Capacitance (pF)": cap_val, "Frequency (Hz)": f, "Frequency (MHz)": f / 1e6,
                        "Re[Ohm]": r, "Im[Ohm]": i, "MagZ": m, "Parallel_Z": p
                    })
            except Exception:
                pass

        return pd.DataFrame(data_rows)


# --- DYNAMIC PLOTTING FUNCTIONS ---
def plot_timeseries_interactive(df: pd.DataFrame, exp_name: str, plot_type: str, z_limit: Optional[float] = None):
    col_map = {
        'MagZ': ('|Z|[Ohm]', '|Z| (Ω)'),
        'ReZ': ('Re[Ohm]', 'Re(Z) (Ω)'),
        'ImZ': ('Im[Ohm]', 'Im(Z) (Ω)'),
        'MagZ_Parallel': ('Parallel_Z', '|Z| Parallel (Ω)')
    }
    y_col, y_label = col_map.get(plot_type, ('|Z|[Ohm]', '|Z| (Ω)'))

    plot_df = df.copy()
    if z_limit:
        plot_df = plot_df[plot_df['|Z|[Ohm]'] <= z_limit]

    fig = go.Figure()
    if plot_df.empty:
        fig.update_layout(title=f"<b>No Data Available (Z-Limit {z_limit}Ω exceeded)</b>", template="plotly_white")
        return fig

    fig.add_trace(go.Scatter(
        x=plot_df['Time_Seconds'], y=plot_df[y_col], mode='lines+markers',
        marker=dict(size=6, color='royalblue'), line=dict(width=2), name=exp_name,
        hovertemplate="Elapsed: %{x:.1f} s<br>" + y_label + ": %{y:.2f}<br>Time: %{customdata}<extra></extra>",
        customdata=plot_df['Raw_Time']
    ))

    yaxis_config = dict(title=y_label)
    if z_limit and plot_type in ['MagZ', 'MagZ_Parallel']:
        yaxis_config['range'] = [0, z_limit * 1.05]

    fig.update_layout(
        title=f"<b>{y_label} vs Time - {exp_name}</b>", xaxis_title="Elapsed Time (Seconds)",
        yaxis=yaxis_config, template="plotly_white", hovermode="x unified", margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def plot_spectrum_static(df: pd.DataFrame, exp_name: str, plot_type: str, target_freq_mhz: float,
                         z_limit: Optional[float] = None):
    col_map = {
        'MagZ': ('MagZ', '|Z| (Ω)'),
        'ReZ': ('Re[Ohm]', 'Re(Z) (Ω)'),
        'ImZ': ('Im[Ohm]', 'Im(Z) (Ω)'),
        'MagZ_Parallel': ('Parallel_Z', '|Z| Parallel (Ω)')
    }
    y_col, y_label = col_map.get(plot_type, ('MagZ', '|Z| (Ω)'))

    plot_df = df.copy()
    if z_limit:
        plot_df = plot_df[plot_df['MagZ'] <= z_limit]

    caps = sorted(plot_df['Capacitance (pF)'].unique())
    fig_z_freq = plt.figure(figsize=(9, 5))
    ax1 = fig_z_freq.add_subplot(111)
    cmap = plt.get_cmap("viridis", len(caps) if len(caps) > 0 else 1)

    plot_values_at_max, plot_freqs_at_max, plot_values_at_target_freq = [], [], []
    target_freq_hz = target_freq_mhz * 1e6

    for i, cap in enumerate(caps):
        cap_df = plot_df[plot_df['Capacitance (pF)'] == cap]
        freqs = cap_df['Frequency (Hz)'].values
        y_vals = cap_df[y_col].values

        if len(y_vals) > 0 and not np.all(np.isnan(y_vals)):
            max_idx = np.nanargmax(y_vals)
            plot_values_at_max.append(y_vals[max_idx])
            plot_freqs_at_max.append(freqs[max_idx] / 1e6)
            idx_closest = (np.abs(freqs - target_freq_hz)).argmin()
            plot_values_at_target_freq.append(y_vals[idx_closest])
        else:
            plot_values_at_max.append(0);
            plot_freqs_at_max.append(0);
            plot_values_at_target_freq.append(0)

        ax1.scatter(freqs / 1e6, y_vals, label=f"{cap} pF", color=cmap(i / len(caps)), s=10)

    df_summary = pd.DataFrame({
        "Capacitance (pF)": caps,
        f"Peak {y_label}": plot_values_at_max,
        f"Frequency at Peak (MHz)": plot_freqs_at_max,
        f"Value at {target_freq_mhz} MHz": plot_values_at_target_freq
    })

    ax1.set_xlabel("Frequency (MHz)")
    ax1.set_ylabel(y_label)
    ax1.set_yscale("log" if plot_type in ['MagZ', 'MagZ_Parallel'] else "linear")
    if z_limit and plot_type in ['MagZ', 'MagZ_Parallel']: ax1.set_ylim(top=z_limit * 1.1)
    ax1.set_title(f"{y_label} vs. Frequency for {exp_name}")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    fig_z_freq.subplots_adjust(right=0.75)

    fig_z_cap = plt.figure(figsize=(7, 5));
    ax2 = fig_z_cap.add_subplot(111)
    ax2.plot(caps, plot_values_at_max, 'bo-', label="Peak Value")
    ax2.set_xlabel("Capacitance (pF)");
    ax2.set_ylabel(f"Peak {y_label}");
    ax2.set_title(f"Peak {y_label} vs. Capacitance");
    ax2.grid(True, linestyle="--", alpha=0.7)

    fig_freq_cap = plt.figure(figsize=(7, 5));
    ax3 = fig_freq_cap.add_subplot(111)
    ax3.plot(caps, plot_freqs_at_max, 'ro-', label="Freq at Peak")
    ax3.set_xlabel("Capacitance (pF)");
    ax3.set_ylabel(f"Frequency at Max {y_label} (MHz)");
    ax3.set_title(f"Freq at Max {y_label} vs. Capacitance");
    ax3.grid(True, linestyle="--", alpha=0.7)

    fig_target_val = plt.figure(figsize=(7, 5));
    ax4 = fig_target_val.add_subplot(111)
    ax4.plot(caps, plot_values_at_target_freq, 'go-', label=f"Val at {target_freq_mhz} MHz")
    ax4.set_xlabel("Capacitance (pF)");
    ax4.set_ylabel(f"{y_label} at {target_freq_mhz} MHz");
    ax4.set_title(f"{y_label} at {target_freq_mhz} MHz vs. Capacitance");
    ax4.grid(True, linestyle="--", alpha=0.7)

    return (fig_z_freq, fig_z_cap, fig_freq_cap, fig_target_val, df_summary)
