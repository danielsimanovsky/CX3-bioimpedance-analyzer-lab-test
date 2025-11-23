import streamlit as st
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Tuple

# --- New libraries for unzipping in memory ---
import zipfile
import tempfile

# Import your backend class
try:
    from testAnalysis import ImpedanceAnalyzer
except ImportError:
    st.error("Missing 'analysis.py'. Please make sure it's in the same folder.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Impedance Analysis Dashboard",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Impedance Analysis Dashboard (Web Version)")
st.markdown("Upload a single `.zip` file containing your data folders to begin.")


# --- 3D PLOTTING FUNCTIONS (Unchanged) ---

def get_all_csv_data(root_dir: Path, selected_folders: List[str],
                     z_limit: Optional[float], coil_inductance: float
                     ) -> pd.DataFrame:
    """Helper to loop through all selected folders and compile CSV data."""
    all_data = []
    L_henry = coil_inductance * 1e-9
    epsilon = 1e-12

    for folder_name in selected_folders:
        exp_path = root_dir / folder_name
        match = re.match(r"([0-9.]+)", folder_name)
        if match:
            try:
                distance_val = float(match.group(1))
            except ValueError:
                distance_val = 0.0
        else:
            distance_val = 0.0

        csv_files = list(exp_path.rglob("*_nf.csv"))

        for file in csv_files:
            try:
                cap_val = float(file.name.split("_")[0])
            except Exception:
                continue

            try:
                df = pd.read_csv(file, skiprows=7)
                df['distance'] = distance_val
                df['capacitance'] = cap_val

                df['frequency_mhz'] = df['frequency[Hz]'] / 1e6
                df['real_z'] = df["Re[Ohm]"]
                df['imag_z'] = df["Im[Ohm]"]
                df['magnitude'] = np.sqrt(df["Re[Ohm]"] ** 2 + df["Im[Ohm]"] ** 2)

                C_farad = df['capacitance'] * 1e-9
                with np.errstate(divide='ignore', invalid='ignore'):
                    safe_freqs = df['frequency[Hz]'] + epsilon
                    safe_reals = df['real_z'] + epsilon
                    if L_henry <= 0 or C_farad <= 0:
                        df['parallel_z'] = np.nan
                    else:
                        XL = 2 * np.pi * safe_freqs * L_henry
                        XC = 1 / (2 * np.pi * safe_freqs * C_farad)
                        term1 = (1 / safe_reals) ** 2
                        term2 = (1 / XL - 1 / XC) ** 2
                        df['parallel_z'] = 1 / np.sqrt(term1 + term2)
                df['parallel_z'] = df['parallel_z'].replace([np.inf, -np.inf], np.nan)

                if z_limit:
                    df = df[df['magnitude'] <= z_limit].copy()

                all_data.append(df)
            except Exception as e:
                print(f"Error reading {file.name}: {e}")

    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)


def plot_3d_distance_plotly(all_data: pd.DataFrame, y_column: str, y_label: str) -> go.Figure:
    fig = go.Figure()
    if len(all_data) > 0:
        plot_data = all_data.sample(min(len(all_data), 50000))
        fig.add_trace(go.Scatter3d(
            x=plot_data['frequency_mhz'], y=plot_data['distance'], z=plot_data[y_column],
            mode="markers", marker=dict(size=2, opacity=0.5, color=plot_data[y_column], colorscale='Viridis'),
            name=y_label
        ))
    fig.update_layout(
        scene=dict(xaxis=dict(title="Frequency (MHz)", type="log"),
                   zaxis=dict(title=y_label, type="log" if y_column in ['magnitude', 'parallel_z'] else 'linear'),
                   yaxis=dict(title="Relative distance/Variable (cm/uH)")),
        title=f"3D {y_label} vs. Frequency vs. Variable",
        legend=dict(itemsizing="constant", font=dict(size=8)))
    return fig


def plot_min_impedance_vs_distance(all_data: pd.DataFrame, steak_size: float, y_column: str, y_label: str) -> Tuple[
    plt.Figure, pd.DataFrame]:
    min_z_by_dist = all_data.groupby('distance')[y_column].min()
    min_freq_by_dist = all_data.loc[all_data.groupby('distance')[y_column].idxmin()].set_index('distance')[
        'frequency_mhz']
    df_results = pd.DataFrame({
        "Variable": min_z_by_dist.index, f"Min {y_label}": min_z_by_dist.values,
        "Frequency at Min (MHz)": min_freq_by_dist.loc[min_z_by_dist.index].values
    })
    df_results["Relative Variable"] = df_results["Variable"] / steak_size
    df_results = df_results.sort_values(by="Relative Variable")
    fig = plt.figure(figsize=(7, 5));
    ax = fig.add_subplot(111)
    ax.plot(df_results["Relative Variable"], df_results[f"Min {y_label}"], marker="o", linestyle="-", color="b")
    ax.set_xlabel("Relative Variable");
    ax.set_ylabel(f"Minimum {y_label}")
    ax.set_title(f"Minimum {y_label} vs Variable");
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()
    return (fig, df_results)


def plot_impedance_range_vs_distance(all_data: pd.DataFrame, steak_size: float, y_column: str, y_label: str) -> Tuple[
    plt.Figure, pd.DataFrame]:
    grouped = all_data.groupby('distance')[y_column]
    min_z = grouped.min();
    max_z = grouped.max()
    df_results = pd.DataFrame({
        "Variable": min_z.index, f"Min {y_label}": min_z.values, f"Max {y_label}": max_z.loc[min_z.index].values,
    })
    df_results[f"Δ{y_label}"] = df_results[f"Max {y_label}"] - df_results[f"Min {y_label}"]
    df_results["Relative Variable"] = df_results["Variable"] / steak_size
    df_results = df_results.sort_values(by="Relative Variable")
    fig = plt.figure(figsize=(7, 5));
    ax = fig.add_subplot(111)
    ax.plot(df_results["Relative Variable"], df_results[f"Δ{y_label}"], marker="o", linestyle="-", color="r")
    ax.set_xlabel("Relative Variable");
    ax.set_ylabel(f"Δ{y_label}")
    ax.set_title(f"{y_label} Range vs Variable");
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()
    return (fig, df_results)


def plot_3d_impedance_vs_capacitance(all_data: pd.DataFrame, steak_size: float, y_column: str, y_label: str) -> Tuple[
    go.Figure, pd.DataFrame]:
    min_mag_data = all_data.loc[all_data.groupby(['distance', 'capacitance'])['magnitude'].idxmin()]
    min_mag_data['Relative Variable'] = min_mag_data['distance'] / steak_size
    fig = go.Figure()
    for dist, group in min_mag_data.groupby('Relative Variable'):
        group = group.sort_values(by='capacitance')
        fig.add_trace(go.Scatter3d(
            x=group['capacitance'], y=group['Relative Variable'], z=group[y_column],
            mode="lines+markers", marker=dict(size=4), line=dict(width=2),
            name=f"Var: {dist:.2f}",
            hovertemplate=f"Var: %{{y:.2f}}<br>{y_label}: %{{z:.2f}} Ω<br>Capacitance: %{{x}} nF<extra></extra>"
        ))
    fig.update_layout(
        scene=dict(xaxis=dict(title="Capacitance (nF)"),
                   zaxis=dict(title=f"{y_label} at Resonance",
                              type="log" if y_column in ['magnitude', 'parallel_z'] else 'linear'),
                   yaxis=dict(title="Relative Variable")),
        title=f"3D {y_label} at Resonance vs. Capacitance vs. Variable"
    )
    df_results = min_mag_data[["Relative Variable", "capacitance", y_column]].rename(columns={
        "capacitance": "Capacitance (nF)", y_column: f"{y_label} at Resonance (Ω)"
    })
    return (fig, df_results)


# --- UI & SESSION STATE LOGIC ---

# Initialize session state for results if it doesn't exist
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# --- 1. Sidebar Configuration ---
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader(
        "Upload your Root Data Folder as a .zip file",
        type="zip"
    )

    st.header("2. Configuration")
    st.subheader("Capacitance Settings")
    cap_start = st.number_input("Capacitance Start (nF)", value=0.0, step=0.1, format="%.2f")
    cap_step = st.number_input("Capacitance Step (nF)", value=0.1, step=0.1, min_value=0.01, format="%.2f")

    st.subheader("Physical Settings")
    steak_size = st.number_input("Steak Size (cm)", min_value=0.1, value=24.0, step=0.1)
    coil_inductance = st.number_input("Coil Inductance (nH)", value=1.0, step=0.1, min_value=0.001, format="%.3f")

    st.subheader("Analysis Settings")
    z_limit = st.number_input("Z-Limit (Ohm) (0 for no limit)", min_value=0, value=100000, step=1000)
    z_limit_val = z_limit if z_limit > 0 else None

    plot_type_str = st.radio(
        "Select Data to Plot:",
        ('|Z| (Magnitude)', '|Z| (Parallel Model)', 'Re(Z) (Real)', 'Im(Z) (Imaginary)'),
        index=0
    )

    plot_type_map = {
        '|Z| (Magnitude)': ('magnitude', '|Z| (Ω)', 'MagZ'),
        '|Z| (Parallel Model)': ('parallel_z', '|Z| Parallel (Ω)', 'MagZ_Parallel'),
        'Re(Z) (Real)': ('real_z', 'Re(Z) (Ω)', 'ReZ'),
        'Im(Z) (Imaginary)': ('imag_z', 'Im(Z) (Ω)', 'ImZ')
    }
    plot_y_column, plot_y_label, plot_type_2d = plot_type_map[plot_type_str]

    if plot_type_str == '|Z| (Parallel Model)' and (coil_inductance <= 0 or cap_step <= 0):
        st.error("Coil Inductance and Cap. Step must be > 0 to use the Parallel Model.")
        st.stop()

    st.header("3. Analysis Steps")
    st.info("Conversion is automatic on every run.")

    run_2d_plots = st.checkbox("2D Plots (per Experiment)", value=True)
    run_3d_dist = st.checkbox("3D Spectra vs. Distance", value=True)
    run_3d_cap = st.checkbox("3D |Z| vs. Capacitance", value=True)
    run_min_z = st.checkbox("Min Value vs. Distance", value=True)
    run_delta_z = st.checkbox("Value Range vs. Distance", value=True)

    run_button = st.button(f"**🚀 Run Analysis**")

# --- MAIN LOGIC ---

if not uploaded_file:
    st.info("Please upload your data .zip file using the sidebar to begin.")
    # Clear results if no file is present
    st.session_state.analysis_results = None
    st.stop()

# Logic to Run Analysis and Store Results
if run_button:
    results_container = {'experiments': {}, 'combined': {}}

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # 1. Extract
        with st.spinner("Extracting data..."):
            try:
                with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir_path)
            except Exception as e:
                st.error(f"Error extracting ZIP file: {e}");
                st.stop()

        unzipped_items = list(temp_dir_path.iterdir())
        if len(unzipped_items) == 1 and unzipped_items[0].is_dir():
            root_dir = unzipped_items[0]
        else:
            root_dir = temp_dir_path

        st.success(f"Data extracted. Found root folder: `{root_dir.name}`")

        try:
            exp_folders = [f for f in root_dir.iterdir() if f.is_dir() and any(char.isdigit() for char in f.name)]
            selected_experiments = sorted([f.name for f in exp_folders])
            if not selected_experiments:
                st.error(f"No valid experiment folders found.");
                st.stop()
        except Exception as e:
            st.error(f"Could not read folders: {e}");
            st.stop()

        # 2. Process 2D Data
        for exp_name in selected_experiments:
            exp_path = root_dir / exp_name
            exp_output_path = temp_dir_path / "results" / exp_name

            try:
                analyzer = ImpedanceAnalyzer(
                    experiment_dir=exp_path, output_dir=exp_output_path,
                    steak_size=steak_size, coil_inductance=coil_inductance, z_limit=z_limit_val
                )
                analyzer.run_spec_to_csv_conversion(cap_start, cap_step)

                if run_2d_plots:
                    # Store figures and dataframes
                    fig1, fig2, fig3, df_sum, df_spec = analyzer.plot_2d_graphs(plot_type_2d)
                    results_container['experiments'][exp_name] = {
                        'figs': (fig1, fig2, fig3),
                        'df_sum': df_sum,
                        'df_spec': df_spec
                    }
            except Exception as e:
                st.warning(f"Skipped {exp_name}: {e}")
                continue

        # 3. Process 3D/Combined Data
        all_data_df = pd.DataFrame()
        if any([run_3d_dist, run_3d_cap, run_min_z, run_delta_z]):
            try:
                all_data_df = get_all_csv_data(root_dir, selected_experiments, z_limit_val, coil_inductance)
            except Exception as e:
                st.error(f"Error loading combined data: {e}")

        if not all_data_df.empty:
            combined_res = {}
            if run_3d_dist:
                combined_res['3d_dist'] = plot_3d_distance_plotly(all_data_df, plot_y_column, plot_y_label)
            if run_min_z:
                combined_res['min_z'] = plot_min_impedance_vs_distance(all_data_df, steak_size, plot_y_column,
                                                                       plot_y_label)
            if run_delta_z:
                combined_res['delta_z'] = plot_impedance_range_vs_distance(all_data_df, steak_size, plot_y_column,
                                                                           plot_y_label)
            if run_3d_cap:
                combined_res['3d_cap'] = plot_3d_impedance_vs_capacitance(all_data_df, steak_size, plot_y_column,
                                                                          plot_y_label)

            results_container['combined'] = combined_res

    # Save to session state
    st.session_state.analysis_results = results_container
    st.session_state.plot_type_display = plot_type_str  # Save the label for display

# --- RENDERING RESULTS FROM SESSION STATE ---

if st.session_state.analysis_results:
    res = st.session_state.analysis_results
    display_type = st.session_state.get('plot_type_display', plot_type_str)

    # 1. Render 2D Plots
    st.header(f"1. 2D Analysis (Plotting {display_type})")

    for exp_name, data in res['experiments'].items():
        with st.expander(f"▼ Results for: {exp_name}", expanded=False):
            fig1, fig2, fig3 = data['figs']

            st.pyplot(fig1)
            st.download_button(
                label=f"📥 Download Spectrum Data ({exp_name})",
                data=data['df_spec'].to_csv(index=False).encode('utf-8'),
                file_name=f"{exp_name}_spectrum_data.csv", mime='text/csv', key=f"dl_spec_{exp_name}"
            )

            col1, col2 = st.columns(2)
            with col1: st.pyplot(fig2)
            with col2: st.pyplot(fig3)

            st.download_button(
                label=f"📥 Download Analysis Data ({exp_name})",
                data=data['df_sum'].to_csv(index=False).encode('utf-8'),
                file_name=f"{exp_name}_analysis_summary.csv", mime='text/csv', key=f"dl_sum_{exp_name}"
            )

    # 2. Render 3D/Combined Plots
    st.header(f"2. 3D & Summary Analysis (Plotting {display_type})")
    combined = res.get('combined', {})

    if '3d_dist' in combined:
        st.subheader(f"3D {display_type} vs. Variable")
        st.plotly_chart(combined['3d_dist'], use_container_width=True)

    if 'min_z' in combined:
        fig, df = combined['min_z']
        st.subheader(f"Minimum {display_type} vs. Variable")
        st.pyplot(fig)
        with st.expander("View Data"): st.dataframe(df)
        st.download_button("📥 Download Min Data", df.to_csv(index=False).encode('utf-8'), "min_data.csv", "text/csv")

    if 'delta_z' in combined:
        fig, df = combined['delta_z']
        st.subheader(f"{display_type} Range (Δ) vs. Variable")
        st.pyplot(fig)
        with st.expander("View Data"): st.dataframe(df)
        st.download_button("📥 Download Delta Data", df.to_csv(index=False).encode('utf-8'), "delta_data.csv",
                           "text/csv")

    if '3d_cap' in combined:
        fig, df = combined['3d_cap']
        st.subheader(f"3D {display_type} at Resonance vs. Capacitance")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("View Raw Data"): st.dataframe(df)
        st.download_button("📥 Download 3D Resonance Data", df.to_csv(index=False).encode('utf-8'), "resonance_data.csv",
                           "text/csv")
