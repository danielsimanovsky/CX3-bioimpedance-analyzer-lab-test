# Save this file as app.py
import streamlit as st
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Tuple

# --- New libraries for unzipping in memory ---
import zipfile
import tempfile 

# Import your backend class
try:
    from analysis import ImpedanceAnalyzer
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

# ---
# 3D PLOTTING FUNCTIONS
# (Copied from our previous app.py, no changes needed)
# ---

def get_all_csv_data(root_dir: Path, selected_folders: List[str], z_limit: Optional[float]
                     ) -> pd.DataFrame:
    all_data = []
    
    for folder_name in selected_folders:
        exp_path = root_dir / folder_name
        
        try:
            distance_val = float(folder_name.split("cm")[0])
        except Exception:
            print(f"Could not parse distance from: {folder_name}")
            continue
            
        csv_files = list(exp_path.rglob("*_nf.csv")) # Only find the converted files
        
        for file in csv_files:
            try:
                cap_val = float(file.name.split("_")[0])
            except Exception:
                continue # Skip non-data CSVs
            
            try:
                df = pd.read_csv(file, skiprows=7)
                df['distance'] = distance_val
                df['capacitance'] = cap_val
                df['magnitude'] = np.sqrt(df["Re[Ohm]"]**2 + df["Im[Ohm]"]**2)
                
                if z_limit:
                    df = df[df['magnitude'] <= z_limit].copy()
                    
                all_data.append(df)
            except Exception as e:
                print(f"Error reading {file.name}: {e}")
                
    if not all_data:
        return pd.DataFrame()
        
    return pd.concat(all_data, ignore_index=True)


def plot_3d_distance_plotly(all_data: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    grouped = all_data.groupby(['distance', 'capacitance'])
    
    for (distance, capacitance), group in grouped:
        fig.add_trace(go.Scatter3d(
            x=group['frequency[Hz]'], 
            y=group['distance'], 
            z=group['magnitude'],
            mode="markers", 
            marker=dict(size=3, opacity=0.7),
            name=f"{distance}cm - {capacitance}nF"
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Frequency (Hz)", type="log"),
            zaxis=dict(title="|Z| (Ω)", type="log"),
            yaxis=dict(title="Relative distance (cm)")
        ),
        title="3D Impedance Spectra vs Relative Distance",
        legend=dict(itemsizing="constant", font=dict(size=8)),
    )
    return fig


def plot_min_impedance_vs_distance(all_data: pd.DataFrame, steak_size: float
                                     ) -> Tuple[plt.Figure, pd.DataFrame]:
    min_z_by_dist = all_data.groupby('distance')['magnitude'].min()
    min_freq_by_dist = all_data.loc[all_data.groupby('distance')['magnitude'].idxmin()]\
                             .set_index('distance')['frequency[Hz]']
    
    df_results = pd.DataFrame({
        "Distance (cm)": min_z_by_dist.index,
        "Min |Z| (Ω)": min_z_by_dist.values,
        "Frequency at Min (Hz)": min_freq_by_dist.loc[min_z_by_dist.index].values
    })
    
    df_results["Relative Distance"] = df_results["Distance (cm)"] / steak_size
    df_results = df_results.sort_values(by="Relative Distance")

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.plot(df_results["Relative Distance"], df_results["Min |Z| (Ω)"], 
            marker="o", linestyle="-", color="b")
    ax.set_xlabel("Relative Distance (Dcap/Ddrive)", fontsize=12)
    ax.set_ylabel("Minimum |Z| (Ω)", fontsize=12)
    ax.set_title("Minimum Impedance vs Distance", fontsize=14, weight="bold")
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()
    
    return (fig, df_results)


def plot_impedance_range_vs_distance(all_data: pd.DataFrame, steak_size: float
                                      ) -> Tuple[plt.Figure, pd.DataFrame]:
    grouped = all_data.groupby('distance')['magnitude']
    min_z = grouped.min()
    max_z = grouped.max()
    
    df_results = pd.DataFrame({
        "Distance (cm)": min_z.index,
        "Min |Z| (Ω)": min_z.values,
        "Max |Z| (Ω)": max_z.loc[min_z.index].values,
    })
    df_results["Δ|Z| (Ω)"] = df_results["Max |Z| (Ω)"] - df_results["Min |Z| (Ω)"]
    df_results["Relative Distance"] = df_results["Distance (cm)"] / steak_size
    df_results = df_results.sort_values(by="Relative Distance")

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.plot(df_results["Relative Distance"], df_results["Δ|Z| (Ω)"], 
            marker="o", linestyle="-", color="r")
    ax.set_xlabel("Relative distance (Dcap/Ddrive)", fontsize=12)
    ax.set_ylabel("Δ|Z| (Ω)", fontsize=12)
    ax.set_title("Impedance Range (ΔZ) vs Distance", fontsize=14, weight="bold")
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.tight_layout()

    return (fig, df_results)


def plot_3d_impedance_vs_capacitance(all_data: pd.DataFrame, steak_size: float
                                     ) -> Tuple[go.Figure, pd.DataFrame]:
    min_z_data = all_data.loc[all_data.groupby(['distance', 'capacitance'])['magnitude'].idxmin()]
    min_z_data['Relative Distance'] = min_z_data['distance'] / steak_size
    
    fig = go.Figure()
    
    for dist, group in min_z_data.groupby('Relative Distance'):
        group = group.sort_values(by='capacitance')
        
        fig.add_trace(go.Scatter3d(
            x=group['capacitance'], 
            y=group['Relative Distance'], 
            z=group['magnitude'],
            mode="lines+markers", marker=dict(size=4), line=dict(width=2),
            name=f"Rel. Dist: {dist:.2f}",
            hovertemplate="Rel. Dist: %{y:.2f}<br>|Z|: %{z:.2f} Ω<br>Capacitance: %{x} nF<extra></extra>"
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Capacitance (nF)"),
            zaxis=dict(title="Impedance |Z| (Ω)", type="log"),
            yaxis=dict(title="Relative Distance (Dcap/Ddrive)")
        ),
        title="3D Minimum Impedance vs Capacitance vs Distance"
    )
    
    df_results = min_z_data[[
        "Relative Distance", "capacitance", "magnitude"
    ]].rename(columns={
        "capacitance": "Capacitance (nF)",
        "magnitude": "Impedance |Z| (Ω)"
    })
    
    return (fig, df_results)

# ---
# START OF NEW STREAMLIT UI (WEB VERSION)
# ---

# --- 1. Sidebar Configuration ---
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader(
        "Upload your Root Data Folder as a .zip file",
        type="zip"
    )
    
    st.header("2. Configuration")
    steak_size = st.number_input(
        "Steak Size (cm)", 
        min_value=0.1, value=24.0, step=0.1
    )
    z_limit = st.number_input(
        "Z-Limit (Ohm) (0 for no limit)", 
        min_value=0, value=100000, step=1000
    )
    z_limit_val = z_limit if z_limit > 0 else None

    st.header("3. Analysis Steps")
    st.info("Conversion is now automatic.")
    
    run_2d_plots = st.checkbox("2D Plots (per Experiment)", value=True)
    run_3d_dist = st.checkbox("3D Spectra vs. Distance", value=True)
    run_3d_cap = st.checkbox("3D |Z| vs. Capacitance", value=True)
    run_min_z = st.checkbox("Min |Z| vs. Distance", value=True)
    run_delta_z = st.checkbox("Δ|Z| vs. Distance", value=True)

    run_button = st.button(f"**🚀 Run Analysis**")

# --- 4. Main Analysis Area ---
if not uploaded_file:
    st.info("Please upload your data .zip file using the sidebar to begin.")
    st.stop()

if run_button:
    # --- This 'with' block creates a secure, temporary folder ---
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        # 1. Unzip the file
        with st.spinner("Extracting data..."):
            try:
                with zipfile.Zipfile(uploaded_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir_path)
            except Exception as e:
                st.error(f"Error extracting ZIP file: {e}")
                st.stop()
        
        # 2. Find the unzipped root folder
        # (It's often inside another folder named after the zip)
        unzipped_items = list(temp_dir_path.iterdir())
        if len(unzipped_items) == 1 and unzipped_items[0].is_dir():
            root_dir = unzipped_items[0]
        else:
            root_dir = temp_dir_path
            
        st.success(f"Data extracted. Found root folder: `{root_dir.name}`")
        
        # 3. Find the experiment folders inside
        try:
            exp_folders = [f for f in root_dir.iterdir() if f.is_dir() and f.name[0].isdigit()]
            selected_experiments = sorted([f.name for f in exp_folders])
            if not selected_experiments:
                st.error(f"No experiment folders (e.g., '1cm_...') found in the zip file.")
                st.stop()
        except Exception as e:
            st.error(f"Could not read experiment folders from zip: {e}")
            st.stop()

        # --- Step 1: Conversion & 2D Plots (Loop) ---
        st.header("1. 2D Analysis (per Experiment)")
        
        for exp_name in selected_experiments:
            with st.expander(f"▼ Results for: {exp_name}", expanded=True):
                
                exp_path = root_dir / exp_name
                # Create a temporary output path for this loop
                exp_output_path = temp_dir_path / "results" / exp_name
                
                st.markdown(f"**Analyzing:** `{exp_name}`")
                
                try:
                    analyzer = ImpedanceAnalyzer(
                        experiment_dir=exp_path,
                        output_dir=exp_output_path,
                        steak_size=steak_size,
                        z_limit=z_limit_val
                    )
                    
                    # --- Run conversion (this is now mandatory) ---
                    with st.spinner(f"[{exp_name}] Converting .spec to .csv..."):
                        status_msg = analyzer.run_spec_to_csv_conversion()
                        st.write(f"Conversion complete: {status_msg}")

                    # --- Run 2D plots ---
                    if run_2d_plots:
                        st.subheader(f"2D Plots for {exp_name}")
                        with st.spinner(f"[{exp_name}] Generating 2D plots..."):
                            fig_z_freq, fig_z_cap = analyzer.plot_2d_graphs()
                            col1, col2 = st.columns(2)
                            with col1:
                                st.pyplot(fig_z_freq)
                            with col2:
                                st.pyplot(fig_z_cap)
                            plt.close('all')
                            
                except Exception as e:
                    st.error(f"Failed to analyze {exp_name}: {e}")
                    continue
        
        st.success("2D analysis complete.")
        st.markdown("---")

        # --- Step 2: 3D Plots (Combined) ---
        st.header("2. 3D & Summary Analysis (Combined)")
        
        all_data_df = pd.DataFrame()
        if any([run_3d_dist, run_3d_cap, run_min_z, run_delta_z]):
            with st.spinner("Loading all data for 3D/Summary plots..."):
                try:
                    # We pass the temporary root_dir
                    all_data_df = get_all_csv_data(
                        root_dir, selected_experiments, z_limit_val
                    )
                    if all_data_df.empty:
                        st.error("No data found for 3D/Summary plots.")
                    else:
                        st.success(f"Loaded {len(all_data_df)} total data points.")
                except Exception as e:
                    st.error(f"Failed to load combined data: {e}")

        if not all_data_df.empty:
            if run_3d_dist:
                st.subheader("3D Spectra vs. Distance (Plotly)")
                with st.spinner("Generating 3D distance plot..."):
                    try:
                        fig = plot_3d_distance_plotly(all_data_df)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not generate 3D distance plot: {e}")

            if run_min_z:
                st.subheader("Minimum |Z| vs. Distance")
                with st.spinner("Generating Min |Z| plot..."):
                    try:
                        fig, df = plot_min_impedance_vs_distance(all_data_df, steak_size)
                        st.pyplot(fig)
                        with st.expander("View Data"):
                            st.dataframe(df)
                    except Exception as e:
                        st.error(f"Could not generate Min |Z| plot: {e}")

            if run_delta_z:
                st.subheader("Impedance Range (ΔZ) vs. Distance")
                with st.spinner("Generating Δ|Z| plot..."):
                    try:
                        fig, df = plot_impedance_range_vs_distance(all_data_df, steak_size)
                        st.pyplot(fig)
                        with st.expander("View Data"):
                            st.dataframe(df)
                    except Exception as e:
                        st.error(f"Could not generate Δ|Z| plot: {e}")

            if run_3d_cap:
                st.subheader("3D |Z| vs. Capacitance (Plotly)")
                with st.spinner("Generating 3D capacitance plot..."):
                    try:
                        fig, df = plot_3d_impedance_vs_capacitance(all_data_df, steak_size)
                        st.plotly_chart(fig, use_container_width=True)
                        with st.expander("View Raw Data"):
                            st.dataframe(df)
                    except Exception as e:
                        st.error(f"Could not generate 3D capacitance plot: {e}")

    st.success(f"🎉 **Full analysis complete!**")
    st.balloons()
    
    # The temporary directory and all its contents are
    # automatically deleted when the 'with' block ends.



