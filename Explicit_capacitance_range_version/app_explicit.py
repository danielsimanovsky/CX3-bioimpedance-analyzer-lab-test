import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
import zipfile
import tempfile

try:
    from analysis_explicit import ImpedanceAnalyzer, plot_timeseries_interactive, plot_spectrum_static
except ImportError:
    st.error("Missing 'analysis.py'. Please make sure it's in the same folder.")
    st.stop()

st.set_page_config(page_title="Impedance Analysis Dashboard", page_icon="🔬", layout="wide")
st.title("🔬 Impedance Analysis Dashboard")
st.markdown("Upload your `.zip` file, load the data once, and dynamically change limits/plots instantly.")

# --- INITIALIZE SESSION STATE ---
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None
    st.session_state.active_mode = None

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("1. Data Loading")
    uploaded_file = st.file_uploader("Upload Root Data Folder (.zip)", type="zip")

    data_mode = st.radio(
        "Select Dataset Type:",
        ("Time-Series (Single Freq monitoring)", "Spectrum (Explicit Cap sweeps)")
    )

    load_button = st.button("**🚀 Extract & Load Data**", help="Click once to load data into memory.")

    st.markdown("---")
    st.header("2. Live Plot Controls")
    st.caption("Changing these instantly updates the graphs.")

    # Show explicit capacitance settings only if Spectrum is selected
    cap_values_list = []
    target_freq_mhz = 1.0
    if data_mode == "Spectrum (Explicit Cap sweeps)":
        st.subheader("Capacitance Setup")
        default_vals = "0,1,2,3,4,5,6,7,8,9,10,100,200,300,400,500,600,700,800,900,1000"
        cap_values_input = st.text_area("Capacitance Values (pF) - Comma Separated", value=default_vals, height=100)
        try:
            cap_values_list = [float(x.strip()) for x in cap_values_input.split(',') if x.strip()]
        except ValueError:
            st.error("Invalid format in Capacitance Values.")

        target_freq_mhz = st.number_input("Target Frequency (MHz)", value=1.0, step=0.1, min_value=0.0)

    coil_inductance = st.number_input("Coil Inductance (nH)", value=1.0, step=0.1, min_value=0.001, format="%.3f")
    z_limit = st.number_input("Z-Limit (Ohm) (0 for no limit)", min_value=0, value=100000, step=1000)
    z_limit_val = z_limit if z_limit > 0 else None

    plot_type_str = st.radio(
        "Select Variable to Plot:",
        ('|Z| (Magnitude)', '|Z| (Parallel Model)', 'Re(Z) (Real)', 'Im(Z) (Imaginary)'),
        index=0
    )

    plot_type_map = {
        '|Z| (Magnitude)': 'MagZ',
        '|Z| (Parallel Model)': 'MagZ_Parallel',
        'Re(Z) (Real)': 'ReZ',
        'Im(Z) (Imaginary)': 'ImZ'
    }
    plot_type_2d = plot_type_map[plot_type_str]

# --- STEP 1: LOAD & EXTRACT DATA ---
if load_button and uploaded_file:
    with st.spinner("Extracting & parsing files... (This only happens once)"):
        memory_data = {}
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir_path)

            unzipped_items = list(temp_dir_path.iterdir())
            root_dir = unzipped_items[0] if len(unzipped_items) == 1 and unzipped_items[0].is_dir() else temp_dir_path

            exp_paths = []
            if list(root_dir.glob("*.spec")): exp_paths.append(root_dir)
            for f in root_dir.iterdir():
                if f.is_dir() and list(f.rglob("*.spec")) and f not in exp_paths:
                    exp_paths.append(f)

            if not exp_paths:
                st.error("No valid folders containing .spec files were found in the uploaded zip.")
                st.stop()

            # Parse each folder and save dataframe to memory
            for exp_path in exp_paths:
                analyzer = ImpedanceAnalyzer(experiment_dir=exp_path, coil_inductance=coil_inductance)

                if data_mode == "Time-Series (Single Freq monitoring)":
                    memory_data[exp_path.name] = analyzer.extract_timeseries_data()
                else:
                    memory_data[exp_path.name] = analyzer.extract_spectrum_data(cap_values_list)

        st.session_state.extracted_data = memory_data
        st.session_state.active_mode = data_mode
        st.success("Data successfully loaded into memory!")

# --- STEP 2: DYNAMICALLY RENDER GRAPHS ---
if st.session_state.extracted_data:
    st.header(f"Results: {st.session_state.active_mode}")

    # Check if user flipped the radio button after loading data
    if st.session_state.active_mode != data_mode:
        st.warning("You changed the Dataset Type. Please click 'Extract & Load Data' again to re-parse the files.")

    else:
        for exp_name, df in st.session_state.extracted_data.items():
            if df.empty:
                st.error(f"Failed to load valid data for {exp_name}.")
                continue

            with st.expander(f"▼ Live Graphs for: {exp_name}", expanded=True):

                if st.session_state.active_mode == "Time-Series (Single Freq monitoring)":
                    fig = plot_timeseries_interactive(df, exp_name, plot_type_2d, z_limit_val)
                    st.plotly_chart(fig, use_container_width=True)

                    st.download_button(
                        label=f"📥 Download Time-Series CSV ({exp_name})",
                        data=df.to_csv(index=False).encode('utf-8'),
                        file_name=f"{exp_name}_timeseries.csv", mime='text/csv', key=f"dl_ts_{exp_name}"
                    )

                else:
                    # Spectrum logic
                    fig1, fig2, fig3, fig4, df_sum = plot_spectrum_static(df, exp_name, plot_type_2d, target_freq_mhz,
                                                                          z_limit_val)

                    st.pyplot(fig1)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(fig2)
                    with col2:
                        st.pyplot(fig3)
                    st.pyplot(fig4)

                    st.download_button(
                        label=f"📥 Download Analysis Summary ({exp_name})",
                        data=df_sum.to_csv(index=False).encode('utf-8'),
                        file_name=f"{exp_name}_analysis_summary.csv", mime='text/csv', key=f"dl_sum_{exp_name}"
                    )
