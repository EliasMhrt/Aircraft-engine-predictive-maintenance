import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Boeing & Ethiopian Airlines - AI Predictive Maintenance",
    page_icon="✈️",
    layout="wide"
)

# --- 2. CRASH-PROOF MODEL LOADER ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('xgboost_rul_model.pkl')
    except Exception:
        return None

model = load_model()

# Stop execution gracefully if the model file is missing
if model is None:
    st.error("⚠️ Model file 'xgboost_rul_model.pkl' not found. Please ensure it is in the same folder as this script.")
    st.stop()

# --- 3. HEADER SECTION ---
st.title("✈️ Aircraft Engine Predictive Maintenance System")
st.caption("Developed by Boeing & Ethiopian Airlines Program Alumni | Powered by NASA C-MAPSS Telemetry & XGBoost AI")
st.markdown("---")

# --- 4. SIDEBAR - ALL 24 TELEMETRY CONTROLS ---
st.sidebar.header("🎛️ Live Flight Telemetry Controls")

# --- QUICK PRESETS FOR LIVE DEMONSTRATIONS ---
st.sidebar.subheader("⚡ Quick Demo Presets")
preset_col1, preset_col2, preset_col3 = st.sidebar.columns(3)

# Initialize session state safely
if 'preset' not in st.session_state:
    st.session_state.preset = "healthy"

if preset_col1.button("🟢 Healthy"):
    st.session_state.preset = "healthy"
if preset_col2.button("🟡 Mid-Life"):
    st.session_state.preset = "midlife"
if preset_col3.button("🔴 Critical"):
    st.session_state.preset = "critical"

# Define profile values based on active preset
if st.session_state.preset == "healthy":
    d_s1, d_s2, d_s3 = 0.0000, 0.0000, 100.0
    d_sens = [518.67, 642.38, 1586.84, 1402.76, 14.62, 21.61, 553.95, 2388.06, 9056.01, 1.30,
              47.35, 521.91, 2388.06, 8137.37, 8.4183, 0.03, 392.0, 2388.0, 100.0, 38.93, 23.36]
elif st.session_state.preset == "midlife":
    d_s1, d_s2, d_s3 = 0.0005, -0.0002, 100.0
    d_sens = [518.67, 642.90, 1594.00, 1414.00, 14.62, 21.61, 552.80, 2388.14, 9075.00, 1.30,
              47.70, 520.90, 2388.14, 8150.00, 8.4600, 0.03, 394.0, 2388.0, 100.0, 38.70, 23.22]
else: # critical
    d_s1, d_s2, d_s3 = -0.0010, 0.0003, 100.0
    d_sens = [518.67, 643.60, 1601.50, 1427.10, 14.62, 21.61, 551.60, 2388.22, 9095.00, 1.30,
              48.10, 519.90, 2388.22, 8165.00, 8.5100, 0.03, 396.0, 2388.0, 100.0, 38.48, 23.08]

# --- CATEGORY 1: OPERATIONAL SETTINGS ---
with st.sidebar.expander("⚙️ Operational Settings", expanded=False):
    s_setting_1 = st.slider("Setting 1 (Altitude/Mach)", -0.0100, 0.0100, float(d_s1), step=0.0001)
    s_setting_2 = st.slider("Setting 2 (Throttle)", -0.0010, 0.0010, float(d_s2), step=0.0001)
    s_setting_3 = st.slider("Setting 3 (TRA Demand)", 90.0, 110.0, float(d_s3), step=0.5)

# --- CATEGORY 2: TEMPERATURE SENSORS ---
with st.sidebar.expander("🔥 Temperature Telemetry", expanded=True):
    s_sensor_1 = st.slider("Sensor 1: Total Temp Fan Inlet [°R]", 515.0, 522.0, float(d_sens[0]), step=0.1)
    s_sensor_2 = st.slider("Sensor 2: Total Temp LPC Outlet [°R]", 640.0, 646.0, float(d_sens[1]), step=0.05)
    s_sensor_3 = st.slider("Sensor 3: Total Temp HPC Outlet [°R]", 1570.0, 1620.0, float(d_sens[2]), step=0.5)
    s_sensor_4 = st.slider("Sensor 4: Total Temp LPT Outlet (EGT) [°R]", 1380.0, 1445.0, float(d_sens[3]), step=0.5)

# --- CATEGORY 3: PRESSURE SENSORS ---
with st.sidebar.expander("💨 Pressure Telemetry", expanded=True):
    s_sensor_5 = st.slider("Sensor 5: Pressure at Fan Inlet [psia]", 14.0, 15.5, float(d_sens[4]), step=0.05)
    s_sensor_6 = st.slider("Sensor 6: Bypass Duct Pressure [psia]", 21.50, 21.70, float(d_sens[5]), step=0.01)
    s_sensor_7 = st.slider("Sensor 7: HPC Outlet Pressure [psia]", 548.0, 558.0, float(d_sens[6]), step=0.2)
    s_sensor_10 = st.slider("Sensor 10: Engine Pressure Ratio", 1.20, 1.40, float(d_sens[9]), step=0.01)
    s_sensor_11 = st.slider("Sensor 11: HPC Static Pressure [psia]", 46.5, 48.8, float(d_sens[10]), step=0.05)

# --- CATEGORY 4: ROTATIONAL SPEED & FLOW ---
with st.sidebar.expander("🌀 Speed & Flow Telemetry", expanded=True):
    s_sensor_8 = st.slider("Sensor 8: Physical Fan Speed [RPM]", 2387.5, 2388.8, float(d_sens[7]), step=0.02)
    s_sensor_9 = st.slider("Sensor 9: Physical Core Speed [RPM]", 9020.0, 9220.0, float(d_sens[8]), step=1.0)
    s_sensor_12 = st.slider("Sensor 12: Fuel Flow Ratio [pps/psia]", 518.0, 524.0, float(d_sens[11]), step=0.1)
    s_sensor_13 = st.slider("Sensor 13: Corrected Fan Speed [RPM]", 2387.5, 2388.8, float(d_sens[12]), step=0.02)
    s_sensor_14 = st.slider("Sensor 14: Corrected Core Speed [RPM]", 8090.0, 8220.0, float(d_sens[13]), step=1.0)
    s_sensor_15 = st.slider("Sensor 15: Bypass Ratio", 8.30, 8.60, float(d_sens[14]), step=0.005)

# --- CATEGORY 5: BLEED AIR & SYSTEM CONSTANTS ---
with st.sidebar.expander("🩸 Bleed Air & System Ratios", expanded=False):
    s_sensor_16 = st.slider("Sensor 16: Fuel-Air Ratio", 0.02, 0.04, float(d_sens[15]), step=0.005)
    s_sensor_17 = st.slider("Sensor 17: Bleed Enthalpy [BTU/lb]", 385.0, 402.0, float(d_sens[16]), step=1.0)
    s_sensor_18 = st.slider("Sensor 18: Demanded Fan Speed", 2380.0, 2390.0, float(d_sens[17]), step=1.0)
    s_sensor_19 = st.slider("Sensor 19: Demanded Corrected Fan Speed", 95.0, 105.0, float(d_sens[18]), step=0.5)
    s_sensor_20 = st.slider("Sensor 20: HPT Coolant Bleed [lb/s]", 38.0, 39.8, float(d_sens[19]), step=0.02)
    s_sensor_21 = st.slider("Sensor 21: LPT Coolant Bleed [lb/s]", 22.8, 23.8, float(d_sens[20]), step=0.02)

# --- 5. ASSEMBLE ALL 24 INPUTS INTO DATAFRAME ---
input_df = pd.DataFrame([{
    'setting_1': s_setting_1, 'setting_2': s_setting_2, 'setting_3': s_setting_3,
    'sensor_1': s_sensor_1,   'sensor_2': s_sensor_2,   'sensor_3': s_sensor_3,   'sensor_4': s_sensor_4,
    'sensor_5': s_sensor_5,   'sensor_6': s_sensor_6,   'sensor_7': s_sensor_7,   'sensor_8': s_sensor_8,
    'sensor_9': s_sensor_9,   'sensor_10': s_sensor_10, 'sensor_11': s_sensor_11, 'sensor_12': s_sensor_12,
    'sensor_13': s_sensor_13, 'sensor_14': s_sensor_14, 'sensor_15': s_sensor_15, 'sensor_16': s_sensor_16,
    'sensor_17': s_sensor_17, 'sensor_18': s_sensor_18, 'sensor_19': s_sensor_19, 'sensor_20': s_sensor_20,
    'sensor_21': s_sensor_21
}])

# --- 6. AI PREDICTION ---
predicted_rul = float(model.predict(input_df)[0])
predicted_rul = max(0.0, predicted_rul)  # Clamp to 0 min

# --- 7. MAIN DISPLAY DASHBOARD ---
col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("📊 Model Output & Lifecycle Prediction")
    st.metric(
        label="Predicted Remaining Useful Life (RUL)",
        value=f"{int(predicted_rul)} Flight Cycles",
        delta=f"{int(predicted_rul - 125)} cycles vs nominal average"
    )
    
    # Life gauge bar
    health_percentage = max(0, min(100, int((predicted_rul / 150.0) * 100))) # Extra clamp for UI safety
    st.write(f"**Engine Health Reserve:** {health_percentage}%")
    st.progress(health_percentage / 100.0)

with col_right:
    st.subheader("🛡️ Operational Health Status")
    if predicted_rul > 60:
        st.success("""
        ### 🟢 STATUS: NORMAL OPERATIONS
        * **Recommendation:** Engine healthy. Flight operations cleared for international routes.
        * **Next Scheduled Inspection:** > 60 Flight Cycles.
        """)
    elif 20 <= predicted_rul <= 60:
        st.warning("""
        ### 🟡 STATUS: MAINTENANCE WARNING
        * **Recommendation:** Thermal & Pressure stress detected. Schedule overhaul inspection at **Addis Ababa Maintenance Hub**.
        * **Action Required:** Monitor core speed and exhaust gas temperatures closely.
        """)
    else:
        st.error("""
        ### 🔴 STATUS: CRITICAL FAILURE RISK
        * **Recommendation:** High probability of imminent component failure.
        * **Action Required:** Ground aircraft immediately. Perform high-pressure turbine overhaul.
        """)

st.markdown("---")

# --- 8. LIVE SENSOR MONITORING TABLE ---
st.subheader("🔬 Real-time Telemetry Breakdown (24 Parameters)")

# Calculate variance from standard healthy baseline
baseline_healthy = [
    0.0, 0.0, 100.0, 518.67, 642.38, 1586.84, 1402.76, 14.62, 21.61, 553.95, 
    2388.06, 9056.01, 1.30, 47.35, 521.91, 2388.06, 8137.37, 8.4183, 0.03, 
    392.0, 2388.0, 100.0, 38.93, 23.36
]

current_vals = input_df.iloc[0].values
deviations = [round(c - b, 4) for c, b in zip(current_vals, baseline_healthy)]

telemetry_summary = pd.DataFrame({
    "Parameter / Sensor": input_df.columns,
    "Current Live Value": [round(v, 4) for v in current_vals],
    "Nominal Baseline": baseline_healthy,
    "Deviation from Nominal": deviations
})

st.dataframe(telemetry_summary, use_container_width=True, hide_index=True)