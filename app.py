import streamlit as st
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import tempfile
import os

st.set_page_config(page_title="Predictive Digital Twin", layout="wide", initial_sidebar_state="expanded")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.title("Digital Twin Setup")
    st.caption("Research prototype only — not for clinical use.")
    
    st.subheader("Data Upload")
    ct_file = st.file_uploader("Upload 4D-CT scan (.nii or .nii.gz)", type=["nii", "gz"])
    rpm_file = st.file_uploader("Upload Respiratory Signal (RPM)", type=["txt", "csv"])

# --- INITIAL STATE ---
if ct_file is None or rpm_file is None:
    st.title("Predictive Digital Twin for Lung Cancer Radiotherapy")
    st.info("👋 Please upload both the 4D-CT scan and the RPM signal in the sidebar to proceed.")
    st.stop()

# --- DATA LOADING (CACHED) ---
@st.cache_data
def load_ct_data(uploaded_file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
        tmp.write(uploaded_file_bytes)
        ct_path = tmp.name
    nii = nib.load(ct_path)
    return nii.get_fdata()

@st.cache_data
def load_rpm_data(uploaded_file_bytes):
    text = uploaded_file_bytes.decode("utf-8")
    rpm_values = []
    for line in text.replace(",", "\n").splitlines():
        line = line.strip()
        if not line: continue
        try: rpm_values.append(float(line))
        except ValueError: continue
    return np.asarray(rpm_values, dtype=float)

try:
    ct_data = load_ct_data(ct_file.getvalue())
except Exception as e:
    st.error(f"Could not read the 4D-CT file: {e}")
    st.stop()

try:
    rpm = load_rpm_data(rpm_file.getvalue())
    if len(rpm) == 0: raise ValueError("No numerical RPM values found.")
except Exception as e:
    st.error(f"Could not read RPM file: {e}")
    st.stop()

shape = ct_data.shape
if len(shape) != 4:
    st.warning("File is not a 4D volume (Expected dimensions: X × Y × Z × phases).")
    st.stop()

n_phases = shape[3]

# --- SIDEBAR PHASE SELECTION ---
with st.sidebar:
    st.divider()
    st.subheader("Phase Selection")
    phase_display = st.selectbox(
        "Select Respiratory Phase",
        options=list(range(1, n_phases + 1)),
        index=0
    )
    phase = phase_display - 1

# --- EXTRACT CURRENT STATE ---
current_ct = ct_data[:, :, :, phase]
z_slice = current_ct.shape[2] // 2
axial = current_ct[:, :, z_slice]

if len(rpm) == n_phases:
    current_rpm = rpm[phase]
else:
    rpm_index = int(phase * (len(rpm) - 1) / max(n_phases - 1, 1))
    current_rpm = rpm[rpm_index]

# --- MAIN DASHBOARD CONTENT ---
st.title("Predictive Digital Twin Dashboard")

# 1. Top Metrics Row
cols = st.columns(4)
cols[0].metric("Volume Dimensions", f"{shape[0]}x{shape[1]}x{shape[2]}")
cols[1].metric("Respiratory Phases", shape[3])
cols[2].metric("Current Phase", f"{phase + 1} / {n_phases}")
cols[3].metric("RPM Signal", f"{current_rpm:.2f}")

# 2. Tumor Logic Computation
threshold = np.percentile(current_ct, 99)
tumor_candidate = current_ct >= threshold
coords = np.argwhere(tumor_candidate)
if len(coords) > 0:
    tumor_center = coords.mean(axis=0)
    tumor_x, tumor_y, tumor_z = tumor_center
else:
    tumor_x, tumor_y, tumor_z = current_ct.shape[0]/2, current_ct.shape[1]/2, current_ct.shape[2]/2

reference_ct = ct_data[:, :, :, 0]
reference_threshold = np.percentile(reference_ct, 99)
reference_coords = np.argwhere(reference_ct >= reference_threshold)
if len(reference_coords) > 0:
    reference_center = reference_coords.mean(axis=0)
else:
    reference_center = np.array([reference_ct.shape[0]/2, reference_ct.shape[1]/2, reference_ct.shape[2]/2])

displacement_voxel = np.array([tumor_x, tumor_y, tumor_z]) - reference_center
voxel_spacing_mm = np.array([3.0, 3.0, 3.0])
displacement_mm = displacement_voxel * voxel_spacing_mm
motion_magnitude = np.linalg.norm(displacement_mm)
RESEARCH_THRESHOLD_MM = 3.4304

# Simulated Dose Computation
xx, yy = np.meshgrid(np.arange(current_ct.shape[0]), np.arange(current_ct.shape[1]), indexing="ij")
sigma = 5.0
dose = np.exp(-((xx - tumor_x)**2 + (yy - tumor_y)**2) / (2 * sigma**2))
dose = dose / np.max(dose) * 100

st.divider()

# 3. Visualizations Row
col_plot1, col_plot2 = st.columns(2)

with col_plot1:
    st.subheader("Current Phase 4D-CT")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(axial.T, cmap="gray", origin="lower")
    ax.set_title(f"Phase {phase + 1} Anatomy")
    ax.axis('off')
    st.pyplot(fig)
    plt.close(fig)

with col_plot2:
    st.subheader("Simulated Dose Overlay")
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.imshow(axial.T, cmap="gray", origin="lower")
    im = ax2.imshow(dose.T, cmap="jet", alpha=0.45, origin="lower", vmin=0, vmax=100)
    ax2.scatter(tumor_y, tumor_x, marker="x", color="white", s=80, linewidths=2)
    ax2.set_title("Target & Dose Field")
    ax2.axis('off')
    st.pyplot(fig2)
    plt.close(fig2)

st.divider()

# 4. Motion Analysis & Safety Status
col_motion, col_safety = st.columns(2)

with col_motion:
    st.subheader("Predicted Tumor Motion")
    mc1, mc2 = st.columns(2)
    mc1.metric("X Displacement", f"{displacement_mm[0]:.2f} mm")
    mc1.metric("Y Displacement", f"{displacement_mm[1]:.2f} mm")
    mc2.metric("Z Displacement", f"{displacement_mm[2]:.2f} mm")
    mc2.metric("3D Displacement", f"{motion_magnitude:.2f} mm")

with col_safety:
    st.subheader("Digital Twin Status")
    st.write(f"**Research Uncertainty Bound:** {RESEARCH_THRESHOLD_MM:.2f} mm")
    
    if motion_magnitude <= RESEARCH_THRESHOLD_MM:
        st.success("✅ Prediction is within the safety threshold.")
        st.info("No replanning review triggered for this phase.")
    else:
        st.error("⚠️ Prediction exceeds the safety threshold!")
        st.warning("Clinical review flagged for potential anatomical deviation.")

st.caption("Predictive Digital Twin for Lung Cancer Radiotherapy | Research prototype | Synthetic demonstration | Not for clinical use")
