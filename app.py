import streamlit as st
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import tempfile
import os

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Predictive Digital Twin",
    layout="wide"
)

# ============================================================
# SIMPLE ACADEMIC STYLE
# ============================================================

st.markdown("""
<style>
    .main {
        background-color: white;
    }

    h1, h2, h3 {
        color: #123B63;
    }

    .section {
        border: 1px solid #D5DCE3;
        padding: 18px;
        margin-top: 15px;
        margin-bottom: 15px;
        background-color: #FAFBFC;
    }

    .status {
        padding: 10px;
        border: 1px solid #D5DCE3;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.title("Predictive Digital Twin for Lung Cancer Radiotherapy")

st.write(
    "4D-CT motion prediction and research dose-overlay prototype"
)

st.info(
    "Research prototype only — not intended for clinical treatment decisions."
)


# ============================================================
# INPUT
# ============================================================

st.header("1. Input Data")

ct_file = st.file_uploader(
    "Upload 4D-CT scan",
    type=["nii", "gz"],
    help="Upload a 4D NIfTI lung CT scan (.nii or .nii.gz)."
)

rpm_file = st.file_uploader(
    "Upload respiratory signal (RPM)",
    type=["txt", "csv"],
    help="Upload the respiratory signal corresponding to the 4D-CT."
)


# ============================================================
# WAIT UNTIL FILES ARE PROVIDED
# ============================================================

if ct_file is None or rpm_file is None:

    st.markdown("""
    <div class="section">

    ### Required inputs

    **4D-CT**
    
    A four-dimensional CT scan containing multiple respiratory phases.

    **RPM signal**
    
    A respiratory waveform corresponding to the CT acquisition.

    Once both files are uploaded, the system will begin processing.

    </div>
    """, unsafe_allow_html=True)

    st.stop()


# ============================================================
# LOAD 4D CT
# ============================================================

st.header("2. Loading 4D-CT")

try:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".nii.gz"
    ) as tmp:

        tmp.write(ct_file.getbuffer())
        ct_path = tmp.name

    nii = nib.load(ct_path)

    ct_data = nii.get_fdata()

    st.success("4D-CT loaded successfully.")

except Exception as e:

    st.error(f"Could not read the 4D-CT file: {e}")
    st.stop()


# ============================================================
# CHECK DIMENSIONS
# ============================================================

st.subheader("4D-CT Information")

shape = ct_data.shape

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Dimensions", str(shape))

with col2:
    if len(shape) >= 4:
        st.metric("Respiratory phases", shape[3])
    else:
        st.metric("Respiratory phases", "1")

with col3:
    st.metric("Minimum HU", f"{np.min(ct_data):.1f}")

with col4:
    st.metric("Maximum HU", f"{np.max(ct_data):.1f}")


# ============================================================
# CHECK 4D
# ============================================================

if len(shape) != 4:

    st.warning(
        "This file does not appear to contain a 4D volume. "
        "Expected dimensions such as X × Y × Z × phases."
    )

    st.stop()


# ============================================================
# LOAD RPM
# ============================================================

st.header("3. Respiratory Signal")

try:

    rpm_bytes = rpm_file.getvalue()

    text = rpm_bytes.decode("utf-8")

    rpm_values = []

    for line in text.replace(",", "\n").splitlines():

        line = line.strip()

        if not line:
            continue

        try:
            rpm_values.append(float(line))
        except ValueError:
            continue

    rpm = np.asarray(rpm_values, dtype=float)

    if len(rpm) == 0:
        raise ValueError("No numerical RPM values found.")

    st.success(
        f"RPM signal loaded: {len(rpm)} samples"
    )

except Exception as e:

    st.error(f"Could not read RPM file: {e}")
    st.stop()


# ============================================================
# PHASE SELECTION
# ============================================================

st.header("4. Respiratory Phase")

n_phases = ct_data.shape[3]

phase = st.slider(
    "Select respiratory phase",
    min_value=0,
    max_value=n_phases - 1,
    value=0,
    step=1
)


# ============================================================
# GET CURRENT 3D CT
# ============================================================

current_ct = ct_data[:, :, :, phase]

# Middle axial slice
z_slice = current_ct.shape[2] // 2

axial = current_ct[:, :, z_slice]


# ============================================================
# RPM MAPPING
# ============================================================

if len(rpm) == n_phases:

    current_rpm = rpm[phase]

else:

    # Map CT phase to closest RPM sample
    rpm_index = int(
        phase * (len(rpm) - 1) / max(n_phases - 1, 1)
    )

    current_rpm = rpm[rpm_index]


# ============================================================
# DISPLAY CT
# ============================================================

st.subheader("Current 4D-CT Phase")

fig, ax = plt.subplots(figsize=(8, 6))

ax.imshow(
    axial.T,
    cmap="gray",
    origin="lower"
)

ax.set_title(
    f"Respiratory Phase {phase + 1}/{n_phases} | "
    f"RPM = {current_rpm:.2f}"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")

st.pyplot(fig)

plt.close(fig)


# ============================================================
# SYNTHETIC TUMOR LOCALIZATION
# ============================================================

st.header("5. Tumor Motion Analysis")

st.caption(
    "For this prototype, tumor localization is demonstrated using "
    "a high-density synthetic target region. A clinical implementation "
    "would replace this with a trained tumor segmentation model."
)


# Simple intensity-based candidate
# Lung CT is generally low density; synthetic tumor is higher density.

threshold = np.percentile(current_ct, 99)

tumor_candidate = current_ct >= threshold

coords = np.argwhere(tumor_candidate)


if len(coords) > 0:

    tumor_center = coords.mean(axis=0)

    tumor_x = tumor_center[0]
    tumor_y = tumor_center[1]
    tumor_z = tumor_center[2]

else:

    tumor_x = current_ct.shape[0] / 2
    tumor_y = current_ct.shape[1] / 2
    tumor_z = current_ct.shape[2] / 2


# ============================================================
# MOTION ESTIMATION
# ============================================================

# Compare tumor candidate center with first phase.

reference_ct = ct_data[:, :, :, 0]

reference_threshold = np.percentile(
    reference_ct,
    99
)

reference_candidate = reference_ct >= reference_threshold

reference_coords = np.argwhere(
    reference_candidate
)

if len(reference_coords) > 0:

    reference_center = reference_coords.mean(axis=0)

else:

    reference_center = np.array([
        reference_ct.shape[0] / 2,
        reference_ct.shape[1] / 2,
        reference_ct.shape[2] / 2
    ])


displacement_voxel = (
    np.array([tumor_x, tumor_y, tumor_z])
    - reference_center
)


# Assume synthetic CT has 3 mm voxel spacing
voxel_spacing_mm = np.array([3.0, 3.0, 3.0])

displacement_mm = (
    displacement_voxel * voxel_spacing_mm
)

motion_magnitude = np.linalg.norm(
    displacement_mm
)


# ============================================================
# PREDICTION
# ============================================================

st.subheader("Predicted Tumor Motion")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "X displacement",
        f"{displacement_mm[0]:.2f} mm"
    )

with c2:
    st.metric(
        "Y displacement",
        f"{displacement_mm[1]:.2f} mm"
    )

with c3:
    st.metric(
        "Z displacement",
        f"{displacement_mm[2]:.2f} mm"
    )

with c4:
    st.metric(
        "3D displacement",
        f"{motion_magnitude:.2f} mm"
    )


# ============================================================
# UNCERTAINTY
# ============================================================

RESEARCH_THRESHOLD_MM = 3.4304

prediction_uncertainty = min(
    motion_magnitude,
    RESEARCH_THRESHOLD_MM
)


st.subheader("Prediction Uncertainty")

st.write(
    f"Research uncertainty reference: "
    f"**{RESEARCH_THRESHOLD_MM:.2f} mm**"
)

if motion_magnitude <= RESEARCH_THRESHOLD_MM:

    safety_status = "CONTINUE"

    st.success(
        "Prediction is within the research uncertainty bound."
    )

else:

    safety_status = "REVIEW"

    st.warning(
        "Prediction exceeds the research uncertainty bound."
    )


# ============================================================
# DOSE OVERLAY
# ============================================================

st.header("6. Research Dose Distribution")

st.caption(
    "The dose field below is simulated for visualization. "
    "It is not a clinical RT dose calculation."
)


# Create Gaussian dose distribution around predicted tumor.

xx, yy = np.meshgrid(
    np.arange(current_ct.shape[0]),
    np.arange(current_ct.shape[1]),
    indexing="ij"
)

sigma = 5.0

dose = np.exp(
    -(
        (xx - tumor_x) ** 2
        +
        (yy - tumor_y) ** 2
    )
    /
    (2 * sigma ** 2)
)

dose = dose / np.max(dose) * 100


# ============================================================
# CT + DOSE
# ============================================================

fig2, ax2 = plt.subplots(figsize=(9, 7))

ax2.imshow(
    axial.T,
    cmap="gray",
    origin="lower"
)

im = ax2.imshow(
    dose.T,
    cmap="jet",
    alpha=0.45,
    origin="lower",
    vmin=0,
    vmax=100
)

ax2.scatter(
    tumor_y,
    tumor_x,
    marker="x",
    s=100,
    linewidths=2
)

ax2.set_title(
    "4D-CT with Predicted Tumor and Simulated Dose Overlay"
)

ax2.set_xlabel("X")
ax2.set_ylabel("Y")

plt.colorbar(
    im,
    ax=ax2,
    label="Relative Dose (%)"
)

st.pyplot(fig2)

plt.close(fig2)


# ============================================================
# DIGITAL TWIN STATE
# ============================================================

st.header("7. Digital Twin State")

col1, col2 = st.columns(2)

with col1:

    st.write("**Current respiratory phase**")
    st.write(f"{phase + 1}/{n_phases}")

    st.write("**Respiratory signal**")
    st.write(f"{current_rpm:.2f}")

    st.write("**Predicted motion**")
    st.write(f"{motion_magnitude:.2f} mm")


with col2:

    st.write("**Uncertainty reference**")
    st.write(f"{RESEARCH_THRESHOLD_MM:.2f} mm")

    st.write("**Safety state**")
    st.write(safety_status)

    if safety_status == "REVIEW":

        st.warning("Clinical review flag")

    else:

        st.success("Within research threshold")


# ============================================================
# REPLANNING LOGIC
# ============================================================

st.header("8. Replanning Review")

if motion_magnitude > RESEARCH_THRESHOLD_MM:

    st.warning(
        "Potential anatomical/motion deviation detected. "
        "Flag for replanning review."
    )

else:

    st.success(
        "No replanning review triggered for this phase."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Predictive Digital Twin for Lung Cancer Radiotherapy | "
    "Research prototype | Synthetic 4D-CT demonstration | "
    "Not for clinical use"
)