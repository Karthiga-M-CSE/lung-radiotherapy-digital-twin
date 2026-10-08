# Predictive Digital Twin for Lung Cancer Radiotherapy

A research prototype designed to model, predict, and analyze lung tumor motion during respiration. Because lung tumors move dynamically as a patient breathes, radiotherapy must actively account for this motion to ensure accurate radiation targeting and the preservation of healthy tissue. 

This project establishes an automated pipeline to predict tumor motion, estimate uncertainty, establish safety margins, and automatically trigger treatment replanning alerts when a tumor trajectory deviates beyond expected thresholds.

## Key Features

- Tumor Extraction & Tracking: Processes clinical 4D-CT scans (NIfTI) and tumor masks to extract critical geometric data including the bounding box, volume, and 3D physical centroid coordinates.
- Motion Prediction Models: Uses Ridge Regression and Constant-Velocity Kalman Filters to predict future tumor positions based on historical positional data and external respiratory signals (RPM).
- Safety Gates & Uncertainty Bounds: Calculates empirical 3D prediction errors and establishes robust statistical safety margins (e.g., P95 prediction error) from held-out validation sets.
- Digital Twin State: Synthesizes the ground truth, predictions, errors, and safety checks into a single continuous frame-by-frame diagnostic dataset representing the state of the tumor.
- Replanning Trigger: Acts as a clinical alarm system that flags a treatment for a Replanning Review if the tumor deviates beyond the safety threshold for consecutive frames.
- Dose Impact Analysis: Interfaces with DICOM (RTDOSE and RTSTRUCT) to map active radiation dose delivery against dynamically shifting target boundaries (CTV).
- Interactive Visualization: Features a Streamlit dashboard to dynamically visualize the 4D-CT data, prediction metrics, and the digital twin's status.

## Installation

Ensure you have Python 3.9+ installed. You can install all required dependencies via pip:

pip install -r requirements.txt

## Usage

### 1. Interactive Dashboard
The easiest way to interact with the digital twin system is through the Streamlit web application. Start the app by running:

streamlit run app.py

This will launch a web interface where you can upload 4D-CT scans and corresponding RPM respiratory signals to visualize the predictive models in action.

### 2. Standalone Pipeline Scripts
The project pipeline is modular and can be run step-by-step via the CLI:

- Extract Ground Truth Trajectories: 
  python layer1_tumor_segmentation.py

- Run Ridge Motion Prediction: 
  python respiratory_motion_prediction.py

- Run Kalman Filter Baseline: 
  python kalman_motion_predictor.py

- Calculate Uncertainty & Safety Gates: 
  python uncertainty_safety_gate.py

- Generate Digital Twin State: 
  python digital_twin_state.py

- Evaluate the Replanning Trigger: 
  python replanning_trigger.py

Disclaimer: This is a research prototype only and is not intended for clinical treatment decisions.
