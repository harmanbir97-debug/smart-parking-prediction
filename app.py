import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="ParkWise | Smart Parking",
    page_icon="🚗",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "smart_parking_model.pkl"

@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

bundle = load_bundle()

st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 2rem;}
.hero {
    padding: 1.6rem 1.8rem; border-radius: 18px;
    background: linear-gradient(120deg, #102b3f, #175b63);
    color: white; margin-bottom: 1.4rem;
}
.hero h1 {color: white; margin: 0 0 .4rem 0; font-size: 2.1rem;}
.hero p {color: #d7e9e9; margin: 0;}
.status-available {
    padding: 1rem; border-radius: 12px; background: #e4f7ec;
    color: #126337; font-size: 1.1rem; font-weight: 700;
}
.status-occupied {
    padding: 1rem; border-radius: 12px; background: #fde9e7;
    color: #a12620; font-size: 1.1rem; font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🚗 ParkWise")
    st.caption("Smart Parking Prediction")
    st.divider()
    st.markdown("**ML model**")
    st.write("Polynomial Ridge Regression")
    st.write("Polynomial degree: 3")
    st.divider()
    st.caption("Academic prototype — predictions use the supplied dataset, not live sensors.")

st.markdown("""
<div class="hero">
  <h1>Smart Parking Dashboard</h1>
  <p>Estimate an individual parking slot's availability using its slot ID and time details.</p>
</div>
""", unsafe_allow_html=True)

if bundle is None:
    st.error("Model file not found.")
    st.markdown(
        "Put **smart_parking_model.pkl** in the same folder as `app.py`. "
        "Export it from your Colab notebook using the setup code provided with this project."
    )
    st.stop()

if not isinstance(bundle, dict) or "model" not in bundle or "feature_cols" not in bundle:
    st.error("The saved model is missing feature metadata. Please export the model bundle, not just the estimator.")
    st.stop()

model = bundle["model"]
feature_cols = bundle["feature_cols"]
slot_ids = sorted(int(x) for x in bundle.get("slot_ids", range(1, 51)))

st.subheader("Make a prediction")
st.write("Select a parking slot and time details, then run the trained regression model.")

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("#### Parking details")
    slot_id = st.selectbox("Parking slot ID", slot_ids)
    hour = st.slider("Hour of day", 0, 23, 14, help="0 = midnight, 12 = noon, 23 = 11 PM.")
    day = st.selectbox(
        "Day value (as encoded in the dataset)",
        list(range(7)),
        help="The dataset represents days as numbers from 0 to 6."
    )
    month = st.selectbox("Month", list(range(1, 13)), index=5)
    predict_clicked = st.button("Predict availability", type="primary", use_container_width=True)

with right:
    st.markdown("#### About the prediction")
    st.markdown("""
    - The score is the model's continuous output, displayed on a 0–100 scale.
    - A score of **50% or above** is labelled *Available*.
    - A score below **50%** is labelled *Occupied*.
    """)
    st.info("This is a dataset-based estimate for one slot, not a live sensor reading or a guarantee of availability.")

if predict_clicked:
    input_row = pd.DataFrame(0.0, index=[0], columns=feature_cols, dtype=float)
    time_features = {
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "day_sin": np.sin(2 * np.pi * day / 7),
        "day_cos": np.cos(2 * np.pi * day / 7),
        "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
    }

    missing = [name for name in time_features if name not in input_row.columns]
    if missing:
        st.error(f"Model feature metadata is missing: {', '.join(missing)}")
        st.stop()

    for name, value in time_features.items():
        input_row.loc[0, name] = value

    slot_column = f"slot_id_{slot_id}"
    # The first/reference slot is represented by all-zero slot dummy columns.
    if slot_column in input_row.columns:
        input_row.loc[0, slot_column] = 1

    score = float(np.clip(model.predict(input_row)[0], 0, 1))
    available = score >= 0.5

    st.divider()
    st.subheader("Prediction result")
    score_col, status_col = st.columns([1, 1.2], gap="large")

    with score_col:
        st.metric("Estimated availability score", f"{score * 100:.1f}%")
        st.progress(score)

    with status_col:
        if available:
            st.markdown('<div class="status-available">● ESTIMATED STATUS: AVAILABLE</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-occupied">● ESTIMATED STATUS: OCCUPIED</div>', unsafe_allow_html=True)
        st.write(f"Slot **{slot_id}** · {hour:02d}:00 · Day {day} · Month {month}")

st.divider()
st.markdown("**Project:** Smart Parking Occupancy Prediction  ·  **Model:** Polynomial Ridge Regression (degree 3)")
st.caption("Prototype limitation: predictions reflect patterns in the training dataset. No real-time parking sensors or live parking database are connected.")
