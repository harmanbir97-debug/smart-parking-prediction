# ParkWise — Smart Parking Prediction

## Files
- `app.py`: Streamlit website source
- `requirements.txt`: Python dependencies
- `smart_parking_model.pkl`: export this from your trained Google Colab notebook and place it beside `app.py`

## 1. Export the trained model from Colab

Run this after the Degree 3 Polynomial Ridge model (`poly_ridge_3`), `X`, and `df` exist:

```python
import joblib
from google.colab import files

model_bundle = {
    "model": poly_ridge_3,
    "feature_cols": list(X.columns),
    "slot_ids": sorted(df["slot_id"].unique())
}

joblib.dump(model_bundle, "smart_parking_model.pkl")
files.download("smart_parking_model.pkl")
```

**Important:** This bundle must be exported from the same notebook state used to train `poly_ridge_3`. The app reconstructs the 55 input columns used by the model.

## 2. Put the model file in this folder

The folder should contain:
```text
Smart_Parking_App/
  app.py
  requirements.txt
  README.md
  smart_parking_model.pkl
```

## 3. Run on Windows

Open Command Prompt in this folder and run:

```bash
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Streamlit will print a local address (usually `http://localhost:8501`) and open it in your browser.

## Notes
- Day values are 0–6, matching the dataset encoding.
- Availability score is a regression output clipped to 0–1 and shown as a percentage. It is not a calibrated probability.
- The 0.5 threshold is used only to label the score AVAILABLE/OCCUPIED.
- This is a dataset-based prototype, not a live sensor-connected parking system.
