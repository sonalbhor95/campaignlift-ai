from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src.config import MODEL_BUNDLE_FILE, PREDICTIONS_FILE, METRICS_FILE, FEATURE_COLUMNS, REPORTS_DIR
from src.predict import load_model_bundle, predict_customer
from src.utils import load_json

st.set_page_config(page_title="CampaignLift AI", page_icon="📈", layout="wide")

st.title("CampaignLift AI: Causal Uplift Modeling for Marketing Targeting")
st.caption("Predict who should receive a campaign based on estimated incremental impact.")

@st.cache_resource
def get_bundle():
    return load_model_bundle(MODEL_BUNDLE_FILE)

bundle = get_bundle()

st.sidebar.header("Customer Feature Input")
customer = {}
for feature in FEATURE_COLUMNS:
    customer[feature] = st.sidebar.number_input(feature, value=0.0, step=0.1)

pred = predict_customer(customer, bundle)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customer Type", pred["customer_type"])
c2.metric("Predicted Uplift", f"{pred['predicted_uplift']:.4f}")
c3.metric("Treat Probability", f"{pred['p_outcome_if_treated']:.4f}")
c4.metric("Control Probability", f"{pred['p_outcome_if_control']:.4f}")

st.subheader("Recommendation")
if pred["recommend_treatment"]:
    st.success("Recommend targeting this customer.")
else:
    st.warning("Do not target this customer based on predicted uplift.")
st.json(pred)

st.subheader("Model Metrics")
if METRICS_FILE.exists():
    st.json(load_json(METRICS_FILE))
else:
    st.info("Run the training pipeline to generate metrics.")

st.subheader("Decile Targeting Summary")
decile_file = REPORTS_DIR / "decile_policy_summary.csv"
if decile_file.exists():
    st.dataframe(pd.read_csv(decile_file))
else:
    st.info("Run the training pipeline to generate decile summary.")

st.subheader("Policy Simulation")
policy_file = REPORTS_DIR / "targeting_policy_summary.csv"
if policy_file.exists():
    st.dataframe(pd.read_csv(policy_file))
else:
    st.info("Run the training pipeline to generate targeting policy summary.")

st.subheader("Prediction Sample")
if PREDICTIONS_FILE.exists():
    preds = pd.read_csv(PREDICTIONS_FILE)
    st.bar_chart(preds["customer_type"].value_counts())
    st.dataframe(preds.head(50))
else:
    st.info("Run the training pipeline to generate predictions.")


st.subheader("Model Monitoring")

monitoring_file = REPORTS_DIR / "model_monitoring_report.json"

if monitoring_file.exists():
    monitoring = load_json(monitoring_file)

    st.write("### Monitoring Summary")
    st.json({
        "reference_rows": monitoring.get("reference_rows"),
        "current_rows": monitoring.get("current_rows"),
        "features_with_drift": monitoring.get("features_with_drift"),
    })

    st.write("### Feature Drift")
    feature_drift = monitoring.get("feature_drift", {})
    if feature_drift:
        drift_df = pd.DataFrame(feature_drift).T.reset_index().rename(columns={"index": "feature"})
        st.dataframe(drift_df)

    st.write("### Prediction Monitoring")
    prediction_monitoring = monitoring.get("prediction_monitoring", {})
    if prediction_monitoring:
        st.json(prediction_monitoring)
else:
    st.info("Run `python scripts/run_monitoring.py` to generate monitoring artifacts.")
