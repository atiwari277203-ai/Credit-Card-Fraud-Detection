
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
from pathlib import Path

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection (XGBoost)")

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model_xgb.joblib"
SCALER_PATH = ARTIFACT_DIR / "scaler.joblib"
FEATS_PATH = ARTIFACT_DIR / "feature_columns.json"
ROC_PATH = ARTIFACT_DIR / "roc_metrics.json"
CM_PATH = ARTIFACT_DIR / "cm_xgb.png"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATS_PATH) as f:
        feature_cols = json.load(f)
    roc = None
    if ROC_PATH.exists():
        with open(ROC_PATH) as f:
            roc = json.load(f)
    return model, scaler, feature_cols, roc

def ensure_columns(df, feature_cols):
    # Order and keep only expected columns
    cols_ok = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return None
    return df[feature_cols]

model, scaler, feature_cols, roc = load_artifacts()

tab1, tab2 = st.tabs(["🔮 Predict", "📈 Metrics & Diagnostics"])

with tab1:
    st.subheader("Batch Prediction (Upload CSV)")
    st.write("Upload a CSV with the same schema as the training data (all V1..V28, Time, Amount).")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    threshold = st.slider("Decision Threshold (probability ≥ threshold => Fraud)", 0.01, 0.99, 0.5, 0.01)

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        df_scaled = df.copy()
        if "Time" in df_scaled.columns and "Amount" in df_scaled.columns:
            df_scaled[["Time", "Amount"]] = scaler.transform(df_scaled[["Time", "Amount"]])
        df_infer = ensure_columns(df_scaled, feature_cols)
        if df_infer is not None:
            proba = model.predict_proba(df_infer)[:, 1]
            preds = (proba >= threshold).astype(int)
            out = df.copy()
            out["fraud_probability"] = proba
            out["fraud_pred"] = preds
            st.success(f"Predicted {int(preds.sum())} frauds out of {len(preds)} rows.")
            st.dataframe(out.head(50))
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")

with tab2:
    st.subheader("Confusion Matrix (from training notebook)")
    if Path(CM_PATH).exists():
        st.image(str(CM_PATH), caption="XGBoost Confusion Matrix")
    else:
        st.info("Run the notebook to generate and save confusion matrix PNG.")

    st.subheader("ROC Curve")
    if roc is not None:
        fpr = np.array(roc["fpr"])
        tpr = np.array(roc["tpr"])
        auc_val = roc["auc"]
        fig = plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
        plt.plot([0,1],[0,1],'--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - XGBoost")
        plt.legend(loc="lower right")
        st.pyplot(fig)
    else:
        st.info("ROC data not found. Run the notebook to create artifacts/roc_metrics.json.")
    
    st.caption("Tip: Re-run training periodically and redeploy to keep performance fresh.")
