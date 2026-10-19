import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from src.trainer import ModelTrainer
from src.drift_detector import DriftDetector

st.set_page_config(
    page_title="ML Model Evaluation & Monitoring",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 ML Model Evaluation & Drift Monitoring")
st.markdown("Track model training runs with **MLflow**, evaluate performance, and monitor production data drift with **PSI** and **KS-tests**.")

# Load Baseline Data
@st.cache_data
def load_datasets():
    train_df = pd.read_csv("data/training_data.csv")
    normal_df = pd.read_csv("data/production_normal.csv")
    drift_df = pd.read_csv("data/production_drifted.csv")
    return train_df, normal_df, drift_df

train_df, normal_df, drift_df = load_datasets()

tab_train, tab_drift = st.tabs(["🚀 Model Training & Tracking", "📡 Production Drift Monitor"])

with tab_train:
    st.subheader("Model Experimentation & Tracking")
    c1, c2, c3 = st.columns(3)
    model_choice = c1.selectbox("Algorithm:", ["RandomForest", "GradientBoosting", "LogisticRegression"])
    n_trees = c2.slider("Estimators / Iterations:", min_value=20, max_value=200, value=80, step=20)
    max_d = c3.slider("Max Depth:", min_value=2, max_value=12, value=5)

    if st.button("Train & Log to MLflow"):
        with st.spinner("Training model and logging metrics..."):
            trainer = ModelTrainer(experiment_name="Credit_Risk_Classification")
            X_train, X_val, y_train, y_val = trainer.prepare_data(train_df)
            model, metrics = trainer.train_model(
                X_train, y_train, X_val, y_val,
                model_type=model_choice,
                params={"n_estimators": n_trees, "max_depth": max_d},
            )
            st.session_state["active_model"] = model
            st.session_state["baseline_metrics"] = metrics
            st.success(f"Trained {model_choice} successfully and logged to MLflow!")

    if "baseline_metrics" in st.session_state:
        m = st.session_state["baseline_metrics"]
        st.markdown("#### Baseline Validation Metrics")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Accuracy", f"{m['accuracy']:.4f}")
        k2.metric("Precision", f"{m['precision']:.4f}")
        k3.metric("Recall", f"{m['recall']:.4f}")
        k4.metric("F1 Score", f"{m['f1']:.4f}")
        k5.metric("ROC-AUC", f"{m['roc_auc']:.4f}")

with tab_drift:
    st.subheader("Production Data & Covariate Shift Analysis")

    prod_choice = st.radio(
        "Select Production Batch to Monitor:",
        ["Batch A: In-Distribution Traffic (Normal)", "Batch B: Macroeconomic Shock (Drifted)"],
        horizontal=True,
    )
    prod_df = normal_df if "Normal" in prod_choice else drift_df

    features = ["income", "credit_score", "debt_to_income", "loan_amount", "employment_years"]
    drift_report = DriftDetector.evaluate_features_drift(train_df, prod_df, features)
    report_df = pd.DataFrame(drift_report)

    # Summary Alert
    flagged_count = sum(1 for r in drift_report if r["requires_attention"])
    if flagged_count >= 2:
        st.error(f"⚠️ **CRITICAL ALERT:** {flagged_count} of {len(features)} features exhibit significant covariate drift! Automated model retraining is recommended.")
    elif flagged_count == 1:
        st.warning(f"⚡ **WARNING:** 1 feature exhibits moderate distribution shift. Monitor closely.")
    else:
        st.success("✅ **STABLE:** Production feature distributions align with training baseline. No significant drift detected.")

    st.markdown("---")
    st.markdown("#### Statistical Drift Scorecard (PSI & KS-Test)")
    st.dataframe(
        report_df[["feature", "psi", "psi_status", "ks_statistic", "ks_pvalue", "requires_attention"]],
        use_container_width=True,
    )

    # Feature distribution overlay plot
    selected_feature = st.selectbox("Inspect Feature Distribution Overlay:", features)

    fig = ff.create_distplot(
        [train_df[selected_feature].dropna(), prod_df[selected_feature].dropna()],
        ["Training Baseline", "Production Batch"],
        show_hist=False,
        colors=["#3b82f6", "#ef4444" if "Drifted" in prod_choice else "#10b981"],
    )
    fig.update_layout(
        title=f"Distribution Density: {selected_feature}",
        xaxis_title=selected_feature,
        yaxis_title="Density",
        margin=dict(t=40, b=20, l=20, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)
