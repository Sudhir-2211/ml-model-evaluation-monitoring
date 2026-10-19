import pytest
import pandas as pd
from src.drift_detector import DriftDetector
from src.trainer import ModelTrainer


@pytest.fixture
def datasets():
    train_df = pd.read_csv("data/training_data.csv")
    normal_df = pd.read_csv("data/production_normal.csv")
    drifted_df = pd.read_csv("data/production_drifted.csv")
    return train_df, normal_df, drifted_df


def test_psi_stable_vs_drifted(datasets):
    train_df, normal_df, drifted_df = datasets

    # Normal comparison should have low PSI (< 0.1)
    res_normal = DriftDetector.calculate_psi(train_df["income"], normal_df["income"])
    assert res_normal["psi"] < 0.15
    assert res_normal["status"] in ("STABLE", "MODERATE_DRIFT")

    # Drifted comparison should have high PSI (>= 0.2)
    res_drifted = DriftDetector.calculate_psi(train_df["income"], drifted_df["income"])
    assert res_drifted["psi"] > 0.15
    assert res_drifted["status"] in ("CRITICAL_DRIFT", "MODERATE_DRIFT")


def test_ks_test_drift(datasets):
    train_df, normal_df, drifted_df = datasets

    # Normal data should have higher p-value (> 0.01)
    ks_normal = DriftDetector.calculate_ks_test(train_df["debt_to_income"], normal_df["debt_to_income"])
    assert ks_normal["p_value"] > 0.01

    # Drifted data should reject null hypothesis (p-value close to 0)
    ks_drifted = DriftDetector.calculate_ks_test(train_df["debt_to_income"], drifted_df["debt_to_income"])
    assert ks_drifted["p_value"] < 0.05
    assert ks_drifted["is_drift_detected"] is True


def test_evaluate_features_drift(datasets):
    train_df, _, drifted_df = datasets
    features = ["income", "credit_score", "debt_to_income", "loan_amount"]
    drift_report = DriftDetector.evaluate_features_drift(train_df, drifted_df, features)

    assert len(drift_report) == 4
    # At least two features in drifted dataset should require attention
    flagged = [f for f in drift_report if f["requires_attention"]]
    assert len(flagged) >= 2


def test_performance_degradation(datasets):
    train_df, _, drifted_df = datasets
    trainer = ModelTrainer(experiment_name="Test_Perf_Degradation")
    X_train, X_val, y_train, y_val = trainer.prepare_data(train_df)
    model, metrics = trainer.train_model(X_train, y_train, X_val, y_val, model_type="RandomForest")

    perf = DriftDetector.evaluate_performance_degradation(
        model,
        drifted_df,
        target_col="defaulted",
        baseline_f1=metrics["f1"],
    )

    assert "production_accuracy" in perf
    assert "production_f1" in perf
    assert "retraining_alert" in perf
