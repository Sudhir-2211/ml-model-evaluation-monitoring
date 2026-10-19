import pytest
import pandas as pd
from src.trainer import ModelTrainer


@pytest.fixture
def sample_data():
    return pd.read_csv("data/training_data.csv")


def test_prepare_data(sample_data):
    trainer = ModelTrainer(experiment_name="Test_Credit_Risk")
    X_train, X_val, y_train, y_val = trainer.prepare_data(sample_data, target_col="defaulted")

    assert len(X_train) == 750
    assert len(X_val) == 250
    assert "defaulted" not in X_train.columns
    assert len(y_train) == 750


def test_train_random_forest(sample_data):
    trainer = ModelTrainer(experiment_name="Test_Credit_Risk")
    X_train, X_val, y_train, y_val = trainer.prepare_data(sample_data, target_col="defaulted")

    model, metrics = trainer.train_model(
        X_train, y_train, X_val, y_val,
        model_type="RandomForest",
        params={"n_estimators": 20, "max_depth": 4},
    )

    assert model is not None
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert 0.5 <= metrics["accuracy"] <= 1.0
    assert 0.5 <= metrics["roc_auc"] <= 1.0
