import os
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow


class ModelTrainer:
    def __init__(self, experiment_name: str = "Credit_Risk_Classification"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    def prepare_data(self, df: pd.DataFrame, target_col: str = "defaulted") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_type: str = "RandomForest",
        params: Dict[str, Any] = None,
    ) -> Tuple[Any, Dict[str, float]]:
        params = params or {}

        if model_type == "RandomForest":
            model = RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 6),
                random_state=42,
            )
        elif model_type == "GradientBoosting":
            model = GradientBoostingClassifier(
                n_estimators=params.get("n_estimators", 100),
                learning_rate=params.get("learning_rate", 0.1),
                max_depth=params.get("max_depth", 4),
                random_state=42,
            )
        else:
            model = LogisticRegression(max_iter=1000, random_state=42)

        with mlflow.start_run(run_name=f"{model_type}_run"):
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            probs = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else preds

            metrics = {
                "accuracy": round(float(accuracy_score(y_val, preds)), 4),
                "precision": round(float(precision_score(y_val, preds, zero_division=0)), 4),
                "recall": round(float(recall_score(y_val, preds, zero_division=0)), 4),
                "f1": round(float(f1_score(y_val, preds, zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(y_val, probs)), 4),
            }

            # Log parameters and metrics
            mlflow.log_params(params)
            mlflow.log_param("model_type", model_type)
            mlflow.log_metrics(metrics)

        return model, metrics
