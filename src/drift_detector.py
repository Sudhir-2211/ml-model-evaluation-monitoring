from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score


class DriftDetector:
    @staticmethod
    def calculate_psi(baseline: pd.Series, production: pd.Series, num_bins: int = 10) -> Dict[str, Any]:
        """
        Calculates Population Stability Index (PSI) to detect covariate shift.
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate shift (Warning)
        PSI >= 0.2: Significant shift (Action Required)
        """
        # Create quantile-based bins from baseline
        quantiles = np.linspace(0, 1, num_bins + 1)
        bins = np.percentile(baseline.dropna(), quantiles * 100)
        bins[0] = -np.inf
        bins[-1] = np.inf
        bins = np.unique(bins)

        if len(bins) < 3:
            # Fallback to linear bins if distribution has low cardinality
            bins = np.linspace(baseline.min(), baseline.max(), num_bins + 1)
            bins[0] = -np.inf
            bins[-1] = np.inf

        base_counts, _ = np.histogram(baseline.dropna(), bins=bins)
        prod_counts, _ = np.histogram(production.dropna(), bins=bins)

        # Convert to percentages with small epsilon to avoid log(0) or division by 0
        eps = 1e-4
        base_pct = (base_counts / max(1, len(baseline))) + eps
        prod_pct = (prod_counts / max(1, len(production))) + eps

        # Normalize to sum to 1
        base_pct = base_pct / base_pct.sum()
        prod_pct = prod_pct / prod_pct.sum()

        psi_val = np.sum((prod_pct - base_pct) * np.log(prod_pct / base_pct))
        psi_val = max(0.0, float(psi_val))

        status = "STABLE"
        if psi_val >= 0.2:
            status = "CRITICAL_DRIFT"
        elif psi_val >= 0.1:
            status = "MODERATE_DRIFT"

        return {
            "psi": round(psi_val, 4),
            "status": status,
        }

    @staticmethod
    def calculate_ks_test(baseline: pd.Series, production: pd.Series) -> Dict[str, Any]:
        """Runs two-sample Kolmogorov-Smirnov test for continuous distributions."""
        res = stats.ks_2samp(baseline.dropna(), production.dropna())
        statistic = round(float(res.statistic), 4)
        pvalue = round(float(res.pvalue), 6)
        is_drift = bool(pvalue < 0.05)

        return {
            "statistic": statistic,
            "p_value": pvalue,
            "is_drift_detected": is_drift,
        }

    @classmethod
    def evaluate_features_drift(
        cls,
        baseline_df: pd.DataFrame,
        production_df: pd.DataFrame,
        features: List[str],
    ) -> List[Dict[str, Any]]:
        results = []
        for feat in features:
            if feat not in baseline_df.columns or feat not in production_df.columns:
                continue

            base_s = baseline_df[feat]
            prod_s = production_df[feat]

            psi_info = cls.calculate_psi(base_s, prod_s)
            ks_info = cls.calculate_ks_test(base_s, prod_s)

            results.append({
                "feature": feat,
                "psi": psi_info["psi"],
                "psi_status": psi_info["status"],
                "ks_statistic": ks_info["statistic"],
                "ks_pvalue": ks_info["p_value"],
                "ks_drift": ks_info["is_drift_detected"],
                "requires_attention": (psi_info["psi"] >= 0.1 or ks_info["is_drift_detected"]),
            })
        return results

    @staticmethod
    def evaluate_performance_degradation(
        model: Any,
        production_df: pd.DataFrame,
        target_col: str = "defaulted",
        baseline_f1: float = 0.85,
    ) -> Dict[str, Any]:
        X_prod = production_df.drop(columns=[target_col])
        y_prod = production_df[target_col]

        preds = model.predict(X_prod)
        prod_acc = round(float(accuracy_score(y_prod, preds)), 4)
        prod_f1 = round(float(f1_score(y_prod, preds, zero_division=0)), 4)

        f1_drop_pct = round(((baseline_f1 - prod_f1) / max(0.01, baseline_f1)) * 100, 2)
        alert_triggered = f1_drop_pct > 10.0

        return {
            "production_accuracy": prod_acc,
            "production_f1": prod_f1,
            "baseline_f1": baseline_f1,
            "f1_drop_pct": f1_drop_pct,
            "retraining_alert": alert_triggered,
        }
