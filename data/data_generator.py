import os
import numpy as np
import pandas as pd


def generate_synthetic_ml_data(output_dir: str = "data"):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    # 1. Baseline Training Distribution (1,000 samples)
    n_train = 1000
    income_train = np.random.normal(loc=65000, scale=18000, size=n_train).clip(20000, 200000)
    credit_score_train = np.random.normal(loc=710, scale=60, size=n_train).clip(450, 850)
    debt_ratio_train = np.random.beta(a=2, b=5, size=n_train) * 0.8
    loan_amount_train = np.random.normal(loc=18000, scale=8000, size=n_train).clip(2000, 60000)
    employment_train = np.random.exponential(scale=5, size=n_train).clip(0, 35)

    # Risk formulation
    risk_score_train = (
        - 0.00003 * income_train
        - 0.01 * credit_score_train
        + 3.5 * debt_ratio_train
        + 0.00004 * loan_amount_train
        - 0.08 * employment_train
        + 5.0
    )
    prob_default_train = 1.0 / (1.0 + np.exp(-risk_score_train))
    defaulted_train = (np.random.rand(n_train) < prob_default_train).astype(int)

    df_train = pd.DataFrame({
        "income": income_train.round(2),
        "credit_score": credit_score_train.round(1),
        "debt_to_income": debt_ratio_train.round(4),
        "loan_amount": loan_amount_train.round(2),
        "employment_years": employment_train.round(1),
        "defaulted": defaulted_train,
    })
    df_train.to_csv(os.path.join(output_dir, "training_data.csv"), index=False)

    # 2. Production In-Distribution Dataset (500 samples)
    n_prod = 500
    income_prod = np.random.normal(loc=65000, scale=18000, size=n_prod).clip(20000, 200000)
    credit_score_prod = np.random.normal(loc=710, scale=60, size=n_prod).clip(450, 850)
    debt_ratio_prod = np.random.beta(a=2, b=5, size=n_prod) * 0.8
    loan_amount_prod = np.random.normal(loc=18000, scale=8000, size=n_prod).clip(2000, 60000)
    employment_prod = np.random.exponential(scale=5, size=n_prod).clip(0, 35)

    risk_score_prod = (
        - 0.00003 * income_prod
        - 0.01 * credit_score_prod
        + 3.5 * debt_ratio_prod
        + 0.00004 * loan_amount_prod
        - 0.08 * employment_prod
        + 5.0
    )
    prob_default_prod = 1.0 / (1.0 + np.exp(-risk_score_prod))
    defaulted_prod = (np.random.rand(n_prod) < prob_default_prod).astype(int)

    df_prod = pd.DataFrame({
        "income": income_prod.round(2),
        "credit_score": credit_score_prod.round(1),
        "debt_to_income": debt_ratio_prod.round(4),
        "loan_amount": loan_amount_prod.round(2),
        "employment_years": employment_prod.round(1),
        "defaulted": defaulted_prod,
    })
    df_prod.to_csv(os.path.join(output_dir, "production_normal.csv"), index=False)

    # 3. Production Drifted Dataset (500 samples: Macroeconomic credit crunch shock)
    income_drift = np.random.normal(loc=48000, scale=15000, size=n_prod).clip(15000, 160000)  # Drop in income
    credit_score_drift = np.random.normal(loc=660, scale=70, size=n_prod).clip(420, 820)       # Drop in scores
    debt_ratio_drift = np.random.beta(a=4, b=4, size=n_prod) * 0.95                             # Surging debt ratio
    loan_amount_drift = np.random.normal(loc=24000, scale=9000, size=n_prod).clip(3000, 70000) # Higher loan requests
    employment_drift = np.random.exponential(scale=3, size=n_prod).clip(0, 25)

    risk_score_drift = (
        - 0.00003 * income_drift
        - 0.01 * credit_score_drift
        + 3.5 * debt_ratio_drift
        + 0.00004 * loan_amount_drift
        - 0.08 * employment_drift
        + 5.0
    )
    prob_default_drift = 1.0 / (1.0 + np.exp(-risk_score_drift))
    defaulted_drift = (np.random.rand(n_prod) < prob_default_drift).astype(int)

    df_drift = pd.DataFrame({
        "income": income_drift.round(2),
        "credit_score": credit_score_drift.round(1),
        "debt_to_income": debt_ratio_drift.round(4),
        "loan_amount": loan_amount_drift.round(2),
        "employment_years": employment_drift.round(1),
        "defaulted": defaulted_drift,
    })
    df_drift.to_csv(os.path.join(output_dir, "production_drifted.csv"), index=False)


if __name__ == "__main__":
    generate_synthetic_ml_data()
    print("Synthetic ML datasets generated successfully.")
