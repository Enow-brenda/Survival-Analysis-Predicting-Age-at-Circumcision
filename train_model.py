"""
Train the Gradient Boosting Survival Analysis (GBSA) model
==========================================================
Reproduces exactly the pipeline from training.ipynb.

USAGE
-----
1. Place the DHS Cameroon 2018 men's recode dataset next to this file:
       CMMR71FL.dta
2. Run:
       python train_model.py
3. Two artifacts will be created in ./model_artifacts/ :
       - gbsa_model.pkl
       - preprocessor.pkl
4. Launch the app:
       streamlit run streamlit_app.py
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyreadstat

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv


DATA_PATH = Path("dataset/CMMR71FL.dta")
OUT_DIR = Path("model_artifacts")
OUT_DIR.mkdir(exist_ok=True)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH.resolve()}.\n"
            "Place CMMR71FL.dta next to this script and run again."
        )

    print(f"Loading {DATA_PATH} ...")
    data, _ = pyreadstat.read_dta(str(DATA_PATH))

    features = ["mv483", "mv483a", "mv012", "mv025", "mv106", "mv190",
                "mv501", "mv130", "mv024", "mv157", "mv158", "mv781"]
    data = data[features].rename(columns={
        "mv483": "circumcision_status",
        "mv483a": "age_circumcision",
        "mv012": "current_age",
        "mv025": "residence",
        "mv106": "education",
        "mv190": "wealth",
        "mv501": "marital_status",
        "mv130": "religion",
        "mv024": "region",
        "mv157": "tv_exposure",
        "mv158": "radio_exposure",
        "mv781": "hiv_testing",
    })

    # Cleaning (same as notebook)
    data = data[data["circumcision_status"] != 8]

    def compute_time(row):
        if row["circumcision_status"] == 1:
            if row["age_circumcision"] == 95:
                return 3
            if row["age_circumcision"] == 0:
                return 0.25
            return row["age_circumcision"]
        return row["current_age"]

    data["event"] = (data["circumcision_status"] == 1).astype(int)
    data["time"] = data.apply(compute_time, axis=1)
    data = data[data["time"] != 98]
    data = data.drop(columns=["circumcision_status", "age_circumcision"])

    categorical = ["residence", "education", "wealth", "marital_status",
                   "religion", "region", "tv_exposure", "radio_exposure",
                   "hiv_testing"]
    for col in categorical:
        data[col] = data[col].astype("category")

    X = data.drop(columns=["time", "event"])
    y_time = data["time"]
    y_event = data["event"]

    X_train, X_test, y_t_train, y_t_test, y_e_train, y_e_test = train_test_split(
        X, y_time, y_event, test_size=0.20, random_state=42
    )

    nominal = ["residence", "marital_status", "religion", "region", "hiv_testing"]
    ordinal = ["education", "wealth", "tv_exposure", "radio_exposure"]
    numeric = ["current_age"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal),
            ("num", StandardScaler(), numeric + ordinal),
        ],
        remainder="drop",
    )

    print("Fitting preprocessor ...")
    X_train_p = preprocessor.fit_transform(X_train)

    y_train = Surv.from_arrays(y_e_train.astype(bool), y_t_train)

    print("Training GradientBoostingSurvivalAnalysis ...")
    gbsa = GradientBoostingSurvivalAnalysis(
        loss="coxph",
        learning_rate=0.05,
        n_estimators=150,
        max_depth=5,
        subsample=0.8,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=42,
    )
    gbsa.fit(X_train_p, y_train)

    joblib.dump(gbsa, OUT_DIR / "gbsa_model.pkl")
    joblib.dump(preprocessor, OUT_DIR / "preprocessor.pkl")
    print(f"Saved artifacts to {OUT_DIR.resolve()}")
    print("Done. Run:  streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()
