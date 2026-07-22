# Survival Analysis: Predicting Age at Male Circumcision in Cameroon

A comprehensive survival analysis project examining the timing of male circumcision in Cameroon using the **2018 Demographic and Health Survey (DHS)** data. The project combines classical statistical methods (Kaplan-Meier, Cox regression) with machine learning (Gradient Boosting Survival Analysis) and includes an interactive bilingual web application for predictions.

## Research Question

At what age are men in Cameroon circumcised, and what demographic, socioeconomic, religious, and regional factors influence that timing?

## Key Findings

- **~93%** of men in Cameroon are circumcised (survey-weighted estimate)
- **Median age at circumcision:** 3–4 years; 75% occur by age 9
- **Top predictors of circumcision timing:**
  - **Region** — strongest predictor (Far-North/North have lower rates; North-West, West, South-West have higher rates)
  - **Religion** — Animists have significantly lower circumcision hazards
  - **Wealth** — positively associated with earlier circumcision
  - **Media exposure** — weekly TV watching linked to earlier circumcision
- **ML model (GBSA) C-Index:** 0.718 with region and wealth dominating feature importance

## Project Structure

```
.
├── survival_analysis.R         # Full statistical analysis pipeline (R)
├── machine_learning.ipynb      # ML model development & evaluation (Jupyter)
├── train_model.py              # Standalone model training script
├── streamlit_app.py            # Bilingual (EN/FR) interactive web app
├── requirements.txt            # Python dependencies
├── dataset/
│   ├── CMMR71FL.SAV            # DHS Cameroon 2018 (SPSS format)
│   └── CMMR71FL.DTA            # DHS Cameroon 2018 (Stata format)
├── model_artifacts/            # Trained model artifacts (generated)
│   ├── gbsa_model.pkl
│   └── preprocessor.pkl
└── assets/
    └── bg-health.jpg           # Background image for Streamlit app
```

## Files Description

### `survival_analysis.R`
Full epidemiological analysis pipeline in R covering:
- Data loading, cleaning, and survival variable construction
- Survey-weighted descriptive statistics (DHS complex survey design)
- Kaplan-Meier estimation (overall and stratified by covariates)
- Log-rank tests for all 10 covariates
- Survey-weighted Cox proportional hazards regression (univariable & multivariable)
- Model diagnostics (Schoenfeld residuals, martingale/deviance residuals)
- Sensitivity analysis for handling DHS code 95 ("circumcised before age 5")

### `machine_learning.ipynb`
Jupyter notebook developing and comparing four ML survival models:
| Model | C-Index |
|-------|---------|
| Random Survival Forest | 0.603 |
| **Gradient Boosting Survival Analysis** | **0.718** |
| XGBoost Cox | 0.715 |
| XGBoost AFT | 0.288 |

Includes feature importance analysis, time-dependent AUC, and Integrated Brier Score evaluation.

### `train_model.py`
Standalone Python script to train the GBSA model and export artifacts (`gbsa_model.pkl`, `preprocessor.pkl`) to `model_artifacts/`.

### `streamlit_app.py`
Bilingual (English/French) Streamlit web application with three tabs:
- **Predictor** — Enter a respondent profile to get a predicted circumcision age, survival curve, and risk category (EARLY/MEDIUM/LATE)
- **Variables** — Dictionary of all 15 DHS variables used
- **About** — Project summary and methodology

## Dataset

**DHS Cameroon 2018 Men's Individual Recode** (`CMMR71FL`) — 6,978 male respondents, 835 variables. The analysis selects 15 key variables:

| Variable | Description |
|----------|-------------|
| `circumcision_status` | Whether respondent is circumcised (event indicator) |
| `age_circumcision` | Reported age at circumcision (event time) |
| `current_age` | Current age of respondent |
| `residence` | Urban vs. rural |
| `education` | Education level |
| `wealth` | Wealth quintile (poorest to richest) |
| `marital_status` | Marital status |
| `religion` | Religion |
| `region` | Administrative region (12 regions) |
| `tv_exposure` | Frequency of watching TV |
| `radio_exposure` | Frequency of listening to radio |
| `hiv_testing` | Whether ever tested for HIV |
| `weight` | Survey sample weight |
| `psu` | Primary sampling unit |
| `strata` | Sample stratum |

## Getting Started

### Prerequisites

- **R** (≥ 4.1) with packages: `haven`, `survey`, `survival`, `survminer`, `gtsummary`, `tableone`
- **Python** (≥ 3.9)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Enow-brenda/Survival-Analysis-Predicting-Age-at-Circumcision.git
   cd Survival-Analysis-Predicting-Age-at-Circumcision
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model:
   ```bash
   python train_model.py
   ```

4. Launch the interactive app:
   ```bash
   streamlit run streamlit_app.py
   ```

### Running the R Analysis

Open `survival_analysis.R` in RStudio and run the script. Required R packages:
```r
install.packages(c("haven", "survey", "survival", "survminer", "gtsummary", "tableone"))
```

## Methodology

### Statistical Analysis (R)
- **Survey design:** DHS-compliant complex survey design with PSU, strata, and normalized weights
- **Time-to-event:** Age at circumcision (event=1) or current age (event=0, censored)
- **Estimation:** Kaplan-Meier curves, Cox proportional hazards regression
- **Diagnostics:** Schoenfeld residuals (PH assumption), martingale/deviance residuals

### Machine Learning (Python)
- **Preprocessing:** One-hot encoding (nominal), StandardScaler (ordinal/numeric)
- **Model:** Gradient Boosting Survival Analysis (`scikit-survival`)
- **Evaluation:** C-index, time-dependent AUC, Integrated Brier Score
- **Prediction:** Individual survival curves, median survival time, risk categorization

## Data Source

The dataset is publicly available from the [DHS Program](https://dhsprogram.com/) — Cameroon 2018 Men's Individual Recode. Access requires registration and data use agreement.

## License

This project is for academic purposes. The DHS data is subject to the DHS Program's data use agreement.

## Authors

- **Enow Brenda** — [GitHub](https://github.com/Enow-brenda)
