# CampaignLift AI: Causal Uplift Modeling for Marketing Targeting

An end-to-end causal machine learning project using the **open Criteo Uplift Prediction Dataset**.

This project is designed for a portfolio targeting **Data Scientist, Product Data Scientist, Marketing Data Scientist, ML Engineer, and Senior Analytics roles**. It goes beyond normal response prediction by estimating which users are likely to convert **because of a campaign**, not just which users are likely to convert anyway.

---

## 1. Business problem

A standard marketing response model answers:

```text
Who is likely to convert?
```

But that is not enough. Some customers would convert even without an ad, and some may be annoyed by targeting. Uplift modeling answers a better business question:

```text
Who is likely to convert because of the treatment/campaign?
```

This project estimates individual treatment effect/uplift and creates a targeting policy:

```text
Predicted uplift = P(conversion | treated, user features) - P(conversion | control, user features)
```

---

## 2. Data source

Primary open data:

- **Criteo Uplift Prediction Dataset** from Criteo AI Lab.
- Fields include `f0` to `f11`, `treatment`, `conversion`, `visit`, and `exposure`.
- This repo uses a reproducible script to download the unbiased CSV file from Criteo's public download endpoint.

Important: the dataset is large. Start with a sample for local work.

---

## 3. Project architecture

```text
campaignlift_ai/
│
├── app/
│   └── streamlit_app.py
│
├── api/
│   └── main.py
│
├── config/
│   └── project_config.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── reports/
│   ├── model_card.md
│   └── figures/
│
├── scripts/
│   └── run_pipeline.py
│
├── src/
│   ├── config.py
│   ├── data_ingestion.py
│   ├── data_quality.py
│   ├── evaluate.py
│   ├── features.py
│   ├── metrics.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
│
├── tests/
│   └── test_uplift_metrics.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 4. Setup

### Step 1 — Create environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Run a quick pipeline

```bash
python scripts/run_pipeline.py --sample-rows 200000 --target visit
```

For a more portfolio-ready run:

```bash
python scripts/run_pipeline.py --sample-rows 1000000 --target visit
```

For conversion modeling, use:

```bash
python scripts/run_pipeline.py --sample-rows 1000000 --target conversion
```

`conversion` is rarer than `visit`, so use a larger sample.

---

## 5. What the pipeline does

### 5.1 Data ingestion

```bash
python -m src.data_ingestion --sample-rows 200000
```

Downloads the Criteo uplift CSV gzip file and saves a sample to:

```text
data/processed/criteo_uplift_sample.csv
```

### 5.2 Data quality

```bash
python -m src.data_quality
```

Creates:

```text
reports/data_quality_report.csv
```

### 5.3 Model training

```bash
python -m src.train --target visit
```

Trains:

1. **Response model** — predicts outcome without causal separation.
2. **S-Learner** — one model with treatment as a feature.
3. **T-Learner** — separate treated/control models.

Saves:

```text
models/campaignlift_model_bundle.joblib
reports/model_metrics.json
reports/uplift_predictions.csv
reports/decile_policy_summary.csv
reports/figures/qini_curve.png
```

### 5.4 Streamlit app

```bash
streamlit run app/streamlit_app.py
```

The app lets you enter anonymized customer features and returns:

- Predicted uplift
- Treatment recommendation
- Customer type: Persuadable / Sure Thing / Lost Cause / Sleeping Dog
- Portfolio decile targeting table
- Qini-style performance chart

### 5.5 FastAPI scoring service

```bash
uvicorn api.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

Example payload:

```json
{
  "f0": 12.5,
  "f1": 10.2,
  "f2": 8.1,
  "f3": 4.0,
  "f4": 10.5,
  "f5": 5.2,
  "f6": 0.1,
  "f7": 2.4,
  "f8": 9.1,
  "f9": 6.2,
  "f10": 1.7,
  "f11": 3.3
}
```

---

## 6. Suggested portfolio write-up

**Short portfolio description**

Built a causal marketing uplift modeling platform using open incrementality-test data. The system compares response modeling with S-Learner and T-Learner uplift models, ranks customers by predicted treatment effect, visualizes Qini-style curves, and simulates campaign targeting policies by decile.

**Resume bullet**

Developed a causal marketing uplift engine using open incrementality-test data, applying S-Learner and T-Learner models to identify persuadable customers, improve campaign targeting, and reduce wasted ad spend.

---

## 7. Next upgrades

- Add X-Learner.
- Add Causal Forest via EconML.
- Add scikit-uplift AUUC/Qini metrics.
- Add SHAP by comparing feature importance for treated and control models.
- Add MLflow experiment tracking.
- Add dashboard deployment.
- Add cost-aware targeting policy with campaign budget constraints.
