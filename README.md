# CampaignLift AI: Causal Uplift Modeling for Market Targeting

An end-to-end causal machine learning project using the **open Criteo Uplift Prediction Dataset**.


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

