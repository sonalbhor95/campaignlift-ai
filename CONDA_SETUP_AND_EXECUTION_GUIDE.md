# CampaignLift AI — Conda Setup, Execution Steps, and Learning Links

This file gives you a practical step-by-step path to run **CampaignLift AI: Causal Uplift Modeling for Marketing Targeting** on your local machine using **conda**.

Use this guide after unzipping the project folder.

---

## 0. What this project does

This project uses the open Criteo Uplift Prediction Dataset to build a causal marketing ML pipeline.

Instead of asking only:

```text
Who is likely to convert?
```

this project asks:

```text
Who is likely to convert because of the campaign?
```

The project trains:

1. A normal response model.
2. An **S-Learner** uplift model.
3. A **T-Learner** uplift model.
4. A customer-level uplift ranking table.
5. A decile targeting policy summary.
6. A Qini-style uplift curve.
7. A Streamlit dashboard.
8. A FastAPI scoring API.

---

## 1. Prerequisites

Install these before running the project:

1. **Anaconda or Miniconda**
   - Anaconda: https://www.anaconda.com/download
   - Miniconda: https://docs.conda.io/en/latest/miniconda.html

2. **Git**
   - https://git-scm.com/downloads

3. **VS Code**
   - https://code.visualstudio.com/download

4. Optional but recommended VS Code extensions:
   - Python
   - Jupyter
   - GitHub Pull Requests

---

## 2. Important dataset note

The Criteo dataset is large.

The project downloads this public file:

```text
https://go.criteo.net/criteo-research-uplift-v2.1.csv.gz
```

The script reads only a sample by default, so you do not need to load the entire dataset into memory.

Start with:

```bash
--sample-rows 200000
```

Then increase to:

```bash
--sample-rows 1000000
```

Use `visit` first because conversion is much rarer.

---

## 3. Open the project folder

After downloading and unzipping the project, open **Anaconda Prompt** on Windows.

Example Windows path:

```bash
cd C:\Users\YourName\Downloads\campaignlift_ai
```

If your folder is on Desktop:

```bash
cd C:\Users\YourName\Desktop\campaignlift_ai
```

Check that you are inside the correct folder:

```bash
dir
```

You should see files such as:

```text
README.md
requirements.txt
scripts
src
app
api
reports
models
```

On Mac/Linux, use:

```bash
ls
```

---

## 4. Create the conda environment

Recommended environment name:

```text
campaignlift-ai
```

Create the environment:

```bash
conda create -n campaignlift-ai python=3.11 -y
```

Activate it:

```bash
conda activate campaignlift-ai
```

Confirm that Python is coming from the conda environment:

```bash
python --version
where python
```

On Mac/Linux:

```bash
which python
```

You should see Python 3.11 and a path that includes `campaignlift-ai`.

---

## 5. Install project dependencies

Upgrade pip first:

```bash
python -m pip install --upgrade pip
```

Install packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 6. Run quick tests

Run:

```bash
python -m pytest tests -q
```

Expected result:

```text
passed
```

If tests fail, check that you activated the correct conda environment and installed requirements inside that environment.

---

## 7. Run the full pipeline

### Option A — Quick local run using visit target

Use this first:

```bash
python scripts/run_pipeline.py --sample-rows 200000 --target visit
```

This downloads the open Criteo data, reads a sample, runs data quality checks, trains uplift models, and saves outputs.

### Option B — Portfolio-ready run using visit target

After the quick run works, use a larger sample:

```bash
python scripts/run_pipeline.py --sample-rows 1000000 --target visit
```

### Option C — Conversion modeling

Conversion is much rarer, so use a larger sample:

```bash
python scripts/run_pipeline.py --sample-rows 1000000 --target conversion
```

### Option D — Force re-download data

Use this only if the download file becomes corrupted or incomplete:

```bash
python scripts/run_pipeline.py --sample-rows 200000 --target visit --force-download
```

---

## 8. Files created after pipeline execution

After a successful run, check these outputs:

```text
data/raw/criteo-research-uplift-v2.1.csv.gz
data/processed/criteo_uplift_sample.csv
models/campaignlift_model_bundle.joblib
reports/data_quality_report.csv
reports/model_metrics.json
reports/uplift_predictions.csv
reports/decile_policy_summary.csv
reports/figures/qini_curve.png
```

Open these first:

1. `reports/model_metrics.json`
2. `reports/decile_policy_summary.csv`
3. `reports/uplift_predictions.csv`
4. `reports/figures/qini_curve.png`

These are the files you will use to explain the project in your portfolio.

---

## 9. Run the Streamlit dashboard

From the project root folder:

```bash
streamlit run app/streamlit_app.py
```

A browser window should open automatically.

If it does not open, copy the local URL shown in the terminal. It usually looks like:

```text
http://localhost:8501
```

Dashboard sections to review:

- Uplift prediction
- Treatment recommendation
- Persuadable / Sure Thing / Lost Cause / Sleeping Dog customer types
- Targeting deciles
- Qini-style uplift curve
- Campaign ROI simulation

---

## 10. Run the FastAPI scoring API

Open a second Anaconda Prompt window.

Go to the same project folder:

```bash
cd C:\Users\YourName\Downloads\campaignlift_ai
conda activate campaignlift-ai
```

Run:

```bash
uvicorn api.main:app --reload
```

Open this URL:

```text
http://127.0.0.1:8000/docs
```

Use the Swagger UI to test the API endpoint.

Stop the API with:

```bash
CTRL + C
```

---

## 11. Recommended execution order every time

Use this order when you work on the project:

```bash
conda activate campaignlift-ai
python -m pytest tests -q
python scripts/run_pipeline.py --sample-rows 200000 --target visit
streamlit run app/streamlit_app.py
uvicorn api.main:app --reload
```

For final portfolio screenshots:

```bash
python scripts/run_pipeline.py --sample-rows 1000000 --target visit
streamlit run app/streamlit_app.py
```

---

## 12. Common errors and fixes

### Error: `ModuleNotFoundError: No module named src`

Make sure you are running commands from the project root folder, not from inside `src` or `scripts`.

Correct:

```bash
cd C:\Users\YourName\Downloads\campaignlift_ai
python scripts/run_pipeline.py --sample-rows 200000 --target visit
```

Wrong:

```bash
cd C:\Users\YourName\Downloads\campaignlift_ai\scripts
python run_pipeline.py
```

### Error: Criteo download fails

Try again later or use a stable internet connection. The dataset is large, so the download can fail if your connection drops.

Then run:

```bash
python scripts/run_pipeline.py --sample-rows 200000 --target visit --force-download
```

### Error: Memory issue

Use fewer rows:

```bash
python scripts/run_pipeline.py --sample-rows 100000 --target visit
```

Then increase gradually.

### Results look weak with conversion target

That is expected with small samples because conversion is rare.

Use:

```bash
python scripts/run_pipeline.py --sample-rows 1000000 --target conversion
```

or start with `visit` for project development.

---

## 13. How to push this project to GitHub

Create an empty GitHub repository named:

```text
campaignlift-ai
```

Do **not** initialize it with README, `.gitignore`, or license if your local folder already has those files.

Then run these commands from the project root:

```bash
git init -b main
git add .
git commit -m "Initial CampaignLift AI project"
git remote add origin https://github.com/YOUR_USERNAME/campaignlift-ai.git
git remote -v
git push -u origin main
```

If GitHub asks for credentials, use GitHub Desktop or GitHub CLI if command-line authentication is confusing.

---

## 14. Suggested GitHub README improvements after first successful run

After you run the project successfully, update `README.md` with:

1. Final model metrics from `reports/model_metrics.json`.
2. Screenshot of the Qini curve.
3. Screenshot of the Streamlit dashboard.
4. Decile targeting insights.
5. One business ROI example.
6. A short “What I learned” section.
7. A “Future improvements” section.

Good future improvements:

- Add X-Learner.
- Add Causal Forest.
- Add EconML or CausalML version.
- Add model monitoring.
- Add Dockerfile.
- Deploy dashboard on Streamlit Community Cloud.

---

## 15. Learning links

### Data source and uplift modeling

- Criteo AI Lab uplift dataset: https://ailab.criteo.com/criteo-uplift-prediction-dataset/
- TensorFlow Datasets Criteo documentation: https://www.tensorflow.org/datasets/catalog/criteo
- scikit-uplift documentation: https://www.uplift-modeling.com/en/latest/index.html
- scikit-uplift GitHub: https://github.com/maks-sh/scikit-uplift
- Uber CausalML GitHub: https://github.com/uber/causalml
- Microsoft EconML GitHub: https://github.com/py-why/EconML

### Concepts to learn while building

Search these on YouTube:

- `uplift modeling marketing machine learning Python`
- `S learner T learner uplift modeling Python`
- `Qini curve uplift model explained`
- `causal inference for data scientists Python`
- `CATE heterogeneous treatment effect machine learning`
- `marketing campaign uplift modeling`
- `FastAPI machine learning model deployment`
- `Streamlit machine learning dashboard`

Useful YouTube links:

- Introduction to Uplift Modeling: https://www.youtube.com/watch?v=VWjsi-5yc3w
- Why start using uplift models for more efficient marketing campaigns: https://www.youtube.com/watch?v=2J9j7peWQgI
- X-Learner Uplift Model in Python: https://www.youtube.com/watch?v=iMcnT3cbbIg
- FastAPI ML deployment search: https://www.youtube.com/results?search_query=fastapi+machine+learning+model+deployment
- Streamlit ML dashboard search: https://www.youtube.com/results?search_query=streamlit+machine+learning+dashboard

### GitHub and environment setup

- Conda managing environments: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
- Conda managing Python versions: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-python.html
- GitHub docs — adding local code to GitHub: https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github
- GitHub CLI: https://cli.github.com/
- GitHub Desktop: https://desktop.github.com/

---

## 16. Conda cleanup commands

To deactivate the environment:

```bash
conda deactivate
```

To list environments:

```bash
conda env list
```

To remove the environment completely:

```bash
conda remove -n campaignlift-ai --all -y
```

---

## 17. Portfolio story for this project

Use this story in your portfolio:

> CampaignLift AI is an end-to-end causal marketing ML system built using the open Criteo Uplift Prediction Dataset. The project estimates customer-level treatment effect, compares response modeling with uplift modeling, ranks users by predicted incremental lift, and simulates campaign targeting decisions using decile analysis and Qini-style evaluation.

Strong resume bullet:

> Developed a causal marketing uplift engine using open incrementality-test data, applying S-Learner and T-Learner models to identify persuadable customers, prioritize campaign targeting, and estimate incremental business impact.
