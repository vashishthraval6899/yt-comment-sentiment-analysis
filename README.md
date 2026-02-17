# 🎬 YouTube Sentiment AI (Backend & MLOps Pipeline)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![DVC](https://img.shields.io/badge/DVC-Data_Version_Control-9cf)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)
![Flask](https://img.shields.io/badge/Flask-API-red)
![CI/CD](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF)

> **An End-to-End MLOps implementation for measuring Brand Safety and Influencer Impact on YouTube.**

## 🏗️ Architecture
This is not just a model; it is a fully automated **Continuous Training (CT)** pipeline.
1.  **Data Ingestion:** Custom scraper for YouTube Data API (No Kaggle datasets used).
2.  **Data Augmentation:** Used **OpenAI GPT** to generate synthetic minority-class samples (negative/nuanced comments) to solve class imbalance.
3.  **Pipeline Orchestration:** Managed via **DVC (Data Version Control)**.
4.  **Experiment Tracking:** **MLflow** & **Optuna** for hyperparameter tuning.
5.  **Deployment:** CI/CD via GitHub Actions triggers auto-deployment to **Render**.

## 🚀 Key Features
- **Custom Dataset:** Built from scratch to capture modern internet slang (2024-2025 era).
- **LLM-Based Data Labeling:** Automated the labeling process using LLMs to ensure consistency.
- **Reproducibility:** `dvc repro` runs the entire DAG (Ingest → Preprocess → Train) with caching.
- **Controversy Engine:** Calculates a custom "Controversy Score" based on sentiment variance.

## 🛠️ Tech Stack
- **Language:** Python
- **Orchestration:** DVC
- **Tracking:** MLflow
- **API:** Flask
- **Cloud:** Render

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
