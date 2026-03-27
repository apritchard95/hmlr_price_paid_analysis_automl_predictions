# HMLR Property Price Analysis

End-to-end analysis of UK residential property transactions using HM Land Registry's publicly available Price Paid Dataset (2024–2025).

## Overview

This project covers three stages:

1. **Exploratory Data Analysis** — price distributions, transaction volumes, regional variation, property type breakdowns, new build vs established, and year-on-year comparison
2. **Hypothesis Testing** — statistical tests investigating whether observed differences in property prices are significant (Kruskal-Wallis, Welch's t-test, Mann-Whitney U, Chi-squared)
3. **AutoML Classification** — PyCaret model comparison leaderboard predicting property price band from transaction features

## Data

HM Land Registry Price Paid Data — published under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

The data is downloaded programmatically in the notebook from HMLR's public data endpoint. No manual download required — run the notebook and the data will be fetched automatically into a local `data/` folder (not tracked in this repo due to file size).

Coverage: all standard residential property transactions in England and Wales registered with HMLR in 2024 and 2025.

## Key Features

- Automated data download and preprocessing pipeline
- Outlier filtering and categorical encoding
- Price band classification target defined from national quartiles
- Four hypothesis tests with stated H₀/H₁, chosen test with justification, and written interpretation
- PyCaret AutoML comparison across multiple classifiers with 5-fold cross-validation
- Feature importance and confusion matrix outputs (see `Feature Importance.png` and `Confusion Matrix.png`)

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── hmlr_property_price_analysis.ipynb
├── Feature Importance.png
└── Confusion Matrix.png
```

## Running the Notebook

```bash
pip install -r requirements.txt
jupyter notebook hmlr_property_price_analysis.ipynb
```

The data download cells will fetch ~200MB of CSV data on first run. Subsequent runs will skip the download if the files are already present.

## Technical Stack

- Python 3.10
- `pycaret` — AutoML model comparison and tuning
- `scipy` — statistical hypothesis tests
- `pandas`, `numpy` — data processing
- `matplotlib`, `seaborn` — visualisation
