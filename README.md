# Customer Churn Prediction

Predicting which telecom customers are likely to churn, using classification
models and EDA to surface the business drivers behind churn.

## Dataset

`data/telco_churn.csv` — 7,000 customers, 21 features (demographics, account
info, subscribed services, contract/billing details), 26% churn rate.

> **Note on data source:** this project was built in a sandboxed environment
> without live internet access, so `data/generate_data.py` synthesizes a
> dataset that mirrors the schema and statistical patterns of the real
> [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
> (same columns, same realistic churn drivers baked in: month-to-month
> contracts, low tenure, no tech support, electronic-check billing, etc.).
> To use the real dataset instead, download it from Kaggle and save it as
> `data/telco_churn.csv` with the same column names — the rest of the
> pipeline runs unchanged.

## Approach

1. **EDA** (`outputs/eda_overview.png`) — churn rate by contract type,
   tenure distribution, monthly charges, and internet service type.
2. **Preprocessing** — numeric scaling + one-hot encoding via an
   sklearn `ColumnTransformer`/`Pipeline`.
3. **Models** — Logistic Regression, Random Forest, and Gradient Boosting
   (used here as a drop-in for XGBoost, since this sandbox has no internet
   access to install the `xgboost` package — swap in
   `xgboost.XGBClassifier()` locally for an identical-interface upgrade).
4. **Evaluation** — Accuracy, Recall, ROC-AUC, confusion matrix, ROC curves.
5. **Interpretability** — permutation feature importance on the best model.

## Results

| Model | Accuracy | Recall | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.704 | **0.733** | **0.785** |
| Random Forest | 0.721 | 0.692 | 0.777 |
| Gradient Boosting | **0.768** | 0.314 | 0.775 |

Logistic Regression gave the best ROC-AUC and by far the best recall on the
churn class — important here because **missing a churner (false negative)
is usually more costly to the business than a false alarm** (a retention
offer sent to a loyal customer is cheap; losing a customer silently is not).
Gradient Boosting had the highest raw accuracy but at the cost of missing
70% of actual churners — a bad tradeoff for this use case.

## Key business insights (from EDA + feature importance)

- **Contract type is the single biggest churn driver** — month-to-month
  customers churn far more than one/two-year contract holders. Incentivizing
  longer contracts (discounts for annual commitment) is the highest-leverage
  retention lever.
- **Low-tenure customers churn most** — the first few months are the highest
  risk window; an onboarding/retention touchpoint in month 1–3 could help.
- **Fiber optic customers churn more than DSL** — possibly a
  price-to-perceived-value gap worth investigating.
- **No tech support subscription correlates with higher churn** — bundling
  or upselling tech support may reduce churn.

## Files

```
customer_churn/
├── data/
│   ├── generate_data.py     # synthetic dataset generator
│   └── telco_churn.csv      # generated dataset
├── churn_analysis.py        # full EDA + modeling pipeline
├── outputs/
│   ├── eda_overview.png
│   ├── roc_curves.png
│   ├── confusion_matrix_best_model.png
│   ├── feature_importance.png
│   └── model_comparison.csv
└── README.md
```

## Run it yourself

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python data/generate_data.py
python churn_analysis.py
```

## Tech stack
Python, pandas, scikit-learn, matplotlib, seaborn
