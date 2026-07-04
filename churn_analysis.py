"""
Customer Churn Prediction
=========================
Goal: Predict which customers are likely to stop using a telecom service.

Pipeline:
1. Load + clean data
2. EDA - churn patterns across contract type, tenure, charges, services
3. Feature engineering + preprocessing
4. Train Logistic Regression, Random Forest, Gradient Boosting (XGBoost stand-in)
5. Evaluate with Accuracy, Recall, ROC-AUC + confusion matrix + ROC curves
6. Feature importance for business interpretation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
from sklearn.inspection import permutation_importance

sns.set_theme(style="whitegrid")
OUT = "/home/claude/customer_churn/outputs"

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
df = pd.read_csv("/home/claude/customer_churn/data/telco_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])
df["Churn_flag"] = (df["Churn"] == "Yes").astype(int)

print("Dataset shape:", df.shape)
print(df["Churn"].value_counts())

# ------------------------------------------------------------------
# 2. EDA
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Churn rate by contract
churn_by_contract = df.groupby("Contract")["Churn_flag"].mean().sort_values()
churn_by_contract.plot(kind="bar", ax=axes[0, 0], color="#c0392b")
axes[0, 0].set_title("Churn Rate by Contract Type")
axes[0, 0].set_ylabel("Churn Rate")
axes[0, 0].tick_params(axis="x", rotation=20)

# Tenure distribution by churn
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True, ax=axes[0, 1], palette=["#2ecc71", "#c0392b"])
axes[0, 1].set_title("Tenure Distribution by Churn Status")

# Monthly charges by churn
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[1, 0], palette=["#2ecc71", "#c0392b"])
axes[1, 0].set_title("Monthly Charges by Churn Status")

# Churn rate by internet service
churn_by_internet = df.groupby("InternetService")["Churn_flag"].mean().sort_values()
churn_by_internet.plot(kind="bar", ax=axes[1, 1], color="#2980b9")
axes[1, 1].set_title("Churn Rate by Internet Service")
axes[1, 1].set_ylabel("Churn Rate")

plt.tight_layout()
plt.savefig(f"{OUT}/eda_overview.png", dpi=150)
plt.close()
print("Saved EDA overview plot.")

# ------------------------------------------------------------------
# 3. Preprocessing
# ------------------------------------------------------------------
target = "Churn_flag"
drop_cols = ["customerID", "Churn", "Churn_flag"]
X = df.drop(columns=drop_cols)
y = df[target]

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------------
# 4. Models
# ------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced"),
    # Gradient Boosting used as an XGBoost-equivalent (no internet to install xgboost in this sandbox);
    # swap for xgboost.XGBClassifier() locally if you have it installed - same interface.
    "Gradient Boosting (XGBoost-equivalent)": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42),
}

results = {}
plt.figure(figsize=(7, 6))

for name, clf in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results[name] = {"pipeline": pipe, "accuracy": acc, "recall": rec, "roc_auc": auc,
                      "y_pred": y_pred, "y_proba": y_proba}

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.3f}  Recall: {rec:.3f}  ROC-AUC: {auc:.3f}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Model Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/roc_curves.png", dpi=150)
plt.close()
print("Saved ROC curve comparison.")

# ------------------------------------------------------------------
# 5. Confusion matrix for best model (by ROC-AUC)
# ------------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["roc_auc"])
best = results[best_name]
cm = confusion_matrix(y_test, best["y_pred"])

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title(f"Confusion Matrix - {best_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrix_best_model.png", dpi=150)
plt.close()
print(f"\nBest model by ROC-AUC: {best_name}")

# ------------------------------------------------------------------
# 6. Feature importance (permutation importance, model-agnostic)
# ------------------------------------------------------------------
best_pipe = best["pipeline"]
perm = permutation_importance(best_pipe, X_test, y_test, n_repeats=10, random_state=42, scoring="roc_auc")
feat_names = X_test.columns  # permutation importance runs on raw pipeline inputs
imp_df = pd.DataFrame({"feature": feat_names, "importance": perm.importances_mean}).sort_values(
    "importance", ascending=False).head(15)

plt.figure(figsize=(8, 6))
sns.barplot(data=imp_df, x="importance", y="feature", color="#8e44ad")
plt.title(f"Top 15 Feature Importances - {best_name}")
plt.tight_layout()
plt.savefig(f"{OUT}/feature_importance.png", dpi=150)
plt.close()
print("Saved feature importance plot.")

# ------------------------------------------------------------------
# 7. Summary table
# ------------------------------------------------------------------
summary = pd.DataFrame({
    name: {"Accuracy": r["accuracy"], "Recall": r["recall"], "ROC-AUC": r["roc_auc"]}
    for name, r in results.items()
}).T
summary.to_csv(f"{OUT}/model_comparison.csv")
print("\n=== Model Comparison ===")
print(summary.round(3))
