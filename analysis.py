# ─────────────────────────────────────────────
# CUSTOMER CHURN ANALYSIS
# Phase 1: Load & Clean | Phase 2: EDA
# ─────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os

# ── Settings ──────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.family"] = "sans-serif"
OUTPUT = "output"
os.makedirs(OUTPUT, exist_ok=True)

# ── Load ───────────────────────────────────────
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Shape:", df.shape)
print("\nColumn names:\n", df.columns.tolist())
print("\nFirst look:\n", df.head(3))
print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
print("\nData types:\n", df.dtypes)

# ── Clean ──────────────────────────────────────

# TotalCharges is stored as a string — convert to number
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Those coerced NaNs are new customers with 0 tenure — fill with 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Convert Churn to binary (1 = churned, 0 = stayed)
df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)

# Tenure groups — useful for segmentation later
df["Tenure_Group"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0–12 months", "13–24 months", "25–48 months", "49–72 months"]
)

# Charge band — high vs low spenders
df["Charge_Band"] = pd.cut(
    df["MonthlyCharges"],
    bins=[0, 35, 65, 120],
    labels=["Low (<$35)", "Mid ($35–65)", "High (>$65)"]
)

print("\n✅ Cleaning done.")
print("Churn rate overall: {:.1f}%".format(df["Churn_Binary"].mean() * 100))
print("TotalCharges nulls remaining:", df["TotalCharges"].isnull().sum())
print("\nTenure groups:\n", df["Tenure_Group"].value_counts())
print("\nCharge bands:\n", df["Charge_Band"].value_counts())

# ─────────────────────────────────────────────
# PHASE 2: Exploratory Data Analysis
# ─────────────────────────────────────────────

def save(fig, name):
    fig.savefig(f"{OUTPUT}/{name}.png", bbox_inches="tight")
    print(f"  ✅ Saved: {name}.png")
    plt.close(fig)


# ── Chart 1: Overall churn rate ───────────────
fig, ax = plt.subplots(figsize=(5, 4))
counts = df["Churn"].value_counts()
colors = ["#2ecc71", "#e74c3c"]
bars = ax.bar(counts.index, counts.values, color=colors, width=0.5, edgecolor="white")
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 60,
            f"{val:,}\n({val/len(df)*100:.1f}%)", ha="center", va="bottom", fontsize=11)
ax.set_title("Overall churn rate", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Number of customers")
ax.set_ylim(0, 6500)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "01_overall_churn")


# ── Chart 2: Churn by contract type ──────────
fig, ax = plt.subplots(figsize=(7, 4))
ct = df.groupby("Contract")["Churn_Binary"].mean().sort_values(ascending=False) * 100
bars = ax.barh(ct.index, ct.values, color=["#e74c3c", "#e67e22", "#2ecc71"], edgecolor="white")
for bar, val in zip(bars, ct.values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=11)
ax.set_title("Churn rate by contract type", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Churn rate (%)")
ax.set_xlim(0, 55)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "02_churn_by_contract")


# ── Chart 3: Churn by tenure group ────────────
fig, ax = plt.subplots(figsize=(7, 4))
tg = df.groupby("Tenure_Group", observed=True)["Churn_Binary"].mean() * 100
bars = ax.bar(tg.index, tg.values,
              color=["#e74c3c", "#e67e22", "#3498db", "#2ecc71"],
              edgecolor="white", width=0.6)
for bar, val in zip(bars, tg.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=11)
ax.set_title("Churn rate by customer tenure", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Churn rate (%)")
ax.set_ylim(0, 60)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "03_churn_by_tenure")


# ── Chart 4: Churn by monthly charge band ─────
fig, ax = plt.subplots(figsize=(7, 4))
cb = df.groupby("Charge_Band", observed=True)["Churn_Binary"].mean() * 100
bars = ax.bar(cb.index, cb.values,
              color=["#2ecc71", "#e67e22", "#e74c3c"],
              edgecolor="white", width=0.6)
for bar, val in zip(bars, cb.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=11)
ax.set_title("Churn rate by monthly charge band", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Churn rate (%)")
ax.set_ylim(0, 45)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "04_churn_by_charge")


# ── Chart 5: Churn by internet service ────────
fig, ax = plt.subplots(figsize=(7, 4))
inet = df.groupby("InternetService")["Churn_Binary"].mean().sort_values(ascending=False) * 100
bars = ax.barh(inet.index, inet.values,
               color=["#e74c3c", "#e67e22", "#2ecc71"], edgecolor="white")
for bar, val in zip(bars, inet.values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=11)
ax.set_title("Churn rate by internet service type", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Churn rate (%)")
ax.set_xlim(0, 55)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "05_churn_by_internet")


# ── Chart 6: Correlation heatmap ──────────────
fig, ax = plt.subplots(figsize=(8, 5))
num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Churn_Binary"]
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, mask=mask, ax=ax,
            linewidths=0.5, square=True,
            annot_kws={"size": 11})
ax.set_title("Correlation matrix — key numeric features", fontsize=13, fontweight="bold", pad=12)
save(fig, "06_correlation_heatmap")


print("\n🎉 Phase 2 complete! Check your output/ folder for all 6 charts.")

# ─────────────────────────────────────────────
# PHASE 3: Predictive Model
# ─────────────────────────────────────────────

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

print("\n── Phase 3: Building the model ──────────────")

# ── Prepare features ──────────────────────────
# Drop columns we don't need for modelling
drop_cols = ["customerID", "Churn", "Tenure_Group", "Charge_Band"]
df_model = df.drop(columns=drop_cols)

# Encode all categorical columns as numbers
le = LabelEncoder()
cat_cols = df_model.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    df_model[col] = le.fit_transform(df_model[col])

# Split into features (X) and target (y)
X = df_model.drop(columns=["Churn_Binary"])
y = df_model["Churn_Binary"]

# 80% train, 20% test — random_state ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} customers")
print(f"Test set:     {X_test.shape[0]} customers")


# ── Model 1: Logistic Regression ──────────────
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_proba = lr.predict_proba(X_test)[:, 1]
lr_auc   = roc_auc_score(y_test, lr_proba)

print(f"\nLogistic Regression AUC: {lr_auc:.3f}")
print(classification_report(y_test, lr_preds, target_names=["Stayed", "Churned"]))


# ── Model 2: Random Forest ────────────────────
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_auc   = roc_auc_score(y_test, rf_proba)

print(f"Random Forest AUC:        {rf_auc:.3f}")
print(classification_report(y_test, rf_preds, target_names=["Stayed", "Churned"]))


# ── Chart 7: ROC Curve (both models) ─────────
fig, ax = plt.subplots(figsize=(6, 5))
for proba, label, color in [
    (lr_proba, f"Logistic Regression (AUC={lr_auc:.2f})", "#3498db"),
    (rf_proba, f"Random Forest (AUC={rf_auc:.2f})",        "#e74c3c"),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, label=label, linewidth=2, color=color)

ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random baseline")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curve — model comparison", fontsize=13, fontweight="bold", pad=12)
ax.legend(loc="lower right", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "07_roc_curve")


# ── Chart 8: Confusion matrix (Random Forest) ─
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, lr_preds),
    display_labels=["Stayed", "Churned"]
).plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion matrix — Logistic Regression", fontsize=13, fontweight="bold", pad=12)
save(fig, "08_confusion_matrix")


# ── Chart 9: Feature importance (Random Forest)
fig, ax = plt.subplots(figsize=(8, 6))
importances = pd.Series(rf.feature_importances_, index=X.columns)
top10 = importances.sort_values(ascending=True).tail(10)
colors = ["#e74c3c" if i == top10.index[-1] else "#3498db" for i in top10.index]
top10.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
ax.set_title("Top 10 features predicting churn (Random Forest)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Feature importance score")
ax.spines[["top", "right"]].set_visible(False)
save(fig, "09_feature_importance")


# ── Attach churn probability to original data ──
# Use the full dataset (not just test set) for the dashboard
df["Churn_Probability"] = lr.predict_proba(
    df_model.drop(columns=["Churn_Binary"])
)[:, 1]

df["Risk_Level"] = pd.cut(
    df["Churn_Probability"],
    bins=[0, 0.3, 0.6, 1.0],
    labels=["Low", "Medium", "High"]
)

# ── Revenue at risk calculation ────────────────
churners = df[df["Risk_Level"] == "High"]
revenue_at_risk = churners["MonthlyCharges"].sum()
customer_count  = len(churners)

print(f"\n💰 High-risk customers:  {customer_count:,}")
print(f"💸 Monthly revenue at risk: ${revenue_at_risk:,.0f}")
print(f"📅 Annual revenue at risk:  ${revenue_at_risk * 12:,.0f}")

# Save enriched dataset for dashboard
df.to_csv("output/churn_predictions.csv", index=False)
print("\n✅ Predictions saved to output/churn_predictions.csv")
print("🎉 Phase 3 complete!")