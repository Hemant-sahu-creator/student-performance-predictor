"""
Student Performance Predictor - ML Training Pipeline
Author: Hemant Sahu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
import pickle
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC DATASET
# ─────────────────────────────────────────────
def generate_dataset(n=1000, save_path="data/students.csv"):
    np.random.seed(42)
    data = {
        "gender":            np.random.choice(["male", "female"], n),
        "race_ethnicity":    np.random.choice(["group A","group B","group C","group D","group E"], n),
        "parental_education":np.random.choice(
            ["some high school","high school","some college",
             "associate's degree","bachelor's degree","master's degree"], n),
        "lunch":             np.random.choice(["standard","free/reduced"], n),
        "test_preparation":  np.random.choice(["none","completed"], n),
        "math_score":        np.clip(np.random.normal(66, 15, n).astype(int), 0, 100),
        "reading_score":     np.clip(np.random.normal(69, 14, n).astype(int), 0, 100),
        "writing_score":     np.clip(np.random.normal(68, 15, n).astype(int), 0, 100),
    }
    df = pd.DataFrame(data)
    # Boost scores slightly for completed test prep
    mask = df["test_preparation"] == "completed"
    for col in ["math_score","reading_score","writing_score"]:
        df.loc[mask, col] = np.clip(df.loc[mask, col] + np.random.randint(3, 10, mask.sum()), 0, 100)

    df["average_score"] = df[["math_score","reading_score","writing_score"]].mean(axis=1).round(2)
    df["pass_fail"]      = (df["average_score"] >= 50).astype(int)   # 1=Pass, 0=Fail

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ Dataset saved → {save_path}  |  Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
def run_eda(df):
    print("\n" + "="*55)
    print("  EXPLORATORY DATA ANALYSIS")
    print("="*55)
    print(f"\n📊 Shape        : {df.shape}")
    print(f"🔍 Null values  :\n{df.isnull().sum()}")
    print(f"\n📈 Basic Stats  :\n{df.describe().round(2)}")
    print(f"\n🎯 Pass/Fail    :\n{df['pass_fail'].value_counts()}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Student Performance – EDA Dashboard", fontsize=16, fontweight="bold")

    # Score distributions
    for ax, col, color in zip(axes[0],
                               ["math_score","reading_score","writing_score"],
                               ["#4C72B0","#DD8452","#55A868"]):
        ax.hist(df[col], bins=20, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(df[col].mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean={df[col].mean():.1f}")
        ax.set_title(col.replace("_"," ").title())
        ax.legend(fontsize=9)

    # Pass/Fail pie
    counts = df["pass_fail"].value_counts()
    axes[1][0].pie(counts, labels=["Pass","Fail"], autopct="%1.1f%%",
                   colors=["#55A868","#C44E52"], startangle=90, textprops={"fontsize":11})
    axes[1][0].set_title("Pass / Fail Split")

    # Avg score by test prep
    df.groupby("test_preparation")["average_score"].mean().plot(
        kind="bar", ax=axes[1][1], color=["#4C72B0","#DD8452"], edgecolor="white")
    axes[1][1].set_title("Avg Score by Test Preparation")
    axes[1][1].set_xticklabels(axes[1][1].get_xticklabels(), rotation=0)
    axes[1][1].set_ylabel("Average Score")

    # Correlation heatmap
    corr = df[["math_score","reading_score","writing_score","average_score"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues",
                ax=axes[1][2], linewidths=0.5, square=True)
    axes[1][2].set_title("Score Correlation Heatmap")

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/eda_dashboard.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("\n✅ EDA plots saved → plots/eda_dashboard.png")


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    # Score spread & total
    df["score_total"]   = df["math_score"] + df["reading_score"] + df["writing_score"]
    df["score_spread"]  = df[["math_score","reading_score","writing_score"]].std(axis=1).round(2)
    df["verbal_avg"]    = ((df["reading_score"] + df["writing_score"]) / 2).round(2)

    # Encode categoricals
    le = LabelEncoder()
    cat_cols = ["gender","race_ethnicity","parental_education","lunch","test_preparation"]
    for col in cat_cols:
        df[col + "_enc"] = le.fit_transform(df[col])

    print("✅ Feature engineering complete.")
    return df


# ─────────────────────────────────────────────
# 4. TRAIN & EVALUATE MODELS
# ─────────────────────────────────────────────
def train_models(df):
    feature_cols = [
        "gender_enc","race_ethnicity_enc","parental_education_enc",
        "lunch_enc","test_preparation_enc",
        "math_score","reading_score","writing_score",
        "score_total","score_spread","verbal_avg"
    ]
    X = df[feature_cols]
    y = df["pass_fail"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "Logistic Regression":      LogisticRegression(max_iter=500),
        "Random Forest":            RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":        GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    print("\n" + "="*55)
    print("  MODEL TRAINING & EVALUATION")
    print("="*55)

    results = {}
    best_acc, best_name, best_model = 0, "", None

    for name, model in models.items():
        X_tr = X_train_s if name == "Logistic Regression" else X_train
        X_te = X_test_s  if name == "Logistic Regression" else X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        acc    = accuracy_score(y_test, y_pred)
        cv     = cross_val_score(model, X_tr, y_train, cv=5, scoring="accuracy").mean()

        results[name] = {"accuracy": acc, "cv_score": cv}
        print(f"\n🔹 {name}")
        print(f"   Accuracy  : {acc:.4f}")
        print(f"   CV Score  : {cv:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Fail","Pass"]))

        if acc > best_acc:
            best_acc, best_name, best_model = acc, name, model

    print(f"\n🏆 Best Model : {best_name}  (Accuracy = {best_acc:.4f})")

    # Feature importance plot (Random Forest)
    rf = models["Random Forest"]
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    importances.plot(kind="bar", color="#4C72B0", edgecolor="white")
    plt.title("Feature Importances – Random Forest", fontweight="bold")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png", dpi=120)
    plt.close()
    print("✅ Feature importance plot saved → plots/feature_importance.png")

    return best_model, scaler, feature_cols, X_test, y_test


# ─────────────────────────────────────────────
# 5. SAVE MODEL
# ─────────────────────────────────────────────
def save_model(model, scaler, feature_cols):
    os.makedirs("models", exist_ok=True)
    with open("models/best_model.pkl",   "wb") as f: pickle.dump(model, f)
    with open("models/scaler.pkl",       "wb") as f: pickle.dump(scaler, f)
    with open("models/feature_cols.pkl", "wb") as f: pickle.dump(feature_cols, f)
    print("\n✅ Model artifacts saved → models/")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df              = generate_dataset()
    run_eda(df)
    df_fe           = engineer_features(df)
    model, scaler, feature_cols, X_test, y_test = train_models(df_fe)
    save_model(model, scaler, feature_cols)
    print("\n🎉 Pipeline complete! Run `python src/app.py` to start the API.")
