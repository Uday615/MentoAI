"""
ML Training Module - AI Mentor Decision Support System
Trains Logistic Regression, Random Forest, and Gradient Boosting classifiers
using aligned questionnaire features derived from the project CSV files.
"""
import json
import os
import pickle

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_alignment import DOMAIN_MAP, FEATURES, load_aligned_training_data

print("=" * 60)
print("AI Mentor Decision Support - ML Training Pipeline")
print("=" * 60)

print("\n[1/5] Loading and aligning CSV training data...")
df = load_aligned_training_data()

dist = df["Risk_Level"].value_counts().sort_index()
source_dist = df["Source"].value_counts().to_dict()
print(f"  Total samples: {len(df)}")
print(f"  Low Risk:      {dist.get(0, 0)} samples")
print(f"  Moderate Risk: {dist.get(1, 0)} samples")
print(f"  High Risk:     {dist.get(2, 0)} samples")
print(f"  Source split:  {source_dist}")

X = df[FEATURES]
y = df["Risk_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\n[2/5] Training all three models...")
models = {
    "Logistic Regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
        ]
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=12,
        min_samples_leaf=3,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        random_state=42,
        learning_rate=0.08,
        max_depth=3,
    ),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_model_name = None
best_f1 = 0.0

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    f1 = f1_score(y_test, preds, average="weighted")
    acc = accuracy_score(y_test, preds)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted")

    results[name] = {
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1, 4),
        "cv_mean": round(cv_scores.mean(), 4),
        "cv_std": round(cv_scores.std(), 4),
    }

    print(f"\n  [{name}]")
    print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f} | CV F1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(
        classification_report(
            y_test,
            preds,
            labels=[0, 1, 2],
            target_names=["Low Risk", "Moderate Risk", "High Risk"],
            zero_division=0,
        )
    )

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name

print(f"\n[3/5] Best Model: {best_model_name} (F1={best_f1:.4f})")

print("\n[4/5] Saving models and metadata...")
os.makedirs("ml", exist_ok=True)

best_model = models[best_model_name]
with open("ml/model.pkl", "wb") as f:
    pickle.dump(
        {
            "model": best_model,
            "model_name": best_model_name,
            "features": FEATURES,
            "domain_map": DOMAIN_MAP,
            "label_map": {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"},
            "training_source": "aligned_csv_forms",
        },
        f,
    )

with open("ml/all_models.pkl", "wb") as f:
    pickle.dump(
        {
            "models": models,
            "features": FEATURES,
            "domain_map": DOMAIN_MAP,
            "results": results,
            "best_model": best_model_name,
            "training_source": "aligned_csv_forms",
        },
        f,
    )

metrics_payload = {
    "model_results": results,
    "best_model": best_model_name,
    "feature_count": len(FEATURES),
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "risk_distribution": {str(k): int(v) for k, v in dist.to_dict().items()},
    "source_distribution": source_dist,
    "training_source": "aligned_csv_forms",
}

with open("ml/metrics.json", "w") as f:
    json.dump(metrics_payload, f, indent=2)

df.to_csv("ml/aligned_training_data.csv", index=False)

print(f"  Saved: ml/model.pkl (primary - {best_model_name})")
print("  Saved: ml/all_models.pkl (all 3 models)")
print("  Saved: ml/metrics.json (performance report)")
print("  Saved: ml/aligned_training_data.csv (feature-aligned dataset)")

print("\n[5/5] Training complete!")
print("=" * 60)
