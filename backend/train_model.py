import json
import pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from xgboost import XGBClassifier
import joblib

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data/features.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

df = pd.read_csv(DATA)

# Train on all seasons - the latest
latest_season = df["season"].max()
train = df[df["season"] < latest_season].copy()
test  = df[df["season"] == latest_season].copy()

feat_cols = [c for c in df.columns if c not in ("win", "season")]
X_tr, y_tr = train[feat_cols], train["win"]
X_te, y_te = test[feat_cols], test["win"]

# class imbalance
neg, pos = np.bincount(y_tr)
scale_pos_weight = neg / max(pos, 1)

model = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=2.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    scale_pos_weight=scale_pos_weight,
)

model.fit(X_tr, y_tr)

p_te = model.predict_proba(X_te)[:, 1]
print(f"Accuracy: {accuracy_score(y_te, (p_te >= 0.5).astype(int)):.3f}")
print(f"AUC: {roc_auc_score(y_te, p_te):.3f}")
print(f"LogLoss: {log_loss(y_te, p_te):.4f}")


joblib.dump(model, MODEL_DIR / "model.pkl")
json.dump(feat_cols, open(MODEL_DIR / "features.json", "w"))
print("Saved model.pkl and features.json")
