# %%
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    mean_squared_error,
)
from scipy.stats import spearmanr
from xgboost import XGBRegressor


def safe_spearman(y_true, y_pred):
    r = spearmanr(y_true, y_pred)[0]
    return 0.0 if np.isnan(r) else float(r)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def nmae(y_true, y_pred):
    denom = np.mean(np.abs(y_true))
    return float(mean_absolute_error(y_true, y_pred) / denom) if denom != 0 else np.nan


def nmedae(y_true, y_pred):
    denom = np.mean(np.abs(y_true))
    return float(median_absolute_error(y_true, y_pred) / denom) if denom != 0 else np.nan


def nrmse(y_true, y_pred):
    denom = np.mean(np.abs(y_true))
    return float(rmse(y_true, y_pred) / denom) if denom != 0 else np.nan


def compute_metrics(y_true, y_pred):
    return {
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
        "mean_absolute_error": float(mean_absolute_error(y_true, y_pred)),
        "median_absolute_error": float(median_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "root_mean_squared_error": rmse(y_true, y_pred),
        "spearman_correlation": safe_spearman(y_true, y_pred),
        "normalized_mean_absolute_error": nmae(y_true, y_pred),
        "normalized_median_absolute_error": nmedae(y_true, y_pred),
        "normalized_root_mean_squared_error": nrmse(y_true, y_pred),
    }


df = pd.read_csv("filtered_sweeteners_data_307_features.csv")

X = df.iloc[:, 1:-1]
y = df["Sweetness"]

bit_columns = [c for c in X.columns if "Bit" in c]
mordred_columns = [c for c in X.columns if "Bit" not in c]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=5
)

preprocess = ColumnTransformer(
    transformers=[
        ("mordred_scaling", MinMaxScaler(), mordred_columns),
        ("bit_passthrough", "passthrough", bit_columns),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

model = XGBRegressor(
    objective="reg:squarederror",
    random_state=5,
    colsample_bytree=0.6,
    gamma=0.0,
    learning_rate=0.03,
    max_delta_step=4,
    max_depth=6,
    min_child_weight=4,
    n_estimators=250,
    subsample=0.8,
    reg_alpha=0.01,
    reg_lambda=1.0,
)

pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", model),
    ]
)

cv = KFold(n_splits=10, shuffle=True, random_state=5)

metric_names = list(compute_metrics(y_train.iloc[:2], y_train.iloc[:2]).keys())
cv_metrics = {k: [] for k in metric_names}

for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_train), start=1):
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    pipe.fit(X_tr, y_tr)
    y_va_pred = pipe.predict(X_va)

    m = compute_metrics(y_va, y_va_pred)
    for k, v in m.items():
        cv_metrics[k].append(v)

print("\n=== 10 fold CV results on training split ===")
for k in metric_names:
    vals = np.array(cv_metrics[k], dtype=float)
    print(f"{k}: mean {np.nanmean(vals):.6f}   std {np.nanstd(vals):.6f}")

pipe.fit(X_train, y_train)
y_test_pred = pipe.predict(X_test)
test_m = compute_metrics(y_test, y_test_pred)

print("\n=== Test set results ===")
for k in metric_names:
    print(f"{k}: {test_m[k]:.6f}")
    
# %%

    
    
# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier


def compute_metrics_binary(y_true, y_pred, y_prob):

    auc = float(roc_auc_score(y_true, y_prob))
    ap = float(average_precision_score(y_true, y_prob))

    

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": auc,
        "average_precision": ap,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


train_df = pd.read_csv("filtered_sweet_nonsweet_class_train_set.csv")
test_df = pd.read_csv("filtered_sweet_nonsweet_class_test_set.csv")

X_train = train_df.drop(columns=["Name", "Class"])
y_train = train_df["Class"].astype(int)

X_test = test_df.drop(columns=["Name", "Class"])
y_test = test_df["Class"].astype(int)

bit_columns = [c for c in X_train.columns if "Bit" in c]
mordred_columns = [c for c in X_train.columns if "Bit" not in c]

preprocess = ColumnTransformer(
    transformers=[
        ("mordred_scaling", StandardScaler(), mordred_columns),
        ("bit_passthrough", "passthrough", bit_columns),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    colsample_bytree=0.6,
    gamma=0.2,
    learning_rate=0.015,
    max_depth=4,
    min_child_weight=9,
    n_estimators=450,
    reg_alpha=0.01,
    reg_lambda=0.1,
    scale_pos_weight=5,
    subsample=0.55,
)

pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", model),
    ]
)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


metric_names = list(
    compute_metrics_binary(np.array([0, 1]), np.array([0, 1]), np.array([0.1, 0.9])).keys()
)
cv_metrics = {k: [] for k in metric_names}

for tr_idx, va_idx in cv.split(X_train, y_train):

    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    pipe.fit(X_tr, y_tr)

    y_va_pred = pipe.predict(X_va)
    y_va_prob = pipe.predict_proba(X_va)[:, 1]

    m = compute_metrics_binary(y_va.to_numpy(), y_va_pred, y_va_prob)
    for k, v in m.items():
        cv_metrics[k].append(v)

print("\n=== 10 fold StratifiedKFold CV results on training set ===")
for k in metric_names:
    vals = np.array(cv_metrics[k], dtype=float)
    print(f"{k}: mean {np.nanmean(vals):.6f}   std {np.nanstd(vals):.6f}")

pipe.fit(X_train, y_train)
y_test_pred = pipe.predict(X_test)
y_test_prob = pipe.predict_proba(X_test)[:, 1]

test_m = compute_metrics_binary(y_test.to_numpy(), y_test_pred, y_test_prob)

print("\n=== Test set results ===")
for k in metric_names:
    print(f"{k}: {test_m[k]:.6f}")


