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


def cv_metrics(X_df, y_ser):
    out = []
    for tr, va in cv.split(X_df):
        pipe.fit(X_df.iloc[tr], y_ser.iloc[tr])
        pred = pipe.predict(X_df.iloc[va])
        out.append(compute_metrics(y_ser.iloc[va], pred))
    return pd.DataFrame(out)


cv_real = cv_metrics(X_train, y_train)

pipe.fit(X_train, y_train)
test_pred = pipe.predict(X_test)
test_real = pd.Series(compute_metrics(y_test, test_pred))

rng = np.random.default_rng(5)
n_perm = 100

higher_is_better = {"explained_variance", "r2", "spearman_correlation"}
lower_is_better = {
    "mean_absolute_error",
    "median_absolute_error",
    "root_mean_squared_error",
    "normalized_mean_absolute_error",
    "normalized_median_absolute_error",
    "normalized_root_mean_squared_error",
}

cv_perm = []
test_perm = []

for _ in range(n_perm):
    y_perm = pd.Series(rng.permutation(y_train.values), index=y_train.index)

    cv_p = cv_metrics(X_train, y_perm)
    cv_perm.append(cv_p.mean())

    pipe.fit(X_train, y_perm)
    test_p = compute_metrics(y_test, pipe.predict(X_test))
    test_perm.append(test_p)

cv_perm = pd.DataFrame(cv_perm)
test_perm = pd.DataFrame(test_perm)

p_cv = {}
for m in cv_real.columns:
    obs = float(cv_real[m].mean())
    perm = cv_perm[m].to_numpy(dtype=float)
    if m in higher_is_better:
        p_cv[m] = (1 + np.sum(perm >= obs)) / (1 + n_perm)
    elif m in lower_is_better:
        p_cv[m] = (1 + np.sum(perm <= obs)) / (1 + n_perm)
    else:
        p_cv[m] = np.nan

p_test = {}
for m in test_real.index:
    obs = float(test_real[m])
    perm = test_perm[m].to_numpy(dtype=float)
    if m in higher_is_better:
        p_test[m] = (1 + np.sum(perm >= obs)) / (1 + n_perm)
    elif m in lower_is_better:
        p_test[m] = (1 + np.sum(perm <= obs)) / (1 + n_perm)
    else:
        p_test[m] = np.nan

print("\nCV real (mean ± std)")
cv_mean = cv_real.mean()
cv_std = cv_real.std()
for k in cv_real.columns:
    print(f"{k}: {cv_mean[k]:.6f} ± {cv_std[k]:.6f}")

print("\nTest real")
for k in test_real.index:
    print(f"{k}: {test_real[k]:.6f}")

print("\nCV permuted (mean ± std)")
cvp_mean = cv_perm.mean()
cvp_std = cv_perm.std()
for k in cv_perm.columns:
    print(f"{k}: {cvp_mean[k]:.6f} ± {cvp_std[k]:.6f}")

print("\nTest permuted (mean ± std)")
tsp_mean = test_perm.mean()
tsp_std = test_perm.std()
for k in test_perm.columns:
    print(f"{k}: {tsp_mean[k]:.6f} ± {tsp_std[k]:.6f}")

print("\nEmpirical p-values (CV)")
print(pd.Series(p_cv).round(6))

print("\nEmpirical p-values (Test)")
print(pd.Series(p_test).round(6))




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


def compute_metrics_binary(y_true, y_pred, y_proba):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_proba[:, 1])),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
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
    compute_metrics_binary(
        np.array([0, 1]),
        np.array([0, 1]),
        np.array([[0.9, 0.1], [0.1, 0.9]])
    ).keys()
)



higher_is_better = set(metric_names) - {"log_loss"}
lower_is_better = {"log_loss"}


def cv_metrics_binary(X_df, y_ser):
    out = []
    for tr_idx, va_idx in cv.split(X_df, y_ser):

        X_tr, X_va = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        y_tr, y_va = y_ser.iloc[tr_idx], y_ser.iloc[va_idx]

        pipe.fit(X_tr, y_tr)

        y_va_pred = pipe.predict(X_va)
        y_va_proba = pipe.predict_proba(X_va)

        out.append(compute_metrics_binary(y_va.to_numpy(), y_va_pred, y_va_proba))

    return pd.DataFrame(out)


cv_real = cv_metrics_binary(X_train, y_train)

pipe.fit(X_train, y_train)
y_test_pred = pipe.predict(X_test)
y_test_proba = pipe.predict_proba(X_test)
test_real = pd.Series(compute_metrics_binary(y_test.to_numpy(), y_test_pred, y_test_proba))


rng = np.random.default_rng(42)
n_perm = 100

cv_perm = []
test_perm = []

for _ in range(n_perm):
    y_perm = pd.Series(rng.permutation(y_train.values), index=y_train.index)

    cv_p = cv_metrics_binary(X_train, y_perm)
    cv_perm.append(cv_p.mean())

    pipe.fit(X_train, y_perm)
    y_test_pred_p = pipe.predict(X_test)
    y_test_proba_p = pipe.predict_proba(X_test)
    test_perm.append(compute_metrics_binary(y_test.to_numpy(), y_test_pred_p, y_test_proba_p))

cv_perm = pd.DataFrame(cv_perm)
test_perm = pd.DataFrame(test_perm)

p_cv = {}
for m in metric_names:
    obs = float(cv_real[m].mean())
    perm = cv_perm[m].to_numpy(dtype=float)
    if m in higher_is_better:
        p_cv[m] = (1 + np.sum(perm >= obs)) / (1 + n_perm)
    elif m in lower_is_better:
        p_cv[m] = (1 + np.sum(perm <= obs)) / (1 + n_perm)
    else:
        p_cv[m] = np.nan

p_test = {}
for m in metric_names:
    obs = float(test_real[m])
    perm = test_perm[m].to_numpy(dtype=float)
    if m in higher_is_better:
        p_test[m] = (1 + np.sum(perm >= obs)) / (1 + n_perm)
    elif m in lower_is_better:
        p_test[m] = (1 + np.sum(perm <= obs)) / (1 + n_perm)
    else:
        p_test[m] = np.nan

print("\n=== 10 fold StratifiedKFold CV results on training set (real) ===")
for k in metric_names:
    vals = cv_real[k].to_numpy(dtype=float)
    print(f"{k}: mean {np.nanmean(vals):.6f}   std {np.nanstd(vals):.6f}")

print("\n=== Test set results (real) ===")
for k in metric_names:
    print(f"{k}: {test_real[k]:.6f}")

print("\n=== 10 fold StratifiedKFold CV results on training set (permuted, mean ± std over permutations) ===")
for k in metric_names:
    vals = cv_perm[k].to_numpy(dtype=float)
    print(f"{k}: mean {np.nanmean(vals):.6f}   std {np.nanstd(vals):.6f}")

print("\n=== Test set results (permuted, mean ± std over permutations) ===")
for k in metric_names:
    vals = test_perm[k].to_numpy(dtype=float)
    print(f"{k}: mean {np.nanmean(vals):.6f}   std {np.nanstd(vals):.6f}")

print("\n=== Empirical p-values (CV) ===")
print(pd.Series(p_cv).round(6))

print("\n=== Empirical p-values (Test) ===")
print(pd.Series(p_test).round(6))