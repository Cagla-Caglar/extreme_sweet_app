# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import xgboost as xgb
import lightgbm as lgb
import json
import pickle
import os

output_dir = "Voting_Ensemble_and_XGBoost_Regression_Models"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("sweeteners_with_molecular_descriptors_and_fingerprints.csv")
X = df.drop(columns=["Molecule", "Sweetness"])
y = df["Sweetness"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
kfold = KFold(n_splits=10, shuffle=True, random_state=42)


def nrmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return float(rmse / (y_true.max() - y_true.min()))

def spearman_r(a, b):
    r = spearmanr(a, b).correlation
    return float(r) if r is not None else float("nan")

model_specs = [
    ("ExtraTreesRegressor_StandardScaler_25",
     Pipeline([("scaler", StandardScaler()),
               ("model", ExtraTreesRegressor(
                   n_estimators=25,
                   max_features=0.6,
                   min_samples_leaf=0.0160,
                   min_samples_split=0.0260,
                   bootstrap=False,
                   criterion="squared_error",
                   random_state=42
               ))]),
     0.07),

    ("ExtraTreesRegressor_RobustScaler_100",
     Pipeline([("scaler", RobustScaler()),
               ("model", ExtraTreesRegressor(
                   n_estimators=100,
                   max_features=0.6,
                   min_samples_leaf=0.0090,
                   min_samples_split=0.0153,
                   bootstrap=True,
                   criterion="squared_error",
                   random_state=42
               ))]),
     0.07),

    ("ExtraTreesRegressor_StandardScaler_200",
     Pipeline([("scaler", StandardScaler()),
               ("model", ExtraTreesRegressor(
                   n_estimators=200,
                   max_features="sqrt",
                   min_samples_leaf=0.0029,
                   min_samples_split=0.0037,
                   bootstrap=False,
                   criterion="squared_error",
                   random_state=42
               ))]),
     0.33),

    ("DecisionTreeRegressor_RobustScaler",
     Pipeline([("scaler", RobustScaler()),
               ("model", DecisionTreeRegressor(
                   criterion="friedman_mse",
                   splitter="best",
                   max_features=None,
                   min_samples_leaf=0.0119,
                   min_samples_split=0.0529,
                   random_state=42
               ))]),
     0.07),

    ("ExtraTreesRegressor_MinMaxScaler_50",
     Pipeline([("scaler", MinMaxScaler()),
               ("model", ExtraTreesRegressor(
                   n_estimators=50,
                   max_features=0.5,
                   min_samples_leaf=0.0051,
                   min_samples_split=0.0013,
                   bootstrap=False,
                   criterion="squared_error",
                   random_state=42
               ))]),
     0.07),

    ("ExtraTreesRegressor_MinMaxScaler_25",
     Pipeline([("scaler", MinMaxScaler()),
               ("model", ExtraTreesRegressor(
                   n_estimators=25,
                   max_features=0.7,
                   min_samples_leaf=0.0035,
                   min_samples_split=0.0018,
                   bootstrap=True,
                   criterion="squared_error",
                   random_state=42
               ))]),
     0.07),

    ("ElasticNet_MinMaxScaler",
     Pipeline([("scaler", MinMaxScaler()),
               ("model", ElasticNet(
                   alpha=0.001,
                   l1_ratio=0.8437,
                   random_state=42
               ))]),
     0.13),

    ("XGBoostRegressor_MaxAbsScaler",
     Pipeline([("scaler", MaxAbsScaler()),
               ("model", xgb.XGBRegressor(
                   tree_method="auto",
                   random_state=42
               ))]),
     0.07),

    ("LightGBMRegressor_MaxAbsScaler",
     Pipeline([("scaler", MaxAbsScaler()),
               ("model", lgb.LGBMRegressor(
                   min_data_in_leaf=20,
                   random_state=42
               ))]),
     0.13),
]

model_names = [n for n, _, _ in model_specs]
weights = [w for _, _, w in model_specs]
w_arr = np.array(weights, dtype=float)
wsum = float(w_arr.sum())

cv_r2_list, cv_mae_list, cv_rmse_list, cv_nrmse_list, cv_spearman_list = [], [], [], [], []
test_r2_list, test_mae_list, test_rmse_list, test_nrmse_list, test_spearman_list = [], [], [], [], []

for name, pipe, _ in model_specs:
    oof = np.empty(len(y_train), dtype=float)
    for tr_idx, va_idx in kfold.split(X_train):
        m = clone(pipe)
        m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        oof[va_idx] = m.predict(X_train.iloc[va_idx])

    cv_r2_list.append(float(r2_score(y_train, oof)))
    cv_mae_list.append(float(mean_absolute_error(y_train, oof)))
    cv_rmse_list.append(float(np.sqrt(mean_squared_error(y_train, oof))))
    cv_nrmse_list.append(nrmse(y_train, oof))
    cv_spearman_list.append(spearman_r(y_train, oof))

    fitted_tmp = clone(pipe)
    fitted_tmp.fit(X_train, y_train)
    y_pred_test = fitted_tmp.predict(X_test)

    test_r2_list.append(float(r2_score(y_test, y_pred_test)))
    test_mae_list.append(float(mean_absolute_error(y_test, y_pred_test)))
    test_rmse_list.append(float(np.sqrt(mean_squared_error(y_test, y_pred_test))))
    test_nrmse_list.append(nrmse(y_test, y_pred_test))
    test_spearman_list.append(spearman_r(y_test, y_pred_test))

oof_ensemble = np.zeros(len(y_train), dtype=float)
for tr_idx, va_idx in kfold.split(X_train):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr = y_train.iloc[tr_idx]
    fold_pred = np.zeros(len(va_idx), dtype=float)
    for _, pipe, w in model_specs:
        m = clone(pipe)
        m.fit(X_tr, y_tr)
        fold_pred += m.predict(X_val) * float(w)
    oof_ensemble[va_idx] = fold_pred / wsum

ensemble_cv_r2 = float(r2_score(y_train, oof_ensemble))
ensemble_cv_mae = float(mean_absolute_error(y_train, oof_ensemble))
ensemble_cv_rmse = float(np.sqrt(mean_squared_error(y_train, oof_ensemble)))
ensemble_cv_nrmse = nrmse(y_train, oof_ensemble)
ensemble_cv_spearman = spearman_r(y_train, oof_ensemble)

fitted_pipelines = []
ensemble_test_pred = np.zeros(len(y_test), dtype=float)
pipeline_files_by_model_name = {}

for i, (name, pipe, w) in enumerate(model_specs):
    fitted = clone(pipe)
    fitted.fit(X_train, y_train)
    fitted_pipelines.append(fitted)

    fname = f"pipeline_{i}_{name}.pkl"
    pipeline_files_by_model_name[name] = fname
    with open(os.path.join(output_dir, fname), "wb") as f:
        pickle.dump(fitted, f)

    ensemble_test_pred += fitted.predict(X_test) * float(w)

ensemble_test_pred = ensemble_test_pred / wsum

ensemble_test_r2 = float(r2_score(y_test, ensemble_test_pred))
ensemble_test_mae = float(mean_absolute_error(y_test, ensemble_test_pred))
ensemble_test_rmse = float(np.sqrt(mean_squared_error(y_test, ensemble_test_pred)))
ensemble_test_nrmse = nrmse(y_test, ensemble_test_pred)
ensemble_test_spearman = spearman_r(y_test, ensemble_test_pred)

class WeightedEnsembleRegressor:
    def __init__(self, pipelines, weights):
        self.pipelines = pipelines
        self.weights = np.array(weights, dtype=float)

    def predict(self, X):
        if isinstance(X, pd.Series):
            X_in = X.to_frame().T
        elif isinstance(X, dict):
            X_in = pd.DataFrame([X])
        else:
            X_in = X

        wsum_local = float(self.weights.sum())
        preds = None
        for p, w in zip(self.pipelines, self.weights):
            p_pred = np.asarray(p.predict(X_in), dtype=float)
            if preds is None:
                preds = np.zeros_like(p_pred, dtype=float)
            preds += p_pred * float(w)
        return preds / wsum_local

ensemble_model = WeightedEnsembleRegressor(fitted_pipelines, weights)
with open(os.path.join(output_dir, "voting_ensemble_regressor.pkl"), "wb") as f:
    pickle.dump(ensemble_model, f)

weights_by_model_name = {n: float(w) for n, w in zip(model_names, weights)}
models_table = [{"model_name": n, "weight": float(weights_by_model_name[n]), "pipeline_file": pipeline_files_by_model_name[n]} for n in model_names]

ensemble_metadata = {
    "model_names": model_names,
    "weights": weights,
    "weights_by_model_name": weights_by_model_name,
    "pipeline_files_by_model_name": pipeline_files_by_model_name,
    "models": models_table,
    "cv_r2_scores": cv_r2_list,
    "cv_mae": cv_mae_list,
    "cv_rmse": cv_rmse_list,
    "cv_nrmse": cv_nrmse_list,
    "cv_spearman": cv_spearman_list,
    "test_r2_scores": test_r2_list,
    "test_mae": test_mae_list,
    "test_rmse": test_rmse_list,
    "test_nrmse": test_nrmse_list,
    "test_spearman": test_spearman_list,
    "ensemble_cv_r2": ensemble_cv_r2,
    "ensemble_cv_mae": ensemble_cv_mae,
    "ensemble_cv_rmse": ensemble_cv_rmse,
    "ensemble_cv_nrmse": ensemble_cv_nrmse,
    "ensemble_cv_spearman": ensemble_cv_spearman,
    "ensemble_test_r2": ensemble_test_r2,
    "ensemble_test_mae": ensemble_test_mae,
    "ensemble_test_rmse": ensemble_test_rmse,
    "ensemble_test_nrmse": ensemble_test_nrmse,
    "ensemble_test_spearman": ensemble_test_spearman,
    "feature_names": X.columns.tolist(),
    "train_shape": X_train.shape,
    "data_file": "sweeteners_with_molecular_descriptors_and_fingerprints.csv",
    "random_state": 42
}

with open(os.path.join(output_dir, "ensemble_metadata.json"), "w") as f:
    json.dump(ensemble_metadata, f, indent=2)

with open(os.path.join(output_dir, "regression_model_metrics.txt"), "w") as f:
    f.write("=== Individual Model Performance ===\n\n")
    for i, name in enumerate(model_names):
        f.write(f"{name}:\n")
        f.write(f"  CV R2: {cv_r2_list[i]:.6f}\n")
        f.write(f"  CV MAE: {cv_mae_list[i]:.6f}\n")
        f.write(f"  CV RMSE: {cv_rmse_list[i]:.6f}\n")
        f.write(f"  CV NRMSE: {cv_nrmse_list[i]:.6f}\n")
        f.write(f"  CV Spearman: {cv_spearman_list[i]:.6f}\n")
        f.write(f"  Test R2: {test_r2_list[i]:.6f}\n")
        f.write(f"  Test MAE: {test_mae_list[i]:.6f}\n")
        f.write(f"  Test RMSE: {test_rmse_list[i]:.6f}\n")
        f.write(f"  Test NRMSE: {test_nrmse_list[i]:.6f}\n")
        f.write(f"  Test Spearman: {test_spearman_list[i]:.6f}\n\n")

    f.write("=== Voting Ensemble Performance ===\n\n")
    f.write(f"CV R2: {ensemble_cv_r2:.6f}\n")
    f.write(f"CV MAE: {ensemble_cv_mae:.6f}\n")
    f.write(f"CV RMSE: {ensemble_cv_rmse:.6f}\n")
    f.write(f"CV NRMSE: {ensemble_cv_nrmse:.6f}\n")
    f.write(f"CV Spearman: {ensemble_cv_spearman:.6f}\n")
    f.write(f"Test R2: {ensemble_test_r2:.6f}\n")
    f.write(f"Test MAE: {ensemble_test_mae:.6f}\n")
    f.write(f"Test RMSE: {ensemble_test_rmse:.6f}\n")
    f.write(f"Test NRMSE: {ensemble_test_nrmse:.6f}\n")
    f.write(f"Test Spearman: {ensemble_test_spearman:.6f}\n")

df_xgb = pd.read_csv("filtered_sweeteners_data_307_features.csv")
X_xgb = df_xgb.iloc[:, 1:-1]
y_xgb = df_xgb["Sweetness"]

X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
    X_xgb, y_xgb, test_size=0.3, random_state=5
)

mordred_cols = [col for col in X_xgb.columns if "Bit" not in col]
fingerprint_cols = [col for col in X_xgb.columns if "Bit" in col]

scaler_xgb_regression = MinMaxScaler()
scaler_xgb_regression.fit(X_train_xgb[mordred_cols])

with open(os.path.join(output_dir, "scaler_xgboost_307features.pkl"), "wb") as f:
    pickle.dump(scaler_xgb_regression, f)

xgb_reg_metadata = {
    "scaler_type": "MinMaxScaler",
    "feature_names": X_xgb.columns.tolist(),
    "mordred_columns": mordred_cols,
    "fingerprint_columns": fingerprint_cols,
    "n_features": len(X_xgb.columns),
    "train_shape": X_train_xgb.shape,
    "data_file": "filtered_sweeteners_data_307_features.csv",
    "random_state": 5
}

with open(os.path.join(output_dir, "xgboost_metadata.json"), "w") as f:
    json.dump(xgb_reg_metadata, f, indent=2)

print("Ensemble regression models saved")
print(f"Ensemble CV R2: {ensemble_cv_r2:.6f}")
print(f"Ensemble CV NRMSE: {ensemble_cv_nrmse:.6f}")
print(f"Ensemble Test R2: {ensemble_test_r2:.6f}")
print(f"Ensemble Test NRMSE: {ensemble_test_nrmse:.6f}")
print("Voting ensemble regressor saved")
print("XGBoost regression scaler saved")
# %%
# %%
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, log_loss, matthews_corrcoef
)
from sklearn.base import clone
import joblib
import os
import json

output_dir = "Voting_Ensemble_Classification_Models"
os.makedirs(output_dir, exist_ok=True)

train_df = pd.read_csv("sweet_nonsweet_train_set.csv")
test_df = pd.read_csv("sweet_nonsweet_test_set.csv")

X_train = train_df.drop(columns=["Name", "Class"])
y_train = train_df["Class"]
X_test = test_df.drop(columns=["Name", "Class"])
y_test = test_df["Class"]

CLASSIFICATION_THRESHOLD = 0.6


models = [
    ('LogisticRegression_1', Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', LogisticRegression(
            C=7.906,
            penalty='l1',
            solver='saga',
            random_state=42
        ))
    ]), 0.125),
    
    ('GradientBoosting', Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier(
            criterion='friedman_mse',
            learning_rate=0.0215,
            max_depth=7,
            max_features=0.6,
            min_samples_leaf=0.01,
            min_samples_split=0.291,
            n_estimators=400,
            subsample=0.953,
            random_state=42
        ))
    ]), 0.125),
    
    ('LogisticRegression_2', Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', LogisticRegression(
            C=719.686,
            penalty='l1',
            solver='saga',
            random_state=42
        ))
    ]), 0.375),
    
    ('LogisticRegression_3', Pipeline([
        ('scaler', MinMaxScaler()),
        ('model', LogisticRegression(
            C=2.560,
            penalty='l1',
            solver='saga',
            random_state=42
        ))
    ]), 0.125),
    
    ('RandomForest', Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(
            random_state=42
        ))
    ]), 0.125),
    
    ('ExtraTrees', Pipeline([
        ('scaler', RobustScaler()),
        ('model', ExtraTreesClassifier(
            bootstrap=False,
            class_weight='balanced',
            criterion='gini',
            max_features=0.8,
            min_samples_leaf=0.01,
            min_samples_split=0.338,
            n_estimators=25,
            random_state=42
        ))
    ]), 0.125)
]


voting_classifier = VotingClassifier(
    estimators=[(name, model) for name, model, _ in models],
    voting="soft",
    weights=[weight for _, _, weight in models]
)

def calculate_metrics(model, X_train, y_train, X_test, y_test, threshold=0.6):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_cv_proba = np.zeros((len(y_train), 2), dtype=float)

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]

        model_clone = clone(model)
        model_clone.fit(X_fold_train, y_fold_train)
        y_cv_proba[val_idx, :] = model_clone.predict_proba(X_fold_val)

    y_cv_pred = y_cv_proba[:, 1]
    y_cv_pred_class = (y_cv_pred >= threshold).astype(int)

    cv_metrics = {
        "cv_auc": float(roc_auc_score(y_train, y_cv_pred)),
        "cv_balanced_accuracy": float(balanced_accuracy_score(y_train, y_cv_pred_class)),
        "cv_weighted_f1": float(f1_score(y_train, y_cv_pred_class, average="weighted")),
        "cv_weighted_precision": float(precision_score(y_train, y_cv_pred_class, average="weighted", zero_division=0)),
        "cv_weighted_recall": float(recall_score(y_train, y_cv_pred_class, average="weighted")),
        "cv_log_loss": float(log_loss(y_train, y_cv_proba, labels=[0, 1])),
        "cv_mcc": float(matthews_corrcoef(y_train, y_cv_pred_class)),
    }

    fitted = clone(model)
    fitted.fit(X_train, y_train)
    y_test_proba = fitted.predict_proba(X_test)
    y_test_pred = y_test_proba[:, 1]
    y_test_pred_class = (y_test_pred >= threshold).astype(int)

    test_metrics = {
        "test_auc": float(roc_auc_score(y_test, y_test_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_test_pred_class)),
        "test_weighted_f1": float(f1_score(y_test, y_test_pred_class, average="weighted")),
        "test_weighted_precision": float(precision_score(y_test, y_test_pred_class, average="weighted", zero_division=0)),
        "test_weighted_recall": float(recall_score(y_test, y_test_pred_class, average="weighted")),
        "test_log_loss": float(log_loss(y_test, y_test_proba, labels=[0, 1])),
        "test_mcc": float(matthews_corrcoef(y_test, y_test_pred_class)),
    }

    return {**cv_metrics, **test_metrics}

results = []
for name, model, _ in models:
    metrics = calculate_metrics(model, X_train, y_train, X_test, y_test, CLASSIFICATION_THRESHOLD)
    metrics["model"] = name
    results.append(metrics)

metrics = calculate_metrics(voting_classifier, X_train, y_train, X_test, y_test, CLASSIFICATION_THRESHOLD)
metrics["model"] = "VotingEnsemble"
results.append(metrics)

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, "voting_ensemble_classification_model_metrics.csv"), index=False)

pipeline_files_by_model_name = {}
for i, (name, pipeline, _) in enumerate(models):
    fitted_pipeline = clone(pipeline)
    fitted_pipeline.fit(X_train, y_train)
    fname = f"pipeline_{i}_{name}.joblib"
    joblib.dump(fitted_pipeline, os.path.join(output_dir, fname))
    pipeline_files_by_model_name[name] = fname

voting_classifier.fit(X_train, y_train)
joblib.dump(voting_classifier, os.path.join(output_dir, "voting_ensemble_classifier.joblib"))

model_names = [name for name, _, _ in models]
weights = [float(weight) for _, _, weight in models]
weights_by_model_name = {name: float(weight) for name, _, weight in models}
models_table = [
    {"model_name": name, "weight": float(weights_by_model_name[name]), "pipeline_file": pipeline_files_by_model_name[name]}
    for name in model_names
]

metadata = {
    "model_names": model_names,
    "weights": weights,
    "weights_by_model_name": weights_by_model_name,
    "pipeline_files_by_model_name": pipeline_files_by_model_name,
    "models": models_table,
    "feature_names": list(X_train.columns),
    "n_features": int(len(X_train.columns)),
    "voting_type": "soft",
    "threshold": float(CLASSIFICATION_THRESHOLD),
    "data_files": {
        "train": "sweet_nonsweet_train_set.csv",
        "test": "sweet_nonsweet_test_set.csv"
    },
    "random_state": 42
}

with open(os.path.join(output_dir, "classification_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("Classification models saved")

