import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Avalon import pyAvalonTools
from mordred import Calculator, descriptors

from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer


try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

APP_DIR = BASE_DIR.parent
DATA_DIR = Path(os.getenv("SWEET_DATA_DIR", str(APP_DIR / "data")))
MODELS_DIR = APP_DIR / "models"

REG_DIR = MODELS_DIR / "Voting_Ensemble_and_XGBoost_Regression_Models"
CLF_DIR = MODELS_DIR / "Voting_Ensemble_Classification_Models"


def validate_smiles(smiles: str) -> bool:
    if not smiles or not isinstance(smiles, str):
        return False
    smiles = smiles.strip()
    if not smiles:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        if mol.GetNumAtoms() == 0:
            return False
        return True
    except Exception:
        return False


def handle_large_values(x):
    try:
        if isinstance(x, (int, float, np.integer, np.floating)):
            xv = float(x)
            if np.isinf(xv) or abs(xv) > 1e10:
                return np.nan
            return xv
        return x
    except (OverflowError, TypeError, ValueError):
        return np.nan


# ── applicability domain ──────────────────────────────────────────────────────

AD_K = 5  # number of nearest neighbours for AD assessment


def _tanimoto_row_vs_matrix(row, matrix):
    """Tanimoto similarity between a single fingerprint vector and a matrix of fingerprints."""
    ab = matrix.astype(np.float32) @ row.astype(np.float32)
    sa = row.sum()
    sb = matrix.sum(axis=1).astype(np.float32)
    denom = sa + sb - ab
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom == 0, 0.0, ab / denom)


def _compute_ad_threshold(fp_matrix, k=AD_K):
    """10th percentile of training-set mean k-NN Tanimoto similarity distribution."""
    n = fp_matrix.shape[0]
    sim = fp_matrix.astype(np.float32) @ fp_matrix.T.astype(np.float32)
    s_row = fp_matrix.sum(axis=1, keepdims=True).astype(np.float32)
    denom = s_row + s_row.T - sim
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(denom == 0, 0.0, sim / denom)
    np.fill_diagonal(sim, -1.0)
    knn = np.sort(sim, axis=1)[:, -k:]
    mean_knn = knn.mean(axis=1)
    return float(np.percentile(mean_knn, 10))


def assess_applicability_domain(query_fp, train_fp_matrix, threshold, k=AD_K):
    """Check whether a query compound falls within the applicability domain.
    Returns (within_ad, mean_knn_similarity, threshold)."""
    sims = _tanimoto_row_vs_matrix(query_fp, train_fp_matrix)
    top_k = np.sort(sims)[-k:]
    mean_sim = float(top_k.mean())
    within_ad = mean_sim >= threshold
    return within_ad, mean_sim, threshold


# cached AD artifacts
_CLF_AD_FP = None
_CLF_AD_THR = None

_REG_ENS_AD_FP = None
_REG_ENS_AD_THR = None

_REG_XGB_AD_FP = None
_REG_XGB_AD_THR = None


_CALC_2D = Calculator(descriptors, ignore_3D=True)


def calculate_molecular_descriptors(smiles: str) -> dict:
    if not validate_smiles(smiles):
        raise ValueError("Invalid SMILES string provided")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    try:
        result = _CALC_2D(mol)
        desc_dict = result.fill_missing().asdict()
        return {k: handle_large_values(v) for k, v in desc_dict.items()}
    except Exception:
        return {}


def _bitvect_to_int_list(fp, nBits: int | None = None) -> list:
    if fp is None:
        return [0] * int(nBits) if nBits is not None else []
    try:
        s = fp.ToBitString()
        return [1 if ch == "1" else 0 for ch in s]
    except Exception:
        if nBits is None:
            raise
        return [int(fp[i]) for i in range(int(nBits))]


def calculate_morgan_fingerprint(smiles: str, radius: int = 2, nBits: int = 2048) -> list:
    if not validate_smiles(smiles):
        raise ValueError("Invalid SMILES string")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0] * int(nBits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, int(radius), nBits=int(nBits))
    return _bitvect_to_int_list(fp, nBits=nBits)


def calculate_maccs_keys(smiles: str) -> list:
    if not validate_smiles(smiles):
        raise ValueError("Invalid SMILES string")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0] * 167
    fp = MACCSkeys.GenMACCSKeys(mol)
    return _bitvect_to_int_list(fp, nBits=167)


def calculate_atom_pair_fingerprint(smiles: str, nBits: int = 2048) -> list:
    if not validate_smiles(smiles):
        raise ValueError("Invalid SMILES string")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0] * int(nBits)
    fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=int(nBits))
    return _bitvect_to_int_list(fp, nBits=nBits)


def calculate_topological_torsion_hashed(smiles: str, nBits: int = 2048) -> list:
    if not validate_smiles(smiles):
        raise ValueError("Invalid SMILES string")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0] * int(nBits)
    fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=int(nBits))
    return _bitvect_to_int_list(fp, nBits=nBits)


def calculate_topological_torsion_sparse(smiles: str, max_features: int = 82) -> list:
    if not validate_smiles(smiles):
        raise ValueError("Invalid SMILES string")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0] * int(max_features)
    fp = Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol)
    keys = list(fp.GetNonzeroElements().keys())
    result = list(keys) + [0] * (int(max_features) - len(keys))
    return result[:int(max_features)]


def calculate_avalon_fingerprint(smiles: str, nBits: int = 2048) -> list:
    if not validate_smiles(smiles):
        raise ValueError("Invalid SMILES string")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0] * int(nBits)
    try:
        fp = pyAvalonTools.GetAvalonFP(mol, nBits=int(nBits))
    except TypeError:
        fp = pyAvalonTools.GetAvalonFP(mol, int(nBits))
    return _bitvect_to_int_list(fp, nBits=nBits)


def process_smiles_for_regression(smiles: str) -> dict:
    if not validate_smiles(smiles):
        raise ValueError("Invalid SMILES string provided")

    result = {}
    result.update(calculate_molecular_descriptors(smiles))

    try:
        morgan_fp = calculate_morgan_fingerprint(smiles, radius=2, nBits=2048)
    except Exception as e:
        raise ValueError(f"Morgan fingerprint failed: {e}")

    try:
        maccs_fp = calculate_maccs_keys(smiles)
    except Exception as e:
        raise ValueError(f"MACCS fingerprint failed: {e}")

    try:
        atom_pair_fp = calculate_atom_pair_fingerprint(smiles, nBits=2048)
    except Exception as e:
        raise ValueError(f"AtomPair fingerprint failed: {e}")

    try:
        topo_torsion_fp = calculate_topological_torsion_hashed(smiles, nBits=2048)
    except Exception as e:
        raise ValueError(f"TopoTorsion fingerprint failed: {e}")

    try:
        avalon_fp = calculate_avalon_fingerprint(smiles, nBits=2048)
    except Exception as e:
        raise ValueError(f"Avalon fingerprint failed: {e}")

    result.update({f"ECFP4_Bit_{i}": val for i, val in enumerate(morgan_fp)})
    result.update({f"MACCS_Bit_{i}": val for i, val in enumerate(maccs_fp)})
    result.update({f"AtomPair_Bit_{i}": val for i, val in enumerate(atom_pair_fp)})
    result.update({f"TopoTorsion_Bit_{i}": val for i, val in enumerate(topo_torsion_fp)})
    result.update({f"Avalon_Bit_{i}": val for i, val in enumerate(avalon_fp)})

    return result


def process_smiles_for_classification(smiles: str) -> dict:
    if not validate_smiles(smiles):
        raise ValueError("Invalid SMILES string provided")

    result = {}
    result.update(calculate_molecular_descriptors(smiles))

    try:
        morgan_fp = calculate_morgan_fingerprint(smiles, radius=2, nBits=2048)
    except Exception as e:
        raise ValueError(f"Morgan fingerprint failed: {e}")

    try:
        maccs_fp = calculate_maccs_keys(smiles)
    except Exception as e:
        raise ValueError(f"MACCS fingerprint failed: {e}")

    try:
        atom_pair_fp = calculate_atom_pair_fingerprint(smiles, nBits=2048)
    except Exception as e:
        raise ValueError(f"AtomPair fingerprint failed: {e}")

    try:
        topo_torsion_fp = calculate_topological_torsion_sparse(smiles, max_features=82)
    except Exception as e:
        raise ValueError(f"TopoTorsion fingerprint failed: {e}")

    try:
        avalon_fp = calculate_avalon_fingerprint(smiles, nBits=2048)
    except Exception as e:
        raise ValueError(f"Avalon fingerprint failed: {e}")

    result.update({f"ECFP4_Bit_{i}": val for i, val in enumerate(morgan_fp)})
    result.update({f"MACCS_Bit_{i}": val for i, val in enumerate(maccs_fp)})
    result.update({f"AtomPair_Bit_{i}": val for i, val in enumerate(atom_pair_fp)})
    result.update({f"TopoTorsion_Bit_{i}": val for i, val in enumerate(topo_torsion_fp)})
    result.update({f"Avalon_Bit_{i}": val for i, val in enumerate(avalon_fp)})

    return result


def get_raw_features_regression(smiles: str) -> pd.DataFrame:
    feats = process_smiles_for_regression(smiles)
    return pd.DataFrame([feats])


def get_raw_features_classification(smiles: str) -> pd.DataFrame:
    feats = process_smiles_for_classification(smiles)
    return pd.DataFrame([feats])


def _build_one_row(feature_cols, raw_df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in raw_df.columns]
    if missing:
        for c in missing:
            raw_df[c] = np.nan
    row = {c: np.nan for c in feature_cols}
    raw_row = raw_df.iloc[0].to_dict()
    for c in feature_cols:
        if c in raw_row:
            v = raw_row[c]
            if isinstance(v, (bool, np.bool_)):
                row[c] = float(v)
            else:
                row[c] = v
    return pd.DataFrame([row], columns=feature_cols)


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_artifact(path: Path):
    return joblib.load(Path(path))


def _load_joblib(path: Path):
    return _load_artifact(path)


def _resolve_data_path(filename: str) -> Path:
    candidates = [
        DATA_DIR / filename,
        APP_DIR / filename,
        BASE_DIR / filename,
        APP_DIR / "data" / filename,
        BASE_DIR / "data" / filename,
    ]
    for p in candidates:
        if p.exists() and p.is_file() and p.stat().st_size > 0:
            return p
    raise FileNotFoundError(
        f"Data file not found or empty: {filename}. Checked: " + ", ".join(str(c) for c in candidates)
    )


_CLF_TRAIN_DF = None
_CLF_FEATURE_COLS = None
_CLF_MEDIANS = None

_REG_ENSEMBLE_DF = None
_REG_ENSEMBLE_COLS = None
_REG_ENSEMBLE_MEDIANS = None

_XGB_307_DF = None
_XGB_307_COLS = None
_XGB_307_MEDIANS = None


def _load_clf_training_artifacts():
    global _CLF_TRAIN_DF, _CLF_FEATURE_COLS, _CLF_MEDIANS
    global _CLF_AD_FP, _CLF_AD_THR
    if _CLF_TRAIN_DF is not None:
        return
    path = _resolve_data_path("sweet_nonsweet_train_set.csv")
    df = pd.read_csv(path)
    _CLF_TRAIN_DF = df
    _CLF_FEATURE_COLS = [c for c in df.columns if c not in ["Name", "Class"]]
    _CLF_MEDIANS = df[_CLF_FEATURE_COLS].median(numeric_only=True)
    ecfp_cols = [c for c in df.columns if c.startswith("ECFP4_Bit_")]
    _CLF_AD_FP = df[ecfp_cols].values.astype(np.uint8)
    _CLF_AD_THR = _compute_ad_threshold(_CLF_AD_FP)


def _load_reg_ensemble_training_artifacts():
    global _REG_ENSEMBLE_DF, _REG_ENSEMBLE_COLS, _REG_ENSEMBLE_MEDIANS
    global _REG_ENS_AD_FP, _REG_ENS_AD_THR
    if _REG_ENSEMBLE_DF is not None:
        return
    path = _resolve_data_path("sweeteners_with_molecular_descriptors_and_fingerprints.csv")
    df = pd.read_csv(path)
    _REG_ENSEMBLE_DF = df
    _REG_ENSEMBLE_COLS = [c for c in df.columns if c not in ["Molecule", "Sweetness"]]
    X_all = df[_REG_ENSEMBLE_COLS]
    y_all = df["Sweetness"]
    X_tr, _, _, _ = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
    _REG_ENSEMBLE_MEDIANS = X_tr.median(numeric_only=True)
    ecfp_cols = [c for c in X_tr.columns if c.startswith("ECFP4_Bit_")]
    _REG_ENS_AD_FP = X_tr[ecfp_cols].values.astype(np.uint8)
    _REG_ENS_AD_THR = _compute_ad_threshold(_REG_ENS_AD_FP)


def _load_xgb307_training_artifacts():
    global _XGB_307_DF, _XGB_307_COLS, _XGB_307_MEDIANS
    global _REG_XGB_AD_FP, _REG_XGB_AD_THR
    if _XGB_307_DF is not None:
        return
    path = _resolve_data_path("filtered_sweeteners_data_307_features.csv")
    df = pd.read_csv(path)
    _XGB_307_DF = df
    _XGB_307_COLS = df.columns[1:-1].tolist()
    X_all = df.iloc[:, 1:-1]
    X_tr, _ = train_test_split(X_all, train_size=0.7, random_state=5)
    _XGB_307_MEDIANS = X_tr.median(numeric_only=True)
    # AD uses full ECFP4 from the regression dataset (same 316 compounds, different split)
    reg_path = _resolve_data_path("sweeteners_with_molecular_descriptors_and_fingerprints.csv")
    reg_df = pd.read_csv(reg_path)
    ecfp_cols = [c for c in reg_df.columns if c.startswith("ECFP4_Bit_")]
    idx_all = np.arange(len(reg_df))
    idx_tr, _ = train_test_split(idx_all, test_size=0.3, random_state=5)
    _REG_XGB_AD_FP = reg_df.iloc[idx_tr][ecfp_cols].values.astype(np.uint8)
    _REG_XGB_AD_THR = _compute_ad_threshold(_REG_XGB_AD_FP)


def prepare_classification_features(smiles: str) -> pd.DataFrame:
    if not validate_smiles(smiles):
        raise ValueError("Invalid SMILES string")

    _load_clf_training_artifacts()

    raw_df = get_raw_features_classification(smiles)
    x = _build_one_row(_CLF_FEATURE_COLS, raw_df)
    for c in _CLF_FEATURE_COLS:
        if pd.isna(x.loc[0, c]):
            x.loc[0, c] = _CLF_MEDIANS.get(c, np.nan)
    return x[_CLF_FEATURE_COLS]


def prepare_regression_features_ensemble(smiles: str) -> pd.DataFrame:
    if not validate_smiles(smiles):
        raise ValueError("Invalid SMILES string")

    _load_reg_ensemble_training_artifacts()

    raw_df = get_raw_features_regression(smiles)
    x = _build_one_row(_REG_ENSEMBLE_COLS, raw_df)
    for c in _REG_ENSEMBLE_COLS:
        if pd.isna(x.loc[0, c]):
            x.loc[0, c] = _REG_ENSEMBLE_MEDIANS.get(c, np.nan)
    return x[_REG_ENSEMBLE_COLS]


def prepare_regression_features_xgboost(smiles: str) -> pd.DataFrame:
    if not validate_smiles(smiles):
        raise ValueError("Invalid SMILES string")

    _load_xgb307_training_artifacts()

    raw_df = get_raw_features_regression(smiles)
    x = _build_one_row(_XGB_307_COLS, raw_df)
    for c in _XGB_307_COLS:
        if pd.isna(x.loc[0, c]):
            x.loc[0, c] = _XGB_307_MEDIANS.get(c, np.nan)
    return x[_XGB_307_COLS]


def classify_sweet_probability(clf_features_df: pd.DataFrame) -> float:
    meta = _load_json(CLF_DIR / "classification_metadata.json")
    models_table = meta.get("models", None)
    if not models_table:
        model_names = meta["model_names"]
        weights = meta["weights"]
        pipe_map = meta.get("pipeline_files_by_model_name", {})
        models_table = []
        for n, w in zip(model_names, weights):
            models_table.append({"model_name": n, "weight": float(w), "pipeline_file": pipe_map.get(n)})

    proba_sum = None
    wsum = 0.0

    for item in models_table:
        w = float(item["weight"])
        fname = item.get("pipeline_file", None)
        if not fname:
            raise ValueError("Missing pipeline_file in classification metadata")
        pipe = _load_joblib(CLF_DIR / fname)
        p = np.asarray(pipe.predict_proba(clf_features_df), dtype=float)
        if proba_sum is None:
            proba_sum = np.zeros_like(p, dtype=float)
        proba_sum += p * w
        wsum += w

    if proba_sum is None or wsum == 0.0:
        raise ValueError("Invalid classification ensemble state")

    p_ens = proba_sum / wsum
    return float(p_ens[0][1])


def predict_ensemble_sweetness_from_features(reg_features_df: pd.DataFrame) -> float:
    meta = _load_json(REG_DIR / "ensemble_metadata.json")
    models_table = meta.get("models", None)
    if not models_table:
        model_names = meta["model_names"]
        weights = meta["weights"]
        pipe_map = meta.get("pipeline_files_by_model_name", {})
        models_table = []
        for n, w in zip(model_names, weights):
            models_table.append({"model_name": n, "weight": float(w), "pipeline_file": pipe_map.get(n)})

    preds = None
    wsum = 0.0
    for item in models_table:
        w = float(item["weight"])
        fname = item.get("pipeline_file", None)
        if not fname:
            raise ValueError("Missing pipeline_file in regression metadata")
        pipe = _load_artifact(REG_DIR / fname)
        p = np.asarray(pipe.predict(reg_features_df), dtype=float)
        if preds is None:
            preds = np.zeros_like(p, dtype=float)
        preds += p * w
        wsum += w

    if preds is None or wsum == 0.0:
        raise ValueError("Invalid regression ensemble state")

    return float((preds / wsum)[0])


def predict_xgboost_sweetness_from_features(reg_features_df: pd.DataFrame) -> float:
    reg_model = _load_joblib(REG_DIR / "best_xgboost_regressor_model_with_filtered_features_MinMaxScaler.joblib")
    scaler = _load_artifact(REG_DIR / "scaler_xgboost_307features.pkl")
    mordred_cols = [c for c in reg_features_df.columns if "Bit" not in c]
    x_scaled = reg_features_df.copy()
    x_scaled[mordred_cols] = scaler.transform(reg_features_df[mordred_cols])
    pred = reg_model.predict(x_scaled)[0]
    return float(pred)


def get_lime_explanation_xgboost(smiles: str, num_features: int = 50):
    reg_features_df = prepare_regression_features_xgboost(smiles)
    reg_model = _load_joblib(REG_DIR / "best_xgboost_regressor_model_with_filtered_features_MinMaxScaler.joblib")

    path = _resolve_data_path("filtered_sweeteners_data_307_features.csv")
    reg_df = pd.read_csv(path)
    X = reg_df.iloc[:, 1:-1]
    y = reg_df.iloc[:, -1]
    X_tr, _, y_tr, _ = train_test_split(X, y, train_size=0.7, random_state=5)

    scaler = _load_artifact(REG_DIR / "scaler_xgboost_307features.pkl")

    bit_cols = [c for c in X_tr.columns if "Bit" in c]
    cat_idx = [X_tr.columns.get_loc(c) for c in bit_cols]

    mordred_cols = [c for c in X_tr.columns if "Bit" not in c]
    X_tr_scaled = X_tr.copy()
    X_tr_scaled[mordred_cols] = scaler.transform(X_tr[mordred_cols])
    X_tr_scaled = X_tr_scaled.to_numpy(dtype=float)
    x_scaled = reg_features_df.copy()
    x_scaled[mordred_cols] = scaler.transform(reg_features_df[mordred_cols])
    x_scaled = x_scaled.to_numpy(dtype=float)

    explainer = LimeTabularExplainer(
        training_data=X_tr_scaled,
        feature_names=X_tr.columns.tolist(),
        mode="regression",
        categorical_features=cat_idx,
        discretize_continuous=False,
        random_state=5,
    )

    exp = explainer.explain_instance(
        data_row=x_scaled[0],
        predict_fn=reg_model.predict,
        num_features=int(num_features),
    )

    return exp


def get_sweetness_prediction(smiles: str) -> tuple:
    if not validate_smiles(smiles):
        raise ValueError("Invalid SMILES string")

    raw_df = get_raw_features_regression(smiles)
    results_dir = APP_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(results_dir / "raw_molecular_features.csv", index=False)

    # ECFP4 fingerprint for AD assessment (shared across all models)
    query_ecfp = np.array(calculate_morgan_fingerprint(smiles, radius=2, nBits=2048), dtype=np.uint8)

    # Stage 1: classification
    clf_features_df = prepare_classification_features(smiles)
    clf_meta = _load_json(CLF_DIR / "classification_metadata.json")
    threshold = float(clf_meta.get("threshold", 0.6))

    _load_clf_training_artifacts()
    clf_in_ad, clf_sim, clf_thr = assess_applicability_domain(query_ecfp, _CLF_AD_FP, _CLF_AD_THR)

    p_sweet = classify_sweet_probability(clf_features_df)
    is_sweet = bool(p_sweet >= threshold)

    out = {
        "sweet_probability": None,
        "is_sweet": None,
        "ensemble_sweetness": None,
        "xgboost_sweetness": None,
        "ensemble_sweetness_original_scale": None,
        "xgboost_sweetness_original_scale": None,
        "xgboost_lime_explanation": None,
        "ad_classification": {
            "within_ad": bool(clf_in_ad),
            "mean_knn_similarity": float(clf_sim),
            "threshold": float(clf_thr),
        },
        "ad_regression_ensemble": None,
        "ad_regression_xgboost": None,
        "ad_warnings": [],
    }

    if not clf_in_ad:
        out["ad_warnings"].append(
            f"This compound falls outside the applicability domain of the classification model "
            f"(Tanimoto k-NN similarity = {clf_sim:.3f}, threshold = {clf_thr:.3f}). "
            f"Predictions are not reported to ensure reliability."
        )
        return out, clf_features_df

    out["sweet_probability"] = float(p_sweet)
    out["is_sweet"] = bool(is_sweet)

    if is_sweet:
        reg_feat_ens = prepare_regression_features_ensemble(smiles)
        reg_feat_xgb = prepare_regression_features_xgboost(smiles)

        # AD check for Voting Ensemble regression
        _load_reg_ensemble_training_artifacts()
        ens_in_ad, ens_sim, ens_thr = assess_applicability_domain(query_ecfp, _REG_ENS_AD_FP, _REG_ENS_AD_THR)
        out["ad_regression_ensemble"] = {
            "within_ad": bool(ens_in_ad),
            "mean_knn_similarity": float(ens_sim),
            "threshold": float(ens_thr),
        }

        # AD check for XGBoost regression
        _load_xgb307_training_artifacts()
        xgb_in_ad, xgb_sim, xgb_thr = assess_applicability_domain(query_ecfp, _REG_XGB_AD_FP, _REG_XGB_AD_THR)
        out["ad_regression_xgboost"] = {
            "within_ad": bool(xgb_in_ad),
            "mean_knn_similarity": float(xgb_sim),
            "threshold": float(xgb_thr),
        }

        if ens_in_ad:
            ens_pred = predict_ensemble_sweetness_from_features(reg_feat_ens)
            out["ensemble_sweetness"] = float(ens_pred)
            out["ensemble_sweetness_original_scale"] = float(10 ** ens_pred)
        else:
            out["ad_warnings"].append(
                f"This compound falls outside the applicability domain of the Voting Ensemble regression model "
                f"(Tanimoto k-NN similarity = {ens_sim:.3f}, threshold = {ens_thr:.3f}). "
                f"Sweetness intensity prediction is not provided for this model."
            )

        if xgb_in_ad:
            xgb_pred = predict_xgboost_sweetness_from_features(reg_feat_xgb)
            out["xgboost_sweetness"] = float(xgb_pred)
            out["xgboost_sweetness_original_scale"] = float(10 ** xgb_pred)
            out["xgboost_lime_explanation"] = get_lime_explanation_xgboost(smiles, num_features=50)
        else:
            out["ad_warnings"].append(
                f"This compound falls outside the applicability domain of the XGBoost regression model "
                f"(Tanimoto k-NN similarity = {xgb_sim:.3f}, threshold = {xgb_thr:.3f}). "
                f"Sweetness intensity prediction is not provided for this model."
            )

    return out, clf_features_df
