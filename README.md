# Extreme Sweet Database

The Extreme Sweet Database is an open-source web platform that implements a two-stage predictive workflow for sweetness: it first classifies whether a compound is sweet or non-sweet from a SMILES input and then predicts its relative sweetness intensity. Model interpretability analyses, including feature importance rankings, SHAP dependence plots, and LIME explanations, are presented as interactive visualizations within the platform.

The platform also serves as an open-access database providing DFT-optimized three-dimensional structures and vibrational spectra for sweetener compounds. A molecular descriptor dictionary containing mathematical formulations and definitions is included and will be expanded in future updates.

**Web Application:** [https://extreme-sweet.streamlit.app/](https://extreme-sweet.streamlit.app/)

## Citation

If you use this repository, please cite the archived version:

**DOI:** [10.5281/zenodo.18528956](https://doi.org/10.5281/zenodo.18528956)

The associated journal article is currently under peer review. Its reference will be added here upon publication.

## Repository Structure

### Application

| File | Description |
|---|---|
| `extreme_sweet_app.py` | Streamlit application and user interface |
| `backend/sweetness_predictor.py` | Feature calculation, model loading and prediction |
| `backend/train_and_save_models.py` | Training and serialization of the models |

### Analyses

| File | Description |
|---|---|
| `analyses/compute_descriptor_and_fingerprints.py` | Molecular descriptor and fingerprint calculation |
| `analyses/model_interpretability.py` | Feature importance, SHAP and LIME analyses |
| `analyses/xgboost_evaluation.py` | Hyperparameter optimization and evaluation |
| `analyses/y_randomization.py` | Y-randomization test |
| `analyses/tanimato_similarity_analysis/` | Applicability domain assessment |
| `analyses/polarizability_analysis.py` | Polarizability tensor analysis |
| `analyses/dft_and_experimental_spectral_data_comparison.py` | Comparison with experimental spectra |

### Data and Models

| Path | Description |
|---|---|
| `data/` | Descriptor and fingerprint matrices, training and test splits, experimental spectra |
| `models/Voting_Ensemble_and_XGBoost_Regression_Models/` | Regression models, scaler and metadata |
| `models/Voting_Ensemble_Classification_Models/` | Classification models and metadata |

### Visualizations

| File | Description |
|---|---|
| `templates/optimized_3d_structures_viewer.html` | DFT-optimized structures |
| `templates/Sweeteners_IR_Spectrum_Viewer.html` | Infrared spectra |
| `templates/Sweeteners_Raman_Spectrum_Viewer.html` | Raman spectra |
| `templates/Sweeteners_Raman_Activity_Viewer.html` | Raman activity analysis |
| `templates/Interactive_Feature_Importance_for_Regression.html` | Feature importance, regression |
| `templates/Interactive_Feature_Importance_for_Classification.html` | Feature importance, classification |
| `templates/interactive_shap_dependence_plots_for_regression.html` | SHAP dependence plots |
| `templates/lime_for_sweetness_intensity_prediction.html` | LIME explanations |
| `templates/Observed_vs_Predicted_Sweetness_With_Original_Scale.html` | Observed and predicted sweetness |
| `templates/Sweetener_Database.html` | Compound table |

### Documentation

| Path | Description |
|---|---|
| `docs/` | Documentation pages displayed within the application |

## Installation

```bash
pip install -r requirements.txt
streamlit run extreme_sweet_app.py
```

The package versions listed in `requirements.txt` should be used, as the serialized models were produced with those versions.

## License

This project is released under the [MIT License](LICENSE).
