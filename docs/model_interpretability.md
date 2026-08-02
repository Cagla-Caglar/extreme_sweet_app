# Model Interpretability and How to Read the Graphs

Understanding the behavior of machine learning models is essential to ensure the reliability and transparency of predictive systems in cheminformatics. This application integrates multiple interpretability methods to clarify how molecular features influence the predicted sweetness outcomes.

## 1. Feature Importance (Global Interpretability)

Feature importance analysis ranks molecular descriptors according to their overall influence on the model output.

The x-axis shows molecular descriptors, the y-axis shows normalized importance values. Taller bars indicate greater importance for model decisions across the dataset. These charts represent statistical importance, not causal relationships. They help identify descriptors that are most relevant to the model's predictive accuracy.

## 2. SHAP Dependence Plots (Local and Global Interpretability)

SHAP (SHapley Additive exPlanations) quantifies the contribution of each feature to individual predictions based on Shapley values from cooperative game theory.

The x-axis corresponds to scaled feature values, and the y-axis indicates SHAP values representing the feature's contribution to the model output. The color gradient reflects feature value intensity, aiding in detecting interactions. Positive SHAP values indicate that the feature increases predicted sweetness intensity, while negative values indicate a decrease. These plots allow researchers to observe nonlinear behaviors and interaction effects.

## 3. LIME Explanation (Local Interpretability)

LIME (Local Interpretable Model-Agnostic Explanations) approximates the model locally with a simpler surrogate to explain individual predictions.

Feature contribution weights are shown on the x-axis, with feature names ordered by impact on the y-axis. Bars to the right (positive weights) increase predicted sweetness, and bars to the left (negative weights) decrease it. LIME explanations provide insights at the individual molecule level, showing how specific features influence the prediction locally.

## Why Explainability Is Integrated

Explainability in cheminformatics is vital to ensure that model predictions align with chemical reasoning and domain expertise. These methods assist users in verifying whether the model's behavior is consistent with expected molecular properties and guide potential experimental validation.
