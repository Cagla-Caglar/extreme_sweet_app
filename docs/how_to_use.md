# How to Use the Application

This application provides a comprehensive cheminformatics platform for exploring the  relationship between molecular structures and perceived 
sweetness intensity. It offers both predictive models and interactive visualization tools designed for researchers,  practitioners, and those 
with limited experience in machine learning or cheminformatics.

## Application Structure

The sidebar contains all sections of the application. Key components are summarized below.

### 1. Sweetness Prediction System

This module enables prediction of both the sweet/non-sweet classification and the quantitative sweetness intensity (in log-scale and real-scale) 
based on the user-provided SMILES string. Results are computed using pre-trained ensemble and XGBoost regression models. The system provides model 
explanations through LIME analysis to enhance interpretability.

### 2. Model Interpretability

Several interactive tools assist users in understanding how molecular features impact the predictions.

**Feature Importance (Classification & Regression)**  

Interactive bar plots display the relative contribution of molecular descriptors and fingerprints to the model’s output. Higher bars indicate features 
with greater influence. Users may adjust the number of displayed features via sliders. These plots should be interpreted as ranking variables in terms 
of their global importance for the trained model, not necessarily causal relationships.

**SHAP Dependence Plots (Regression)**  

These plots visualize how variations in individual feature values influence SHAP values, which represent the local contribution to the prediction. 
The x-axis corresponds to the scaled feature values, and the y-axis indicates SHAP values. Positive SHAP values suggest an increase in predicted sweetness, 
while negative values suggest a decrease. The color gradient reflects the feature value, aiding the interpretation of nonlinear  patterns.

**LIME Explanation (Regression)**  

LIME (Local Interpretable Model-agnostic Explanations) provides local explanations for specific predictions. Feature contributions are shown via horizontal 
bar plots, where bars to the right (positive weights) increase the predicted sweetness, and bars to the left (negative weights) decrease it. Dropdown menus 
allow the user to select specific compounds and inspect the relationship between molecular features and predictions.

### 3. Interactive DFT-Based Spectral Databases

Infrared and Raman spectral datasets are provided for visualization and comparison. Users can select compounds of interest and overlay their spectra for comparative analysis.

**IR and Raman Spectra**  

Normalized spectra are visualized interactively. Multiple compounds can be selected simultaneously to explore trends or differences.

**Raman Activity Viewer**  

This section presents Raman activity predictions for sweetener compounds. The visual style mirrors that of the spectra plots.

### 4. Optimized 3D Structures Viewer

A 3D molecular visualization tool allows users to inspect the optimized geometries of sweeteners. Structures can be interactively rotated, zoomed, and 
downloaded as MOL files.

### 5. Molecular Descriptor Dictionary

This section provides an initial dictionary of molecular descriptors that were found to be most influential in the classification and regression analyses 
performed to distinguish sweet versus non-sweet compounds and to explain the variations in sweetness intensity, respectively. Each descriptor is accompanied 
by a scientifically curated explanation describing its potential relevance based on the feature importance and SHAP analyses conducted in this study. This dictionary currently focuses on the key features identified in this research but will be expanded in future updates to provide comprehensive explanations for 
a wider range of molecular descriptors commonly used in cheminformatics.

## Data Export Functionality

All interactive visualization modules offer options to download the underlying data in CSV or MOL formats. This facilitates further analysis, integration into reports, or replication of findings.

## Important Note on Interpretability

While this application provides advanced tools for understanding model behavior, it is essential to remember that feature importance methods (SHAP, LIME, Feature Importance) reflect correlations within the dataset and model assumptions. They do not establish  
causal relationships between molecular features and sweetness perception.

For any scientific conclusions, these analyses should be complemented with domain expertise and experimental validation.
