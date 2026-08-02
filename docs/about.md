# An Interactive Platform Integrating Quantum Chemical Calculations and Machine Learning for Sweetness Prediction

The Extreme Sweet Database is an interactive web-based scientific platform designed to
serve as a reference resource for researchers studying molecular modeling, vibrational
spectroscopy, and machine learning applications in sweetness prediction. This platform
is built upon the findings obtained from the doctoral thesis titled *Characterization of
Sweeteners by Vibrational Spectroscopy and Machine Learning Techniques*, conducted by 
Çağla Çağlar under the supervision of Prof. Dr. Ozan Ünsalan, along with additional independent 
research carried out after the completion of the doctoral work. The web application was created
without any institutional funding, affiliation, or infrastructure support, representing a
fully self-initiated and independently executed scientific endeavor.

This study involves an extensive dataset of 316 molecules subjected to DFT geometry optimization, of which 
278 yielded converged frequency calculations providing infrared (IR) spectra, Raman spectra, and Raman activity 
analyses. These computational results are provided in an interactive and downloadable format, allowing researchers 
to explore molecular properties dynamically.

In addition to the DFT dataset, two compound sets were retrieved from the supplementary
materials of the SweetenersDB database, as published in the study by Ning Tang (Tang,
2023). The first set comprises 649 compounds annotated with categorical taste labels as
sweet or non-sweet, while the second includes 316 sweet compounds associated with
logarithmic sweetness intensity values. Both datasets provide SMILES representations and
compound identifiers. Since they do not contain any precomputed molecular descriptors or
structural fingerprints, all cheminformatics features, including molecular descriptors
and fingerprints, were systematically calculated within the scope of this study. These
feature sets were subsequently utilized in the development of classification and
regression models aimed at predicting sweetness perception based on molecular structure.

To improve model interpretability, this platform also incorporates feature importance
evaluations, SHAP (SHapley Additive exPlanations), and LIME (Local Interpretable
Model-Agnostic Explanations) analyses. These explainability techniques provide insights
into the key molecular descriptors influencing sweetness perception, helping researchers
better understand the underlying molecular mechanisms.

## Integration of the Molecular Descriptor Dictionary

In addition to computational data and predictive modeling tools, this platform features a
Molecular Descriptor Dictionary, which provides detailed scientific explanations of the
most influential molecular descriptors identified in the sweetness analysis. This
dictionary is an original contribution designed to help researchers interpret
descriptor-based predictions more effectively.

In the future, this dictionary will be expanded to include a comprehensive set of
molecular descriptors widely used in cheminformatics studies. It aims to become a
valuable reference for scientists working in quantitative structure-activity
relationship (QSAR) modeling, drug discovery, and computational chemistry.

## Key Features of Extreme Sweet Database

✅ **DFT-Based Quantum Chemical Data**: Optimized 3D geometries, IR, Raman spectra, and
Raman activity data  
✅ **Machine Learning Models**: Interactive classification and regression analysis for
sweetness prediction  
✅ **Model Explainability Techniques**: Feature importance analysis, SHAP, and LIME
visualizations  
✅ **Comprehensive Molecular Descriptor Dictionary**: Detailed explanations of key
descriptors, expanding in scope over time  
✅ **Fully Interactive & Downloadable Data**: All molecular structures, spectra, and model
outputs can be explored dynamically

This platform is developed to bridge the gap between computational chemistry,
cheminformatics, and machine learning in sweetness perception studies. It provides a
comprehensive, freely accessible resource for researchers, data scientists, and food
scientists interested in molecular modeling, vibrational spectroscopy, and artificial
intelligence applications in food science.

For more information, please explore the relevant sections of the database.

## References

Tang, N. (2023). Insights into Chemical Structure-Based Modeling for New Sweetener Discovery. *Foods*, *12*(13). https://doi.org/10.3390/foods12132563
